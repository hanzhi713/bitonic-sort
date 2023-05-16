#include <cstdint>
#include <utility>
#include <type_traits>

namespace bitonic {

template <typename InputIt> struct Compare {
  using val = decltype(*std::declval<InputIt>()) &;
  __device__ __host__ __forceinline__ bool operator()(val a, val b) const { return a < b; }
};

struct DummyTag {};

/*
From provided number of threads in thread block, number of elements processed by
one thread and array length calculates the offset and length of data block,
which is processed by current thread block.
*/
template <uint32_t NT, uint32_t VT>
__forceinline__ __device__ void calcDataBlockLength(uint32_t &offset, uint32_t &dataBlockLength, uint32_t arrayLength) {
  uint32_t elemsPerThreadBlock = NT * VT;
  offset = blockIdx.x * elemsPerThreadBlock;
  dataBlockLength = min(arrayLength - offset, elemsPerThreadBlock);
}

template <typename V> inline constexpr bool NotDummy = !std::is_same<V, DummyTag *>::value;

/*
Executes one step of bitonic merge.
"OffsetGlobal" is needed to calculate correct thread index for global bitonic
merge. "TableLen" is needed for global bitonic merge to verify if elements are
still inside array boundaries.
*/
template <typename K, typename V, uint32_t NT, bool isFirstStepOfPhase, typename Compare>
__forceinline__ __device__ void bitonicMergeStep(K keys, V values, uint32_t offsetGlobal, uint32_t tableLen,
                                                 uint32_t dataBlockLen, uint32_t stride, Compare &comp) {
  // Every thread compares and exchanges 2 elements
  for (uint32_t tx = threadIdx.x; tx < dataBlockLen >> 1; tx += NT) {
    uint32_t indexThread = offsetGlobal + tx;
    uint32_t offset = stride;
    auto sm1 = stride - 1;

    // In NORMALIZED bitonic sort, first STEP of every PHASE demands different
    // offset than all other STEPS. Also, in first step of every phase, offset
    // sizes are generated in ASCENDING order (normalized bitnic sort requires
    // DESCENDING order). Because of that, we can break the loop if index +
    // offset >= length (bellow). If we want to generate offset sizes in
    // ASCENDING order, than thread indexes inside every sub-block have to be
    // reversed.
    if (isFirstStepOfPhase) {
      auto mod = indexThread & sm1;
      offset = 2 * mod + 1;
      indexThread = (indexThread & ~sm1) + (sm1 - mod);
    }

    uint32_t index = indexThread * 2 - (indexThread & sm1);
    if (index + offset >= tableLen) {
      break;
    }

    auto k1 = keys[index];
    auto k2 = keys[index + offset];
    if (!comp(k1, k2)) {
      keys[index + offset] = k1;
      keys[index] = k2;

      if (NotDummy<V>) {
        auto tmp = values[index];
        values[index] = values[index + offset];
        values[index + offset] = tmp;
      }
    }
  }
}

template <typename T> using pointed_t = std::remove_reference_t<decltype(*std::declval<T>())>;

template <typename V, uint32_t NT, uint32_t VT> __device__ __forceinline__ pointed_t<V> *getValueSmem() {
  __shared__ pointed_t<V> valuesTile[NT * VT];
  return valuesTile;
}

template <uint32_t NT, uint32_t VT> __device__ __forceinline__ DummyTag *getValueSmem<DummyTag *>() { return nullptr; }

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <typename K, typename V, uint32_t NT, uint32_t VT, typename Compare>
__global__ void bitonicSortKernel(K keys, V values, uint32_t tableLen, Compare comp) {
  __shared__ pointed_t<K> keysTile[NT * VT];
  auto valuesTile = getValueSmem<V, NT, VT>();

  uint32_t offset, dataBlockLength;
  calcDataBlockLength<NT, VT>(offset, dataBlockLength, tableLen);

  // Reads data from global to shared memory.
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += NT) {
    keysTile[tx] = keys[offset + tx];
    if (NotDummy<V>)
      valuesTile[tx] = values[offset + tx];
  }
  __syncthreads();

  // Bitonic sort PHASES
  for (uint32_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1) {
    // Bitonic merge STEPS
    for (uint32_t stride = subBlockSize; stride > 0; stride >>= 1) {
      if (stride == subBlockSize) {
        bitonicMergeStep<K, V, NT, true>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength, stride, comp);
      } else {
        bitonicMergeStep<K, V, NT, false>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength, stride, comp);
      }

      __syncthreads();
    }
  }

  // Stores data from shared to global memory
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += NT) {
    keys[offset + tx] = keysTile[tx];
    if (NotDummy<V>)
      values[offset + tx] = valuesTile[tx];
  }
}

/*
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared
memory size.
*/
template <typename K, typename V, uint32_t NT, uint32_t VT, bool isFirstStepOfPhase, typename Compare>
__global__ void bitonicMergeLocalKernel(K keys, V values, uint32_t tableLen, uint32_t step, Compare comp) {
  __shared__ pointed_t<K> keysTile[NT * VT];
  auto valuesTile = getValueSmem<V, NT, VT>();

  bool isFirstStepOfPhaseCopy = isFirstStepOfPhase; // isFirstStepOfPhase is not editable (constant)
  uint32_t offset, dataBlockLength;
  calcDataBlockLength<NT, VT>(offset, dataBlockLength, tableLen);

  // Reads data from global to shared memory.
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += NT) {
    keysTile[tx] = keys[offset + tx];
    if (NotDummy<V>)
      valuesTile[tx] = values[offset + tx];
  }
  __syncthreads();

  // Bitonic merge
  for (uint32_t stride = 1 << (step - 1); stride > 0; stride >>= 1) {
    if (isFirstStepOfPhaseCopy) {
      bitonicMergeStep<K, V, NT, true>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength, stride, comp);
      isFirstStepOfPhaseCopy = false;
    } else {
      bitonicMergeStep<K, V, NT, false>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength, stride, comp);
    }
    __syncthreads();
  }

  // Stores data from shared to global memory
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += NT) {
    keys[offset + tx] = keysTile[tx];
    if (NotDummy<V>)
      values[offset + tx] = valuesTile[tx];
  }
}

/*
Global bitonic merge for sections, where stride IS GREATER than max shared
memory size.
*/
template <typename K, typename V, uint32_t NT, uint32_t VT, bool isFirstStepOfPhase, typename Compare>
__global__ void bitonicMergeGlobalKernel(K keys, V values, uint32_t tableLen, uint32_t step, Compare comp) {
  uint32_t offset, dataBlockLength;
  calcDataBlockLength<NT, VT>(offset, dataBlockLength, tableLen);

  bitonicMergeStep<K, V, NT, isFirstStepOfPhase>(keys, values, offset / 2, tableLen, dataBlockLength, 1 << (step - 1),
                                                 comp);
}

/*
Tests if number is power of 2.
*/
bool isPowerOfTwo(uint32_t value) { return (value != 0) && ((value & (value - 1)) == 0); }

/*
Return the next power of 2 for provided value. If value is already power of 2,
it returns value.
*/
uint32_t nextPowerOf2(uint32_t value) {
  if (isPowerOfTwo(value)) {
    return value;
  }

  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value++;

  return value;
}

/*
Sorts data with parallel NORMALIZED BITONIC SORT.
*/
template <uint32_t threadsBitonicSort = 128, uint32_t elemsBitonicSort = 4, uint32_t threadsGlobalMerge = 256,
          uint32_t elemsGlobalMerge = 4, uint32_t threadsLocalMerge = 256, uint32_t elemsLocalMerge = 4, typename K,
          typename V, typename Compare = Compare<K>>
void sort_by_key(K d_keys, V d_values, uint32_t arrayLength, Compare comp = {}) {
  uint32_t arrayLenPower2 = nextPowerOf2(arrayLength);

  uint32_t elemsPerBlockBitonicSort = threadsBitonicSort * elemsBitonicSort;
  uint32_t elemsPerBlockMergeLocal = threadsLocalMerge * elemsLocalMerge;

  // Number of phases, which can be executed in shared memory (stride is lower
  // than shared memory size)
  uint32_t phasesBitonicSort = log2((double)min(arrayLenPower2, elemsPerBlockBitonicSort));
  uint32_t phasesMergeLocal = log2((double)min(arrayLenPower2, elemsPerBlockMergeLocal));
  uint32_t phasesAll = log2((double)arrayLenPower2);

  // Sorts blocks of input data with bitonic sort
  uint32_t elemsPerThreadBlock = threadsBitonicSort * elemsBitonicSort;

  bitonicSortKernel<K, V, threadsBitonicSort, elemsBitonicSort>
      <<<(arrayLength - 1) / elemsPerThreadBlock + 1, threadsBitonicSort>>>(d_keys, d_values, arrayLength, comp);

  // Bitonic merge
  for (uint32_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++) {
    uint32_t step = phase;

    // Merges array, if data blocks are larger than shared memory size. It executes
    // only one STEP of one PHASE per kernel launch.
    while (step > phasesMergeLocal) {
      if (phase == step) {
        bitonicMergeGlobalKernel<K, V, threadsGlobalMerge, elemsGlobalMerge, true>
            <<<(arrayLength - 1) / (threadsGlobalMerge * elemsGlobalMerge) + 1, threadsGlobalMerge>>>(
                d_keys, d_values, arrayLength, step, comp);
      } else {
        bitonicMergeGlobalKernel<K, V, threadsGlobalMerge, elemsGlobalMerge, false>
            <<<(arrayLength - 1) / (threadsGlobalMerge * elemsGlobalMerge) + 1, threadsGlobalMerge>>>(
                d_keys, d_values, arrayLength, step, comp);
      }
      step--;
    }

    // Merges array when stride is lower than shared memory size. It executes all
    // remaining STEPS of current PHASE.
    if (phase == step) {
      bitonicMergeLocalKernel<K, V, threadsLocalMerge, elemsLocalMerge, true>
          <<<(arrayLength - 1) / (threadsLocalMerge * elemsLocalMerge) + 1, threadsLocalMerge>>>(
              d_keys, d_values, arrayLength, step, comp);
    } else {
      bitonicMergeLocalKernel<K, V, threadsLocalMerge, elemsLocalMerge, false>
          <<<(arrayLength - 1) / (threadsLocalMerge * elemsLocalMerge) + 1, threadsLocalMerge>>>(
              d_keys, d_values, arrayLength, step, comp);
    }
  }
}

template <uint32_t threadsBitonicSort = 128, uint32_t elemsBitonicSort = 4, uint32_t threadsGlobalMerge = 256,
          uint32_t elemsGlobalMerge = 4, uint32_t threadsLocalMerge = 256, uint32_t elemsLocalMerge = 4, typename K,
          typename Compare = Compare<K>>
void sort(K d_keys, uint32_t arrayLength, Compare comp = {}) {
  sort_by_key<threadsBitonicSort, elemsBitonicSort, threadsGlobalMerge, elemsGlobalMerge, threadsLocalMerge,
              elemsLocalMerge, K, DummyTag *, Compare>(d_keys, nullptr, arrayLength, comp);
}

}