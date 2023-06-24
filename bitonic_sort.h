#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
#include <utility>

namespace bitonic {

#define BITONIC_CUDA_TRY(X)                                                                        \
  do {                                                                                             \
    cudaError_t error = (X);                                                                       \
    if (cudaSuccess != error) {                                                                    \
      cudaGetLastError();                                                                          \
      std::cerr << "CUDA error at: " << __FILE__ << __LINE__ << ": " << cudaGetErrorName(error)    \
                << " " << cudaGetErrorString(error) << std::endl;                                  \
      abort();                                                                                     \
    }                                                                                              \
  } while (0);

template <typename K> struct Compare {
  __device__ __host__ __forceinline__ bool operator()(const K &a, const K &b) const {
    return a < b;
  }
};

struct DummyTag {};

/*
From provided number of threads in thread block, number of elements processed by
one thread and array length calculates the offset and length of data block,
which is processed by current thread block.
*/
__forceinline__ __device__ void calcDataBlockLength(uint32_t &offset, uint32_t &dataBlockLength,
                                                    uint32_t arrayLength, uint32_t VT) {
  uint32_t elemsPerThreadBlock = blockDim.x * VT;
  offset = blockIdx.x * elemsPerThreadBlock;
  dataBlockLength = min(arrayLength - offset, elemsPerThreadBlock);
}

template <typename V> constexpr bool NotDummy = !std::is_same<V, DummyTag>::value;

/*
Executes one step of bitonic merge.
"OffsetGlobal" is needed to calculate correct thread index for global bitonic
merge. "TableLen" is needed for global bitonic merge to verify if elements are
still inside array boundaries.
*/
template <typename K, typename V, bool isFirstStepOfPhase, typename CompareT>
__forceinline__ __device__ void bitonicMergeStep(K *keys, V *values, uint32_t offsetGlobal,
                                                 uint32_t tableLen, uint32_t dataBlockLen,
                                                 uint32_t stride, CompareT &comp) {
  // Every thread compares and exchanges 2 elements
  for (uint32_t tx = threadIdx.x; tx < dataBlockLen >> 1; tx += blockDim.x) {
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

template <typename K, typename V>
__device__ __forceinline__ void *getSmem(K *&k, V *&v, uint32_t VT) {
  constexpr auto align = std::max(alignof(K), alignof(V));
  extern __shared__ __align__(align) char smem[];
  k = reinterpret_cast<K *>(smem);
  // v is always aligned, since blockDim.x should always be a multiple of 32
  // note: when V is DummyTag, v will point to an out-of-bound address, which won't matter because
  // we're not using it.
  v = reinterpret_cast<V *>(k + blockDim.x * VT);
}

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <typename K, typename V, typename CompareT>
__global__ void bitonicSortKernel(K *keys, V *values, uint32_t VT, uint32_t tableLen,
                                  CompareT comp) {
  K *keysTile;
  V *valuesTile;
  getSmem(keysTile, valuesTile, VT);

  uint32_t offset, dataBlockLength;
  calcDataBlockLength(offset, dataBlockLength, tableLen, VT);

  // Reads data from global to shared memory.
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += blockDim.x) {
    keysTile[tx] = keys[offset + tx];
    if (NotDummy<V>)
      valuesTile[tx] = values[offset + tx];
  }
  __syncthreads();

  // Bitonic sort PHASES
  for (uint32_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1) {
    // Bitonic merge STEPS
    uint32_t stride = subBlockSize;
    bitonicMergeStep<K, V, true>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength, stride,
                                 comp);
    stride >>= 1;
    __syncthreads();
    for (; stride > 0; stride >>= 1) {
      bitonicMergeStep<K, V, false>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength,
                                    stride, comp);
      __syncthreads();
    }
  }

  // Stores data from shared to global memory
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += blockDim.x) {
    keys[offset + tx] = keysTile[tx];
    if (NotDummy<V>)
      values[offset + tx] = valuesTile[tx];
  }
}

/*
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared
memory size.
*/
template <typename K, typename V, bool isFirstStepOfPhase, typename CompareT>
__global__ void bitonicMergeLocalKernel(K *keys, V *values, uint32_t VT, uint32_t tableLen,
                                        uint32_t step, CompareT comp) {
  K *keysTile;
  V *valuesTile;
  getSmem(keysTile, valuesTile, VT);

  bool isFirstStepOfPhaseCopy = isFirstStepOfPhase; // isFirstStepOfPhase is not editable (constant)
  uint32_t offset, dataBlockLength;
  calcDataBlockLength(offset, dataBlockLength, tableLen, VT);

  // Reads data from global to shared memory.
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += blockDim.x) {
    keysTile[tx] = keys[offset + tx];
    if (NotDummy<V>)
      valuesTile[tx] = values[offset + tx];
  }
  __syncthreads();

  // Bitonic merge
  for (uint32_t stride = 1 << (step - 1); stride > 0; stride >>= 1) {
    if (isFirstStepOfPhaseCopy) {
      bitonicMergeStep<K, V, true>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength,
                                   stride, comp);
      isFirstStepOfPhaseCopy = false;
    } else {
      bitonicMergeStep<K, V, false>(keysTile, valuesTile, 0, dataBlockLength, dataBlockLength,
                                    stride, comp);
    }
    __syncthreads();
  }

  // Stores data from shared to global memory
  for (uint32_t tx = threadIdx.x; tx < dataBlockLength; tx += blockDim.x) {
    keys[offset + tx] = keysTile[tx];
    if (NotDummy<V>)
      values[offset + tx] = valuesTile[tx];
  }
}

/*
Global bitonic merge for sections, where stride IS GREATER than max shared
memory size.
*/
template <typename K, typename V, bool isFirstStepOfPhase, typename CompareT>
__global__ void bitonicMergeGlobalKernel(K *keys, V *values, uint32_t VT, uint32_t tableLen,
                                         uint32_t step, CompareT comp) {
  uint32_t offset, dataBlockLength;
  calcDataBlockLength(offset, dataBlockLength, tableLen, VT);

  bitonicMergeStep<K, V, isFirstStepOfPhase>(keys, values, offset / 2, tableLen, dataBlockLength,
                                             1 << (step - 1), comp);
}

/*
Tests if number is power of 2.
*/
constexpr bool isPowerOfTwo(uint32_t value) { return (value != 0) && ((value & (value - 1)) == 0); }

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

/**
 * Find the largest VT (value per thread) possible for a kernel. This is mainly limited by the
 * total amount of shared memory available.
 */
template <uint32_t KVSize, typename F>
int calcBlock(F func, int max_smem, int sm_count, int &block_size, int &VT) {
  int grid;
  BITONIC_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&grid, &block_size, func));
  int block_per_sm = grid / sm_count;
  int values_per_block = max_smem / block_per_sm / KVSize;
  VT = values_per_block / block_size;
  // in the rare case when shared mem size is not enough to process 1 k-v pair per block, reduce the
  // block size.
  while (VT == 0) {
    block_size -= 32;
    if (block_size == 0) {
      throw std::invalid_argument("Key or value size too large");
    }
    VT = values_per_block / block_size;
  }
  // round down to a power of 2
  VT = 1 << (uint32_t)log2((double)VT);
  int smem_sz = VT * block_size * KVSize;
  BITONIC_CUDA_TRY(
      cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz));
  return smem_sz;
}

/**
 * Sorts data with parallel NORMALIZED BITONIC SORT.
 * For the time being, we only support sorting of up to 2^31 - 1 elements. Please be aware that this
 * function only submit kernel onto the stream you specified (or the default stream if not
 * specified) and does not wait for them to complete.
 * @param keys_first Pointer pointing to the start of the key array
 * @param values_first Pointer pointing to the start of the value array
 * @param arrayLength Length of the key array (which is also the length of the value array)
 * @param cudaStream_t Optional. The cuda stream to launch the kernels. Default to the default
 * stream (0).
 * @param comp Optional. Custom comparator function. Default comparator uses the < operator
 */
template <typename K, typename V, typename CompareT = Compare<K>>
void sort_by_key(K *keys_first, V *values_first, uint32_t arrayLength, cudaStream_t stream = 0,
                 CompareT comp = {}) {
  constexpr auto KVSize = sizeof(K) + (NotDummy<V> ? sizeof(V) : 0);
  static int elemsBS = -1, elemsLM, elemsGM = 2, threadsBS, threadsLM, threadsGM;
  if (elemsBS == -1) {
    int device_id;
    BITONIC_CUDA_TRY(cudaGetDevice(&device_id));
    int max_smem, resv, sm_count;
    BITONIC_CUDA_TRY(
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
    BITONIC_CUDA_TRY(
        cudaDeviceGetAttribute(&resv, cudaDevAttrReservedSharedMemoryPerBlock, device_id));
    BITONIC_CUDA_TRY(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

    max_smem -= resv;

    (void)calcBlock<KVSize>(bitonicSortKernel<K, V, CompareT>, max_smem, sm_count, threadsBS,
                            elemsBS);
    int smem_sz = calcBlock<KVSize>(bitonicMergeLocalKernel<K, V, true, CompareT>, max_smem,
                                    sm_count, threadsLM, elemsLM);
    BITONIC_CUDA_TRY(cudaFuncSetAttribute(bitonicMergeLocalKernel<K, V, false, CompareT>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz));
    int grid;
    BITONIC_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
        &grid, &threadsGM, bitonicMergeGlobalKernel<K, V, true, CompareT>));
    std::cerr << elemsBS << " " << elemsLM << " " << elemsGM << " " << threadsBS << " " << threadsLM
              << " " << threadsGM << std::endl;
  }

  uint32_t elemsPerBlockBS = threadsBS * elemsBS;
  uint32_t elemsPerBlockLM = threadsLM * elemsLM;
  // Sorts blocks of input data with bitonic sort
  bitonicSortKernel<K, V>
      <<<(arrayLength - 1) / elemsPerBlockBS + 1, threadsBS, elemsPerBlockBS * KVSize, stream>>>(
          keys_first, values_first, elemsBS, arrayLength, comp);
  uint32_t arrayLenPower2 = nextPowerOf2(arrayLength);
  // Number of phases, which can be executed in shared memory (stride is lower
  // than shared memory size)
  uint32_t phasesBS = log2((double)min(arrayLenPower2, elemsPerBlockBS));
  uint32_t phaseLM = log2((double)min(arrayLenPower2, elemsPerBlockLM));
  uint32_t phasesAll = log2((double)arrayLenPower2);

  auto gridLM = (arrayLength - 1) / elemsPerBlockLM + 1;
  auto gridGM = (arrayLength - 1) / (threadsGM * elemsGM) + 1;
  auto smemLM = elemsPerBlockLM * KVSize;
  // Bitonic merge
  for (uint32_t phase = phasesBS + 1; phase <= phasesAll; phase++) {
    uint32_t step = phase;

    // Merges array, if data blocks are larger than shared memory size. It executes
    // only one STEP of one PHASE per kernel launch.
    while (step > phaseLM) {
      if (phase == step) {
        bitonicMergeGlobalKernel<K, V, true><<<gridGM, threadsGM, 0, stream>>>(
            keys_first, values_first, elemsGM, arrayLength, step, comp);
      } else {
        bitonicMergeGlobalKernel<K, V, false><<<gridGM, threadsGM, 0, stream>>>(
            keys_first, values_first, elemsGM, arrayLength, step, comp);
      }
      step--;
    }

    // Merges array when stride is lower than shared memory size. It executes all
    // remaining STEPS of current PHASE.
    if (phase == step) {
      bitonicMergeLocalKernel<K, V, true><<<gridLM, threadsLM, smemLM, stream>>>(
          keys_first, values_first, elemsLM, arrayLength, step, comp);
    } else {
      bitonicMergeLocalKernel<K, V, false><<<gridLM, threadsLM, smemLM, stream>>>(
          keys_first, values_first, elemsLM, arrayLength, step, comp);
    }
  }
}

template <typename K, typename CompareT = Compare<K>>
void sort(K *keys_first, uint32_t arrayLength, cudaStream_t stream = 0, CompareT comp = {}) {
  sort_by_key<K, DummyTag, CompareT>(keys_first, nullptr, arrayLength, stream, comp);
}

} // namespace bitonic

