#include "bitonic_sort.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

void sort_by_key_test() {
  std::mt19937 rand(std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> dist(1, 102400000);

  for (int i = 0; i < 10; i++) {
    int num_items = dist(rand);
    thrust::device_vector<uint32_t> d_keys(num_items);
    thrust::device_vector<uint32_t> d_values(num_items);

    thrust::sequence(d_keys.begin(), d_keys.end());
    thrust::sequence(d_values.begin(), d_values.end());

    thrust::default_random_engine eng(std::chrono::steady_clock::now().time_since_epoch().count());
    thrust::shuffle(d_keys.begin(), d_keys.end(), eng);

    thrust::device_vector<uint32_t> d_keys_copy = d_keys;
    thrust::device_vector<uint32_t> d_values_copy = d_values;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
    bitonic::sort_by_key(d_keys_copy.data().get(), d_values_copy.data().get(), num_items);

    if (d_keys != d_keys_copy || d_values != d_values_copy) {
      std::cout << "test " << i << " failed" << std::endl;
    } else {
      std::cout << "test " << i << " passed" << std::endl;
    }
  }
}

struct __align__(4) CustomData {
  uint8_t data[4];
  bool operator==(const CustomData &other) const {
    uint32_t td, od;
    memcpy(&td, data, sizeof(uint32_t));
    memcpy(&od, other.data, sizeof(uint32_t));
    return td == od;
  }
};

struct Compare {
  int idx;
  Compare(int idx) : idx(idx) {}
  __device__ __host__ bool operator()(CustomData &a, CustomData &b) const {
    return a.data[idx] < b.data[idx];
  }
};

void sort_custom_compare_stable_test() {
  std::mt19937 rand(std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> dist(1, 102400000);

  for (int i = 0; i < 10; i++) {
    int num_items = dist(rand);
    thrust::device_vector<CustomData> d_keys(num_items);

    auto cast = thrust::device_pointer_cast(reinterpret_cast<uint32_t *>(d_keys.data().get()));
    thrust::sequence(cast, cast + num_items);
    thrust::default_random_engine eng(std::chrono::steady_clock::now().time_since_epoch().count());
    thrust::shuffle(d_keys.begin(), d_keys.end(), eng);

    thrust::device_vector<CustomData> d_keys_copy = d_keys;
    for (int j = 0; j < 4; j++) {
      thrust::stable_sort(d_keys.begin(), d_keys.end(), Compare(j));
      bitonic::sort(d_keys_copy.data().get(), num_items, 0, Compare(j));
    }

    if (d_keys != d_keys_copy) {
      std::cout << "test " << i << " failed" << std::endl;
    } else {
      std::cout << "test " << i << " passed" << std::endl;
    }
  }
}

template <int sz> struct SzSt {
  char data[sz];
};

template <int sz> struct CpSzSt {
  __device__ __host__ __forceinline__ bool operator()(const SzSt<sz> &a, const SzSt<sz> &b) const {
    return a.data[0] < b.data[0];
  }
};

int main() {
  sort_by_key_test();
  sort_custom_compare_stable_test();
}