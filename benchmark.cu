#include "bitonic_sort.h"
#include <algorithm>
#include <chrono>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

template <typename T> struct Compare {
  __host__ __device__ bool operator()(const T &a, const T &b) const { return a < b; }
};

template <typename T> void test(std::string name) {
  std::ofstream out(name);

  std::mt19937 rand(std::chrono::steady_clock::now().time_since_epoch().count());
  uint32_t num_items = 1000;

  thrust::device_vector<T> d_keys;
  thrust::device_vector<T> d_values;
  thrust::device_vector<T> d_keys_copy;
  thrust::device_vector<T> d_values_copy;
  thrust::device_vector<char> _cub_ws(8590921984L / 2 * sizeof(T) / sizeof(uint32_t));
  auto cub_ws = _cub_ws.data().get();
  thrust::default_random_engine eng(std::chrono::steady_clock::now().time_since_epoch().count());
  for (int i = 0; i < 20; i++) {
    d_keys.resize(num_items);
    d_values.resize(num_items);

    thrust::sequence(d_keys.begin(), d_keys.end());
    thrust::sequence(d_values.begin(), d_values.end());
    thrust::shuffle(d_keys.begin(), d_keys.end(), eng);

    d_keys_copy = d_keys;
    d_values_copy = d_values;

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    size_t temp_sz;
    cub::DeviceMergeSort::SortPairs(nullptr, temp_sz, d_keys_copy.begin(), d_values_copy.begin(),
                                    num_items, Compare<T>());
    // std::cout << temp_sz << std::endl;
    cub::DeviceMergeSort::SortPairs(cub_ws, temp_sz, d_keys_copy.begin(), d_values_copy.begin(),
                                    num_items, Compare<T>());
    cudaDeviceSynchronize();
    auto merge_sort_time = (std::chrono::steady_clock::now() - start).count() * 1e-6;

    d_keys_copy = d_keys;
    d_values_copy = d_values;

    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();
    bitonic::sort_by_key(d_keys_copy.data().get(), d_values_copy.data().get(), num_items);
    cudaDeviceSynchronize();
    auto bitonic_sort_time = (std::chrono::steady_clock::now() - start).count() * 1e-6;

    cub::DoubleBuffer<T> k_buff(d_keys.data().get(), d_keys_copy.data().get());
    cub::DoubleBuffer<T> v_buff(d_values.data().get(), d_values_copy.data().get());

    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();
    cub::DeviceRadixSort::SortPairs(nullptr, temp_sz, k_buff, v_buff, num_items);
    cub::DeviceRadixSort::SortPairs(cub_ws, temp_sz, k_buff, v_buff, num_items);
    cudaDeviceSynchronize();
    auto radix_sort_time = (std::chrono::steady_clock::now() - start).count() * 1e-6;

    out << ((float)num_items) / 1000000 << ',' << merge_sort_time << ',' << radix_sort_time << ','
        << bitonic_sort_time << std::endl;
    num_items *= 2;
  }
}

int main() {
  test<uint32_t>("result_32.csv");
  test<uint64_t>("result_64.csv");
}
