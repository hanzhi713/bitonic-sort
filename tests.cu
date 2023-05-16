#include "bitonic_sort.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <thrust/device_vector.h>

int main() {
  std::mt19937 rand(std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> dist(10240, 10240000);

  for (int i = 0; i < 100; i++) {
    int num_items = dist(rand);
    thrust::host_vector<uint32_t> keys(num_items);
    thrust::host_vector<uint32_t> values(num_items);
    for (int j = 0; j < num_items; j++) {
      keys[j] = j;
      values[j] = j;
    }
    std::shuffle(keys.begin(), keys.end(), rand);
    thrust::device_vector<uint32_t> d_keys = keys;
    thrust::device_vector<uint32_t> d_values = values;

    thrust::device_vector<uint32_t> d_keys_copy = d_keys;
    thrust::device_vector<uint32_t> d_values_copy = d_values;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
    bitonic::sort_by_key(d_keys_copy.data().get(), d_values_copy.data().get(), num_items);
    bitonic::sort(d_keys_copy.data().get(), num_items);

    if (d_keys != d_keys_copy) {
      std::cout << "test " << i << " failed" << std::endl;
      thrust::host_vector<uint32_t> h_keys = d_values;
      thrust::host_vector<uint32_t> h_keys2 = d_values_copy;

      for (int j = 0; j < num_items; j++) {
        if (h_keys2[j] != h_keys[j])
          std::cout << h_keys[j] << "," << h_keys2[j] << '\n';
      }
      std::cout << std::endl;
    } else {
      std::cout << "test " << i << " passed" << std::endl;
    }
  }
}