#!/bin/bash
set -e
nvcc -O2 -I. -lcuda -arch=compute_70 -code=sm_70 -std=c++14 \
--expt-extended-lambda --expt-relaxed-constexpr -lineinfo \
tests.cu

echo "compile success"
rm -f report*
# nsys profile --trace=cuda,nvtx --stats=true ./a.out
./a.out