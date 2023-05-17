#!/bin/bash
set -e
nvcc -O3 -I. -lcuda -arch=compute_70 -code=sm_70 -std=c++14 \
--expt-extended-lambda --expt-relaxed-constexpr -lineinfo \
$1

echo "compile success"
rm -f report*
# nsys profile --trace=cuda,nvtx --stats=true ./a.out
./a.out