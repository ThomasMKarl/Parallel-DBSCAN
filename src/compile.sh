export bs=256

echo 'Blocksize: ' $bs
echo ''

echo 'single precision: '
echo 'nvcc -arch=compute_61 -code=sm_61 dbscan.cu -c -O3 -std=c++14 -Xcompiler "-Wall -Wextra -std=c++14 -fopenmp" -DPRECISION=float  -DBLOCK_SIZE=$bs'
nvcc -arch=compute_61 -code=sm_61 dbscan.cu -c -O3 -std=c++14 -Xcompiler "-Wall -Wextra -std=c++14 -fopenmp" -DPRECISION=float  -DBLOCK_SIZE=$bs
echo 'g++ main.cpp dbscan.o -L/usr/local/cuda/lib64 -lcudart -O3 -Wall -Wextra -pedantic -std=c++14 -DPRECISION=float -o dbscan'
g++ main.cpp dbscan.o -L/usr/local/cuda/lib64 -lcudart -O3 -Wall -Wextra -pedantic -std=c++14 -DPRECISION=float -o dbscan

echo ''

echo 'double precision: '
echo 'nvcc -arch=compute_61 -code=sm_61 dbscan.cu -c -O3 -std=c++14 -Xcompiler "-Wall -Wextra -std=c++14 -fopenmp" -DPRECISION=double -DBLOCK_SIZE=$bs'
nvcc -arch=compute_61 -code=sm_61  dbscan.cu -c -O3 -std=c++14 -Xcompiler "-Wall -Wextra -std=c++14 -fopenmp" -DPRECISION=double -DBLOCK_SIZE=$bs
echo 'g++ main.cpp dbscan.o -L/usr/local/cuda/lib64 -lcudart -O3 -Wall -Wextra -pedantic -std=c++14 -DPRECISION=double -o dbscan_d'
g++ main.cpp dbscan.o -L/usr/local/cuda/lib64 -lcudart -O3 -Wall -Wextra -pedantic -std=c++14 -DPRECISION=double -o dbscan_d

rm -f dbscan.o
rm -f *~
rm -f \#*\#

unset bs

echo ''
echo 'done.'
