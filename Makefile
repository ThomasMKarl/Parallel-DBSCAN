VERSION = 1.01

CXX=g++
CFLAGS=-Wall -Wextra -std=c++14
LDFLAGS=-L ./lib -lcudart

NVCC=nvcc
BLOCKSIZE=256
NFLAGS=-arch=compute_61 -code=sm_61 -O3 -std=c++14
CULIB=-L/usr/local/cuda/lib64
THRUSTINC=-I/usr/local/cuda/include

lib: include/dbscan.h src/dbscan.cu
	mkdir lib
	$(NVCC) $(THRUSTINC) src/dbscan.cu -c $(NFLAGS) -Xcompiler "$(CFLAGS)" -DPRECISION=float   -DBLOCK_SIZE=$(BLOCKSIZE) --compiler-options '-fPIC' --shared -o lib/libdbscan.so.1
	ln -rs lib/libdbscan.so.1 lib/libdbscan.so
	$(NVCC) $(THRUSTINC) src/dbscan.cu -c $(NFLAGS) -Xcompiler "$(CFLAGS)" -DPRECISION=double  -DBLOCK_SIZE=$(BLOCKSIZE) --compiler-options '-fPIC' --shared -o lib/libdbscand.so.1
	ln -rs lib/libdbscand.so.1 lib/libdbscand.so


install: include/dbscan.h bin/dbscan bin/dbscan_d lib/libdbscan.so.1 lib/libdbscand.so.1
	make lib
	cp include/dbscan.h /usr/include
	cp lib/libdbscan.so.1  /usr/lib
	ln -s /usr/lib/libdbscan.so.1 /usr/lib/libdbscan.so
	cp lib/libdbscand.so.1 /usr/lib
	ln -s /usr/lib/libdbscand.so.1 /usr/lib/libdbscand.so

uninstall:
	rm /usr/include/dbscan.h
	rm /usr/lib/libdbscan.so.1
	rm /usr/lib/libdbscand.so.1

test: lib include/dbscan.h src/dbscan.cu src/main.cpp
	mkdir bin
	$(CXX) src/main.cpp $(LDFLAGS) $(CULIB) $(CFLAGS) -DPRECISION=float  -o bin/dbscan -ldbscan
	$(CXX) src/main.cpp $(LDFLAGS) $(CULIB) $(CFLAGS) -DPRECISION=double -o bin/dbscan_d -ldbscand
	bin/dbscan test/cluster.dat 1.5 150 > test/outlier_1.5_150.dat
	bin/dbscan test/cluster.dat 2.5 150 > test/outlier_2.5_150.dat
	bin/dbscan test/cluster.dat 1.5 200 > test/outlier_1.5_200.dat
	bin/dbscan test/cluster.dat 2.5 200 > test/outlier_2.5_200.dat

doc: doc/Doxyfile include/dbscan.h src/dbscan.cu src/main.cpp test
	make test
	doxygen doc/Doxyfile

.PHONY: clean
clean:
	rm -rf bin/dbscan
	rm -rf bin/dbscan_d
	rm -rf lib/libdbscan.so
	rm -rf lib/libdbscan.so.1
	rm -rf lib/libdbscand.so
	rm -rf lib/libdbscand.so.1
	rm -rf doc/html
	rm -rf doc/latex
