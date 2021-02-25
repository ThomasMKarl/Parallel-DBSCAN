VERSION = 1.01

CXX=g++
CFLAGS=-Wall -Wextra -std=c++14
LDFLAGS=-L lib/ -lcudart

NVCC=nvcc
BLOCKSIZE=256
NFLAGS=-arch=compute_61 -code=sm_61 -O3 -std=c++14
CULIB=-L/usr/local/cuda/lib64
THRUSTINC=-I/usr/local/cuda/include

lib: include/dbscan.h src/dbscan.cu include/kmeans.h src/kmeans.cu
	mkdir -p lib
	$(NVCC) $(THRUSTINC) -I include/ src/dbscan.cu -c $(NFLAGS) -Xcompiler "$(CFLAGS)" -DBLOCK_SIZE=$(BLOCKSIZE) --compiler-options '-fPIC' --shared -o lib/libdbscan.so
	$(NVCC) $(THRUSTINC) -I include/ src/kmeans.cu -c $(NFLAGS) -Xcompiler "$(CFLAGS)" -DBLOCK_SIZE=$(BLOCKSIZE) --compiler-options '-fPIC' --shared -o lib/libkmeans.so


.PHONY: bin
bin:
	make lib
	mkdir -p bin
	$(CXX) -I include/ src/db_main.cpp $(LDFLAGS) $(CULIB) $(CFLAGS) -o bin/dbscan -ldbscan
	$(CXX) -I include/ src/km_main.cpp $(LDFLAGS) $(CULIB) $(CFLAGS) -o bin/kmeans -lkmeans

install: include/dbscan.h bin/dbscan lib/libdbscan.so.1 lib/libdbscand.so.1 include/kmeans.h bin/kmeans lib/libkmeans.so.1 lib/libkmeans.so.1
	make lib
	cp include/dbscan.h /usr/include
	cp include/kmeans.h /usr/include
	cp lib/libdbscan.so.1  /usr/lib
	cp lib/libkmeans.so.1  /usr/lib
	ln -s /usr/lib/libdbscan.so.1 /usr/lib/libdbscan.so
	ln -s /usr/lib/libkmeans.so.1 /usr/lib/libkmeans.so

uninstall:
	rm /usr/include/dbscan.h
	rm /usr/lib/libdbscan.so.1
	rm /usr/lib/libdbscan.so
	rm /usr/include/kmeans.h
	rm /usr/lib/libkmeans.so.1
	rm /usr/lib/libkmeans.so


.PHONY: test
test: lib include/dbscan.h src/dbscan.cu src/db_main.cpp include/kmeans.h src/kmeans.cu src/km_main.cpp
	make bin
	bin/dbscan test/cluster.dat 1.5 150 > test/outlier_1.5_150.dat
	bin/dbscan test/cluster.dat 2.5 150 > test/outlier_2.5_150.dat
	bin/dbscan test/cluster.dat 1.5 200 > test/outlier_1.5_200.dat
	bin/dbscan test/cluster.dat 2.5 200 > test/outlier_2.5_200.dat
	#bin/kmeans test/cluster.dat 4 500 > test/kmeans.dat

.PHONY: doc
doc: doc/Doxyfile include/dbscan.h src/dbscan.cu src/db_main.cpp include/kmeans.h src/kmeans.cu src/km_main.cpp
	doxygen doc/Doxyfile

.PHONY: clean
clean:
	rm -rf bin/dbscan
	rm -rf bin/kmeans
	rm -rf lib/libdbscan.so
	rm -rf lib/libkmeans.so
	rm -rf doc/html
	rm -rf doc/latex
