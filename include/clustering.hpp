/**
 * @file Header file containing function definitions and classes for dbscan and
 * kmeans.
 *
 * @author (last to touch it) $Author: bv $
 *
 * @version $Revision: 1.2 $
 *
 * @date $Date: 2021/02/24 $
 *
 * Contact: Thomas.Karl@ur.de
 *
 * Created on: Fr Nov 29 00:39:23 2019
 */

#pragma once

#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

template <typename T> class Data {
public:
  Data() = default;

  Data(uint dimension, uint dataSize) {
    data.resize(dimension);
    for (auto &dimension : data)
      dimension.resize(dataSize);
  }

  thrust::host_vector<thrust::host_vector<T>> data{};
};

template <typename T> class DeviceData {
public:
  DeviceData(Data<T> &hostData) {
    uint size = hostData.data[0].size();
    uint dimension = hostData.data.size();
    dataSize = size;

    if (cudaMalloc(&data, dimension * size * sizeof(T)) != cudaSuccess)
      exit(-1);

    ptrData.resize(dimension);
    for (uint i = 0; i < dimension; ++i) {
      if (cudaMemcpy(&data + (i * size), hostData.data[i].data(),
                     size * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        exit(-1);

      ptrData[i] = thrust::device_pointer_cast(data + (i * size));
    }
  }
  ~DeviceData() {
    if (cudaFree(data) != cudaSuccess)
      exit(-1);
  }

  T *data;
  uint dataSize{};
  thrust::host_vector<thrust::device_ptr<T>> ptrData{};
};

template <typename T> class Setup {
public:
  Setup() = default;
  Setup(size_t min, T eps) : minimumClusterSize(min), epsilon(eps){};
  ~Setup() = default;
  uint minimumClusterSize{0};
  T epsilon{0};
};

/**
 * @brief Reads data row-vice from file.
 *
 * @param filename Name of the file containing the data. The file
 * must store the coordinates in each row seperated by
 * whitespaces or tabulators.
 * @param data contains each coordinate of input data in one entry
 * @return returns 0 if run succsessfull, -1 otherwise
 */
template <typename T> int readDataFromFile(std::string filename, Data<T> &data);
