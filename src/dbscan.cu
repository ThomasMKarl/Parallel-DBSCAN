#include "dbscan.hpp"

template <typename T>
__host__ T calculateDistance(uint indexFrom, uint indexTo, Data<T> &data) {
  uint dimension = data.data.size();
  T distance{0.0f};
  T coordinateDistance{0.0f};
  for (const auto &coordinate : data.data) {
    coordinateDistance = coordinate[indexFrom] - coordinate[indexTo];

    coordinateDistance *= coordinateDistance;

    distance += coordinateDistance;
  }

  return static_cast<T>(std::sqrt(distance));
}
template __host__ float calculateDistance(uint indexFrom, uint indexTo,
                                          Data<float> &data);
template __host__ double calculateDistance(uint indexFrom, uint indexTo,
                                           Data<double> &data);

/////////////////////////////////////////////////////////////////////
template <typename T>
__device__ T calculateDistance(uint indexFrom, uint indexTo, uint dimension,
                               uint dataSize, T *data) {
  T distance = 0;
  T coordinateDistance;
  uint offset;
  for (uint coordinate = 0; coordinate < dimension; coordinate++) {
    offset = coordinate * dataSize;
    coordinateDistance = data[indexFrom + offset] - data[indexTo + offset];

    coordinateDistance *= coordinateDistance;

    distance += coordinateDistance;
  }

  return static_cast<T>(std::sqrt(distance));
}
template __device__ float calculateDistance(uint indexFrom, uint indexTo,
                                            uint dimension, uint dataSize,
                                            float *data);
template __device__ double calculateDistance(uint indexFrom, uint indexTo,
                                             uint dimension, uint dataSize,
                                             double *data);

/////////////////////////////////////////////////////////////////////
template <typename T>
int gdbscan(Data<T> &data, Setup<T> &setup, Data<T> &cluster) {
  uint dimension = data.data.size();
  uint dataSize = data.data[0].size();

  auto &clusterOffsets = cluster.data[1];
  auto &clusterIndices = cluster.data[0];

  thrust::host_vector<uint> clusterSizes(dataSize);
  uint numberOfHits;
  for (uint index = 0; index < dataSize; ++index) {
    numberOfHits = 0;
    for (uint other = 0; other < dataSize; ++other)
      if (calculateDistance(index, other, data) < setup.epsilon)
        numberOfHits++;

    if (numberOfHits > setup.minimumClusterSize)
      clusterSizes[index] = numberOfHits - 1;
    else
      clusterSizes[index] = 0;
  }

  uint offset = 0;
  clusterOffsets.resize(dataSize);
  for (uint i = 1; i < dataSize; ++i) {
    clusterOffsets[i] = clusterSizes[i - 1] + offset;
    offset += clusterOffsets[i];
  }

  uint fullClusterSize =
      std::reduce(std::begin(clusterSizes), std::end(clusterSizes), 0);

  clusterIndices.resize(fullClusterSize);
  for (uint index = 0; index < dataSize; ++index) {
    numberOfHits = 0;
    for (uint other = 0; other < dataSize; ++other) {
      if (calculateDistance(index, other, data) < setup.epsilon &&
          index != other) {
        clusterIndices[clusterOffsets[index] + numberOfHits] = other;
        numberOfHits++;
        if (numberOfHits == clusterSizes[index])
          break;
      }
    }
  }

  return EXIT_SUCCESS;
}
template int gdbscan(Data<float> &data, Setup<float> &setup,
                     Data<float> &cluster);
template int gdbscan(Data<double> &data, Setup<double> &setup,
                     Data<double> &cluster);

/////////////////////////////////////////////////////////////////////
template <typename T>
int gdbscan_cuda(Data<T> &data, Setup<T> &setup, Data<uint> &cluster) {
  uint dimension = data.data.size();
  uint dataSize = data.data[0].size();

  DeviceData<T> deviceData(data);

  uint threadsPerBlock = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

  thrust::device_vector<uint> clusterSizes{dataSize};
  computeVertices_kernel<<<BLOCK_SIZE, threadsPerBlock, 0>>>(
      setup.epsilon, setup.minimumClusterSize,
      thrust::raw_pointer_cast(clusterSizes.data()), deviceData.data, dataSize,
      dimension);

  if (cudaDeviceSynchronize() != cudaSuccess)
    return EXIT_FAILURE;

  thrust::device_vector<uint> clusterStartIndices{dataSize};
  thrust::exclusive_scan(thrust::device, clusterSizes.begin(),
                         clusterSizes.end(), clusterStartIndices.begin());

  uint fullClusterSize = thrust::reduce(
      thrust::device, clusterSizes.begin(), clusterSizes.end());

  thrust::device_vector<uint> clusterIndices{fullClusterSize};
  computeCluster_kernel<<<BLOCK_SIZE, threadsPerBlock, 0>>>(
      setup.epsilon, setup.minimumClusterSize,
      thrust::raw_pointer_cast(clusterSizes.data()),
      thrust::raw_pointer_cast(clusterStartIndices.data()),
      thrust::raw_pointer_cast(clusterIndices.data()), deviceData.data,
      dataSize, dimension);

  if (cudaDeviceSynchronize() != cudaSuccess)
    return EXIT_FAILURE;

  cluster.data[0] = clusterIndices;
  cluster.data[1] = clusterStartIndices;

  return EXIT_SUCCESS;
}
template int gdbscan_cuda(Data<float> &data, Setup<float> &setup,
                          Data<uint> &cluster);
template int gdbscan_cuda(Data<double> &data, Setup<double> &setup,
                          Data<uint> &cluster);

/////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void computeVertices_kernel(T epsilon, uint minimumClusterSize,
                                       uint *clusterSizes, T *data,
                                       uint dataSize, uint dimension) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint numberOfHits = 0;
  if (idx < dataSize) {
    for (uint other = 0; other < dataSize; ++other)
      if (calculateDistance(idx, other, dimension, dataSize, data) < epsilon)
        numberOfHits++;

    if (numberOfHits > minimumClusterSize)
      clusterSizes[idx] = numberOfHits - 1;
    else
      clusterSizes[idx] = 0;
  }
}
template __global__ void computeVertices_kernel(float epsilon,
                                                uint minimumClusterSize,
                                                uint *clusterSizes, float *data,
                                                uint dataSize, uint dimension);
template __global__ void computeVertices_kernel(double epsilon,
                                                uint minimumClusterSize,
                                                uint *clusterSizes,
                                                double *data, uint dataSize,
                                                uint dimension);

/////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void
computeCluster_kernel(T epsilon, uint minimumClusterSize, uint *clusterSizes,
                      uint *clusterStartIndices, uint *clusterIndices, T *data,
                      uint dataSize, uint dimension) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint offset = 0;
  if (idx < dataSize * dimension && clusterSizes[idx] != 0) {
    for (uint other = 0; other < dataSize; ++other) {
      if (calculateDistance(idx, other, dimension, dataSize, data) < epsilon &&
          idx != other) {
        clusterIndices[clusterStartIndices[idx] + offset] = other;
        offset++;
        if (offset == clusterSizes[idx])
          break;
      }
    }
  }
}
template __global__ void
computeCluster_kernel(float epsilon, uint minimumClusterSize,
                      uint *clusterSizes, uint *clusterStartIndices,
                      uint *clusterIndices, float *data, uint dataSize,
                      uint dimension);
template __global__ void
computeCluster_kernel(double epsilon, uint minimumClusterSize,
                      uint *clusterSizes, uint *clusterStartIndices,
                      uint *clusterIndices, double *data, uint dataSize,
                      uint dimension);

/////////////////////////////////////////////////////////////////////
template <typename T>
int readDataFromFile(std::string filename, Data<T> &data) {
  std::ifstream input(filename);
  if (!input) {
    printf("ERROR: could not read %s.\n", filename.c_str());
    return EXIT_FAILURE;
  }

  std::string word;
  uint index = 0;
  uint dimension = 0;
  uint all = data.data[0].size() * data.data.size();

  while (input >> word) {
    data.data[dimension][index] = (T)std::stod(word);

    if (++dimension == data.data.size()) {
      dimension = 0;
      index++;
    }

    if (index == all - 1)
      break;
  }

  return EXIT_SUCCESS;
}
template int readDataFromFile(std::string filename, Data<float> &data);
template int readDataFromFile(std::string filename, Data<double> &data);

/////////////////////////////////////////////////////////////////////
int printStartIndices(thrust::host_vector<uint> &clusterStartIndices) {
  for (uint i = 0; i < clusterStartIndices.size(); ++i)
    printf("%d ", clusterStartIndices[i]);
  printf("\n");

  return EXIT_SUCCESS;
}

/////////////////////////////////////////////////////////////////////
template <typename T>
int printOutliers(thrust::host_vector<uint> &outlierIndices, Data<T> &data) {
  uint dimension = data.data.size();

  for (uint i = 0; i < outlierIndices.size(); ++i) {
    for (const auto &coordinate : data.data) {
      printf("%f ", coordinate[outlierIndices[i]]);
    }
    printf("\n");
  }
  printf("\n");

  return EXIT_SUCCESS;
}
template int printOutliers(thrust::host_vector<uint> &outlierIndices,
                           Data<float> &data);
template int printOutliers(thrust::host_vector<uint> &outlierIndices,
                           Data<double> &data);

/////////////////////////////////////////////////////////////////////
int computeOutlierIndices(Data<uint> &cluster, uint numberOfCluster,
                          uint minimumClusterSize,
                          thrust::host_vector<uint> &outlierIndices) {
  auto &clusterOffsets = cluster.data[1];
  auto &clusterIndices = cluster.data[0];

  uint clusterStartIndicesSize = clusterOffsets.size();
  if (numberOfCluster >= clusterStartIndicesSize) {
    printf("ERROR: There are only %d data points!", clusterStartIndicesSize);
    return EXIT_FAILURE;
  }

  uint startIndex = clusterOffsets[numberOfCluster];
  if (numberOfCluster == clusterStartIndicesSize - 1)
    clusterStartIndicesSize = clusterIndices.size() - startIndex;
  else
    clusterStartIndicesSize = clusterOffsets[numberOfCluster + 1] - startIndex;

  if (clusterStartIndicesSize < minimumClusterSize) {
    outlierIndices.push_back(numberOfCluster);
    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}
