/**
 * @file Header file containing function definitions for dbscan.
 * Defines BLOCK_SIZE 265 if unset.
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

#include "clustering.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 265
#endif

template <typename T>
__host__ T calculateDistance(uint indexFrom, uint indexTo, Data<T> &data);

template <typename T>
__device__ T calculateDistance(uint indexFrom, uint indexTo, uint dimension,
                               uint dataSize, T *data);

/**
 * @brief Performs a G-DBSCAN of input data.
 *
 * @param data contains each coordinate of input data in one entry
 * @param setup specifies $\varepsilon$ environment for the search and
 * minimum size of clusters
 * @param cluster contains indices of all clusters in cluster.data[0]
 * and corresponding starting indices of clusters in cluster.data[1]
 * after run
 * @return returns 0 if run succsessfull, -1 otherwise
 */
template <typename T>
int gdbscan_cuda(Data<T> &data, Setup<T> &setup, Data<uint> &cluster);

/**
 * @brief Performs a G-DBSCAN of input data.
 *
 * @param data contains each coordinate of input data in one entry
 * @param setup specifies $\varepsilon$ environment for the search and
 * minimum size of clusters
 * @param cluster contains indices of all clusters in cluster.data[0]
 * and corresponding starting indices of clusters in cluster.data[1]
 * after run
 * @return returns 0 if run succsessfull, -1 otherwise
 */
template <typename T>
int gdbscan(Data<T> &data, Setup<T> &setup, Data<T> &cluster);

/**
 * @brief CUDA kernel to search for cluster sizes.
 *
 * @param epsilon (from SimulationSetup) searches for
 * cluster point in $\varepsilon$ environment
 * @param minimumClusterSize (from SimulationSetup)
 * @param size contains the sizes of each cluster after run, size is set to zero
 * if smaller than minimumClusterSize of @param setup
 * @param data contains coordinates
 * @param dataSize size of the data set
 * @param dimension dimensionality of the data set
 */
template <typename T>
__global__ void computeVertices_kernel(T epsilon, uint minimumClusterSize,
                                       uint *clusterSizes, T *data,
                                       uint dataSize, uint dimension);

/**
 * @brief CUDA kernel to store the indices of clusters.
 *
 * @param epsilon (from SimulationSetup) searches for cluster point in
 * $\varepsilon$ environment
 * @param minimumClusterSize (from SimulationSetup)
 * @param clusterSizes contains the sizes of each cluster
 * @param clusterStartIndices contains starting indices of clusters in cluster
 * array after run
 * @param clusterIndices contains indices of all clusters after run
 * @param data contains coordinates
 * @param dataSize size of the data set
 * @param dimension dimensionality of the data set
 */
template <typename T>
__global__ void
computeCluster_kernel(T epsilon, uint minimumClusterSize, uint *clusterSizes,
                      uint *clusterStartIndices, uint *clusterIndices, T *data,
                      uint dataSize, uint dimension);

/**
 * @brief Sends start points of clusters in clusterStartIndices
 * to standard output
 *
 * @param clusterStartIndices contains starting indices of clusters
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int printStartIndices(thrust::host_vector<uint> &clusterStartIndices);

/**
 * @brief Sends outlier coordinates to standard output
 *
 * @param outlierIndices contains indices of outlier points
 * @param data contains each coordinate of input data in one entry
 * @return returns 0 if run succsessfull, -1 otherwise
 */
template <typename T>
int printOutliers(thrust::host_vector<uint> &outlierIndices, Data<T> &data);

/**
 * @brief Sends cluster indices to standard output
 *
 * Prints indices in cluster n to standard output.
 * Fills outlierIndices.
 *
 * @param cluster contains indices of all clusters
 * @param cluster_start contains starting indices of clusters in cluster vector
 * @param numberOfcluster
 * @param minimumClusterSize
 * @param outlierIndices gets filled with indices of outliers
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int computeOutlierIndices(Data<uint> &cluster, uint numberOfCluster,
                          uint minimumClusterSize,
                          thrust::host_vector<uint> &outlierIndices);
