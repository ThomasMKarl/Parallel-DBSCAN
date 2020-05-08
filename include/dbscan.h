/**
 * @file Header file containing function definitions.
 * Defines DBSCAN_H if included.
 * Defines BLOCK_SIZE 265 if unset.
 * Defines PRECISION float if unset. 
 * 
 * @author (last to touch it) $Author: bv $
 *
 * @version $Revision: 1.0 $
 *
 * @date $Date: 2019/11/29 $
 *
 * Contact: Thomas.Karl@ur.de
 *
 * Created on: Fr Nov 29 00:39:23 2019
 */
#ifndef DBSCAN_H
#define DBSCAN_H

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<vector>
#include<string>
#include<cassert>
#include<fstream>
#include<cmath>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 265
#endif

#ifndef PRECISION
#define PRECISION float
#endif

typedef PRECISION real;

/**
 * @brief Performs a G-DBSCAN of two dimensional input data.
 *
 * @param xdata contains coordinates in x-direction
 * @param ydata contains coordinates in y-direction
 * @param eps searches for cluster point in $\varepsilon$ environment
 * @param min minimum size of clusters
 * @param cluster contains indices of all clusters after run
 * @param cluster_start contains starting indices of clusters in cluster vector after run
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int cuda_gdbscan(thrust::device_vector<real>&, thrust::device_vector<real>&, real, unsigned int, thrust::device_vector<uint>&, thrust::device_vector<uint>&);

/**
 * @brief Performs a G-DBSCAN of two dimensional input data.
 *
 * @param xdata contains coordinates in x-direction
 * @param ydata contains coordinates in y-direction
 * @param eps searches for cluster point in $\varepsilon$ environment
 * @param min minimum size of clusters
 * @param cluster contains indices of all clusters after run
 * @param cluster_start contains starting indices of clusters in cluster vector after run
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int gdbscan(thrust::device_vector<real>&, thrust::device_vector<real>&, real, unsigned int, thrust::device_vector<uint>&, thrust::device_vector<uint>&);

/**
 * @brief CUDA kernel to search for cluster sizes.
 *
 * @param eps searches for cluster point in $\varepsilon$ environment
 * @param min minimum size of clusters
 * @param size contains the sizes of each cluster after run, size is set to zero if smaller than min
 * @param xdata contains coordinates in x-direction
 * @param ydata contains coordinates in y-direction
 * @param N size of the data set
 */
__global__ void vertex_kernel( real, uint, uint*,               real*, real*, uint);

/**
 * @brief CUDA kernel to store the indices of clusters.
 *
 * @param eps searches for cluster point in $\varepsilon$ environment
 * @param min minimum size of clusters
 * @param size contains the sizes of each cluster
 * @param cluster_start contains starting indices of clusters in cluster array after run
 * @param cluster contains indices of all clusters after run
 * @param xdata contains coordinates in x-direction
 * @param ydata contains coordinates in y-direction
 * @param N size of the data set
 */
__global__ void cluster_kernel(real, uint, uint*, uint*, uint*, real*, real*, uint);

/**
 * @brief Reads two dimensional data from file.
 *
 * @param filename Name of the file containing the data. The file must store the x and y coordinates 
 * in each row seperated by whitespaces or tabulators. 
 * @param cluster contains indices of all clusters
 * @param cluster_start contains starting indices of clusters in cluster vector
 * @param xdata contains coordinates in x-direction after run
 * @param ydata contains coordinates in y-direction after run
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int read_data(std::string, thrust::host_vector<real>&, thrust::host_vector<real>&);

/**
 * @brief Sends start points of clusters in cluster_start to standard output
 *
 * @param cluster_start contains starting indices of clusters 
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int print_table(thrust::host_vector<uint>&);

/**
 * @brief Sends outlier coordinates to standard output
 *
 * @param outliers contains indices of outlier points
 * @param xdata contains coordinates in x-direction
 * @param ydata contains coordinates in y-direction 
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int print_outlier(thrust::host_vector<uint>&, thrust::host_vector<real>&, thrust::host_vector<real>&);

/**
 * @brief Sends cluster indices to standard output
 *
 * Prints indices in cluster n to standard output. 
 * Fills vector outliers with outlier indices.
 *
 * @param cluster contains indices of all clusters
 * @param cluster_start contains starting indices of clusters in cluster vector
 * @param n number of cluster
 * @param min minimum cluster size
 * @param outliers gets filled with indices of otliers
 * @return returns 0 if run succsessfull, -1 otherwise
 */
int print_cluster(thrust::host_vector<uint>&, thrust::host_vector<uint>&, uint, uint, thrust::host_vector<uint>&);

#endif //DBSCAN_H
