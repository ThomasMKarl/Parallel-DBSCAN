#include "../include/dbscan.h"

int gdbscan(thrust::host_vector<real> &xdata, thrust::host_vector<real> &ydata,
	    real eps, unsigned int min, thrust::host_vector<uint> &cluster, thrust::host_vector<uint> &index)
{
  uint N = xdata.size();
  if(N != ydata.size())
  {
    printf("DBSCAN failed. Inconsistent vector sizes.\n");
    return EXIT_FAILURE;
  }

  thrust::host_vector<unsigned int> size(N);
  index.resize(N);

  float xdist, ydist;
  unsigned int counter;
  for(unsigned int i = 0; i < N; ++i)
  {
    counter = 0;
    for(unsigned int j = 0; j < N; ++j)
    {
      xdist = xdata[i] - xdata[j];
      ydist = ydata[i] - ydata[j];
      if( (real)sqrt(xdist*xdist + ydist*ydist) < eps) counter++;
    }
    if(counter > min) size[i] = counter - 1;
    else              size[i] = 0;
  }

  counter = 0;
  index[0] = 0;
  for(unsigned int i = 1; i < N; ++i)
  {
    index[i] = index[i-1] + counter;
    counter += index[i];
  }
  
  cluster.resize(size[N-1]+index[N-1]);
  for(unsigned int i = 0; i < N; ++i)
  {
    counter = 0;
    for(unsigned int j = 0; j < N; ++j)
    {
      xdist = xdata[i] - xdata[j];
      ydist = ydata[i] - ydata[j];
      if( sqrtf(xdist*xdist + ydist*ydist) < eps && i != j)
      {
	cluster[index[i] + counter] = j;
	counter++;
	if(counter == size[i]) break;
      }
    }
  }
  
  return EXIT_SUCCESS;
}

int cuda_gdbscan(thrust::device_vector<real> &xdata, thrust::device_vector<real> &ydata,
		 real eps, unsigned int min, thrust::device_vector<uint> &cluster, thrust::device_vector<uint> &index)
{
  uint N = xdata.size();
  if(N != ydata.size())
  {
    printf("DBSCAN failed. Inconsistent vector sizes.\n");
    return EXIT_FAILURE;
  }
  
  thrust::device_vector<unsigned int> size(N);
  index.resize(N);

  unsigned int threads_per_block = (N + BLOCK_SIZE -1) / BLOCK_SIZE;

  vertex_kernel<<<BLOCK_SIZE, threads_per_block, 0>>>
                 (eps, min, thrust::raw_pointer_cast(size.data()),
		  thrust::raw_pointer_cast(xdata.data()), thrust::raw_pointer_cast(ydata.data()), N);
  if(cudaDeviceSynchronize() != cudaSuccess) return EXIT_FAILURE;

  thrust::exclusive_scan(thrust::device, size.begin(), size.end(), index.begin());

  cluster.resize(size[N-1]+index[N-1]);
  cluster_kernel<<<BLOCK_SIZE, threads_per_block, 0>>>
                  (eps, min, thrust::raw_pointer_cast(size.data()), thrust::raw_pointer_cast(index.data()), thrust::raw_pointer_cast(cluster.data()),
		   thrust::raw_pointer_cast(xdata.data()), thrust::raw_pointer_cast(ydata.data()), N);
  if(cudaDeviceSynchronize() != cudaSuccess) return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

__global__
void vertex_kernel(real eps, uint min, uint *size, real *xdata, real *ydata, uint N)
{
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  
  real xdist, ydist;
  unsigned int counter = 0;
  if(i < N)
  {
    for(unsigned int j = 0; j < N; ++j)
    {
      xdist = xdata[i] - xdata[j];
      ydist = ydata[i] - ydata[j];
      if( sqrtf(xdist*xdist + ydist*ydist) < eps) counter++;
    }
    if(counter > min) size[i] = counter - 1;
    else              size[i] = 0;
  }
}

__global__
void cluster_kernel(real eps, uint min, uint *size, uint *index, uint *cluster,
		    real *xdata, real *ydata, uint N)
{
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  
  real xdist, ydist;
  unsigned int counter = 0;
  if(i < N && size[i] != 0)
  {
    for(unsigned int j = 0; j < N; ++j)
    {
      xdist = xdata[i] - xdata[j];
      ydist = ydata[i] - ydata[j];
      if( sqrtf(xdist*xdist + ydist*ydist) < eps && i != j)
      {
        cluster[index[i] + counter] = j;
        counter++;
        if(counter == size[i]) break;
      }
    }
  }
}

int read_data(std::string filename, thrust::host_vector<real> &xdata, thrust::host_vector<real> &ydata)
{
  std::ifstream input(filename);
  if(!input)
  {
    printf("ERROR: could not read %s.\n", filename.c_str());
    return EXIT_FAILURE;
  }
 
  std::string word;
  uint counter = 0;
  while(input >> word)
  {
    counter += 1;
    if(counter == 1)
    {
      xdata.push_back((real)std::stod(word));
    }
    if(counter == 2)
    {
      ydata.push_back((real)std::stod(word));
      counter = 0;
    }
  }
  
  return EXIT_SUCCESS;
}

int print_table(thrust::host_vector<uint> &cluster_start)
{
  for(uint i = 0; i < cluster_start.size(); ++i)
    printf("%d ", cluster_start[i]);
  printf("\n");

  return EXIT_SUCCESS;
}

int print_outlier(thrust::host_vector<uint> &outliers, thrust::host_vector<real> &xdata, thrust::host_vector<real> &ydata)
{
  for(uint i = 0; i < outliers.size(); ++i)
    printf("%f %f\n", xdata[outliers[i]], ydata[outliers[i]]);
  printf("\n");

  return EXIT_SUCCESS;
}

int print_cluster(thrust::host_vector<uint> &cluster, thrust::host_vector<uint> &cluster_start, uint n, uint min, thrust::host_vector<uint> &outlier)
{
  uint size = cluster_start.size();
  if(n >= size)
  {
    printf("ERROR: There are only %d data points!", size);
    return EXIT_FAILURE;
  }

  uint start = cluster_start[n];
  if(n == size-1) size = cluster.size()     - start;
  else            size = cluster_start[n+1] - start;

  if(size < min)
  {
    outlier.push_back(n);
    return EXIT_SUCCESS;
  }
  
  //for(uint i = 0; i < size; ++i) printf("%d\n", cluster[start+i]);  
  //printf("\n");

  return EXIT_SUCCESS;
}
