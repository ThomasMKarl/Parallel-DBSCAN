#include "kmeans.hpp"

template<typename T>
__global__ void get_cluster(T *xdata,
			    T *xcentroids,
			    T *ydata,
			    T *ycentroids,
			    size_t c,
			    size_t *cluster,
			    size_t N)
{
  size_t idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < N)
  {
    T new_dist, dist, distx, disty;
    distx = xdata[idx]-xcentroids[0];
    disty = ydata[idx]-ycentroids[0];
    dist = sqrtf( distx*distx + disty*disty);
    size_t cl = 0;
  
    for(size_t i = 1; i < c; ++i)
    {
      distx = xdata[idx]-xcentroids[i];
      disty = ydata[idx]-ycentroids[i];
      new_dist = sqrtf( distx*distx + disty*disty);
      //printf("%f %f %f\n", xcentroids[i], dist, new_dist);
      if(new_dist < dist)
      {
        dist = new_dist;
        cl = i;
      }
      //else cl = 1234;
      __syncthreads();
    }
    
    //printf("%ld %ld\n", idx, cl);
    cluster[idx] = cl;
  }
}
template
__global__ void get_cluster<float>(float *xdata,
			    float *xcentroids,
			    float *ydata,
			    float *ycentroids,
			    size_t c,
			    size_t *cluster,
			    size_t N);
template
__global__ void get_cluster<double>(double *xdata,
			    double *xcentroids,
			    double *ydata,
			    double *ycentroids,
			    size_t c,
			    size_t *cluster,
			    size_t N);

////////////////////////////////////////////////////////////
template<typename T>
inline bool contains(thrust::host_vector<T> &rand,
	             T r,
	             T size)
{
  for(size_t i = 0; i < size; ++i)
  {
    if(rand[i] == r) return true;
  }
  return false;
}
template
bool contains<float>(thrust::host_vector<float> &rand,
	      float r,
	      float size);
template
bool contains<double>(thrust::host_vector<double> &rand,
	      double r,
	      double size);
  
////////////////////////////////////////////////////////////
template<typename T>
int read_data(std::string filename,
	      thrust::host_vector<T> &xdata,
	      thrust::host_vector<T> &ydata)
{
  std::ifstream input(filename);
  if(!input)
  {
    std::cerr << "ERROR: could not read " << filename << ".\n";
    return EXIT_FAILURE;
  }
 
  std::string word;
  size_t count = 0;
  while(input >> word)
  {
    if(count == 0)
    {
      xdata.push_back((T)std::stod(word));
      count++;
    }
    else
    {
      ydata.push_back((T)std::stod(word));
      count = 0;
    }
  }
  
  return EXIT_SUCCESS;
}

template
int read_data<float>(std::string filename,
	      thrust::host_vector<float> &xdata,
	      thrust::host_vector<float> &ydata);
template
int read_data<double>(std::string filename,
	      thrust::host_vector<double> &xdata,
	      thrust::host_vector<double> &ydata);
  
////////////////////////////////////////////////////////////
template<typename T>
void print_vector(thrust::device_vector<T> &v)
{
  for(size_t i = 0; i < v.size(); ++i)
    std::cout << v[i] << " ";
  std::cout << "\n";
}
template
void print_vector<float>(thrust::device_vector<float> &v);
template
void print_vector<double>(thrust::device_vector<double> &v);
  
////////////////////////////////////////////////////////////
template<typename T>
size_t kmeans(thrust::host_vector<T> &h_xdata,
	      thrust::host_vector<T> &h_ydata,
	      thrust::host_vector<T> &h_xcentroids,
	      thrust::host_vector<T> &h_ycentroids,
	      size_t K,
	      size_t maxiter)
{
  size_t N = h_xdata.size();
  size_t threadsPerBlock = BLOCK_SIZE;
  size_t blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  
  thrust::device_vector<T> d_xdata;
  thrust::device_vector<T> d_ydata;
  d_xdata = h_xdata;
  d_ydata = h_ydata;
  
  thrust::device_vector<T> d_xcentroids(K);
  thrust::device_vector<T> d_ycentroids(K);
  d_xcentroids = h_xcentroids;
  d_ycentroids = h_ycentroids;
  
  thrust::device_vector<size_t> d_cluster(N);
  

  size_t sum;
  size_t iter = 0;
  bool done = false;
  float resultx, resulty;
  size_t *offs = (size_t*)malloc((K+1)*sizeof(size_t));
  while(iter < maxiter && !done)
  {
    get_cluster<<<blocks,threadsPerBlock>>>
      (thrust::raw_pointer_cast(d_xdata.data()),
       thrust::raw_pointer_cast(d_xcentroids.data()),
       thrust::raw_pointer_cast(d_ydata.data()),
       thrust::raw_pointer_cast(d_ycentroids.data()),
       K,
       thrust::raw_pointer_cast(d_cluster.data()),
       N);
    if(cudaDeviceSynchronize() != cudaSuccess) return EXIT_FAILURE;
    
    thrust::device_vector<size_t> help = d_cluster;
    thrust::sort_by_key(thrust::device, d_cluster.begin(), d_cluster.end(), d_xdata.begin());
    thrust::sort_by_key(thrust::device, help.begin(), help.end(), d_ydata.begin());  
    help.resize(0);

    print_vector(d_xcentroids);
    print_vector(d_ycentroids);
    
    std::cout << "#cluster offsets:\n";
    sum = 0;
    for(size_t i = 0; i < K; ++i)
    {
      offs[i] = sum;
      std::cout << "#" << sum << " ";
      if(sum == 0) sum++;
      sum += thrust::count(d_cluster.begin(), d_cluster.end(), i);
    }
    offs[K] = N;
    std::cout << "\n";
    
    done = true;
    for(size_t i = 0; i < K; ++i)
    {
 	resultx = thrust::reduce(thrust::device,
				 thrust::raw_pointer_cast(d_xdata.data())+offs[i],
				 thrust::raw_pointer_cast(d_xdata.data())+offs[i+1]-1,
				 0)/(offs[i+1]-offs[i]+1);
	resulty = thrust::reduce(thrust::device,
				 thrust::raw_pointer_cast(d_ydata.data())+offs[i],
				 thrust::raw_pointer_cast(d_ydata.data())+offs[i+1]-1,
				 0)/(offs[i+1]-offs[i]+1);

      if(d_xcentroids[i] != resultx || d_ycentroids[i] != resulty) done = false;
      d_xcentroids[i] = resultx;
      d_ycentroids[i] = resulty;
    }

    iter++;
  }
  
  h_xdata = d_xdata;
  h_ydata = d_ydata;
  h_xcentroids = d_xcentroids;
  h_ycentroids = d_ycentroids;

  return iter;
}
template
size_t kmeans<float>(thrust::host_vector<float> &h_xdata,
	      thrust::host_vector<float> &h_ydata,
	      thrust::host_vector<float> &h_xcentroids,
	      thrust::host_vector<float> &h_ycentroids,
	      size_t K,
	      size_t maxiter);
template
size_t kmeans<double>(thrust::host_vector<double> &h_xdata,
	      thrust::host_vector<double> &h_ydata,
	      thrust::host_vector<double> &h_xcentroids,
	      thrust::host_vector<double> &h_ycentroids,
	      size_t K,
	      size_t maxiter);

////////////////////////////////////////////////////////////
template<typename T>
int seeding(thrust::host_vector<T> &h_xdata,
	    thrust::host_vector<T> &h_ydata,
	    thrust::host_vector<T> &h_xcentroids,
	    thrust::host_vector<T> &h_ycentroids)
{
  srand(time(NULL));

  size_t K = h_xcentroids.size();
  size_t N = h_xdata.size();
  if(K > N)
  {
    std::cerr << "More clusters than data points!\n";
    return EXIT_FAILURE;
  }
    
  thrust::host_vector<size_t> randv(K);
  size_t r;
  std::cout << "#initial values:\n";
  for(size_t i = 0; i < K; ++i)
  {
    r = rand()%N;
    while(contains(randv,r,K)) r = rand()%N;
    randv[i] = r;
    h_xcentroids[i] = h_xdata[r];
    h_ycentroids[i] = h_ydata[r];
    std::cout << "#" << h_xcentroids[i] << " " << h_ycentroids[i] << "\n";
  }
  std::cout << "\n";

  return EXIT_SUCCESS;
}
template
int seeding<float>(thrust::host_vector<float> &h_xdata,
	    thrust::host_vector<float> &h_ydata,
	    thrust::host_vector<float> &h_xcentroids,
	    thrust::host_vector<float> &h_ycentroids);
template
int seeding<double>(thrust::host_vector<double> &h_xdata,
	    thrust::host_vector<double> &h_ydata,
	    thrust::host_vector<double> &h_xcentroids,
	    thrust::host_vector<double> &h_ycentroids);
