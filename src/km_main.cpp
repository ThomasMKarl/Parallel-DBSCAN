#include "kmeans.h"

int main(int argc, char *argv[])
{
  if(argc != 4)
  {
    std::cerr << "usage: " << argv[0] << " <filename> <K> <maxiter>\n";
    return EXIT_FAILURE;
  }
  std::string filename = argv[1];
  size_t K = atoi(argv[2]);
  size_t maxiter = atoi(argv[3]);

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
  thrust::host_vector<float> h_xdata;
  thrust::host_vector<float> h_ydata;
  if(read_data(filename, h_xdata, h_ydata) != EXIT_SUCCESS) return EXIT_FAILURE;
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  thrust::host_vector<float> h_xcentroids(K);
  thrust::host_vector<float> h_ycentroids(K);
  if(seeding(h_xdata, h_ydata, h_xcentroids, h_ycentroids) != EXIT_SUCCESS) return EXIT_FAILURE;
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  size_t iter = kmeans(h_xdata, h_ydata, h_xcentroids, h_ycentroids, K, maxiter);
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::cout << "finished after " << iter << " iterations...\n";

  return EXIT_SUCCESS;
}
