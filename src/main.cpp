#include "../include/dbscan.h"

int main(int argc, char *argv[])
{
  if(argc != 4)
  {
    printf("Usage: %s <filename> epsilon minimum\n", argv[0]);
    return EXIT_FAILURE;
  }
  
  std::string filename = argv[1];

  real eps;
  if(isdigit(argv[2][0])) eps = atof(argv[2]);
  else
  {
    printf("ERROR: can not convert argument 1 to real number.\n Usage: %s <filename> epsilon minimum\n", argv[0]);
    return EXIT_FAILURE;
  }
  
  unsigned int min;
  if(isdigit(argv[3][0])) min = atoi(argv[3]);
  else
  {
    printf("ERROR: can not convert argument 2 to integer number.\n Usage: %s <filename> epsilon minimum\n", argv[0]);
    return EXIT_FAILURE;
  }
  
  thrust::host_vector<real> xdata;
  thrust::host_vector<real> ydata;
  read_data(filename, xdata, ydata);
  
  thrust::host_vector<uint> cluster;
  thrust::host_vector<uint> cluster_start;
  {
    thrust::device_vector<real> d_xdata = xdata;
    thrust::device_vector<real> d_ydata = ydata;
    thrust::device_vector<uint> d_cluster;
    thrust::device_vector<uint> d_cluster_start;
    if(cuda_gdbscan(d_xdata, d_ydata, eps, min, d_cluster, d_cluster_start) != EXIT_SUCCESS) return EXIT_FAILURE;
    cluster       = d_cluster;
    cluster_start = d_cluster_start;
  }
  
  thrust::host_vector<uint> outliers;
  for(unsigned int i = 0; i < cluster_start.size(); ++i)
  {
    if(print_cluster(cluster, cluster_start, i, min,
      outliers) != EXIT_SUCCESS) return EXIT_FAILURE;
  }

  printf("There are %ld outliers and %ld clusters.\n", outliers.size(), cluster_start.size()-outliers.size());

  if(print_outlier(outliers, xdata, ydata) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  
  return EXIT_SUCCESS;
}
