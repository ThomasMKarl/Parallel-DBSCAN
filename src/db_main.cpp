#include "clustering.hpp"
#include "dbscan.hpp"

#ifndef PRECISION
#define PRECISION float
#endif

typedef PRECISION real;


int main(int argc, char *argv[])
{
  if(argc < 6)
  {
    printf("Usage: %s <filename> epsilon minimumClusterSize dimensions dataSize\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::string filename = argv[1];

  real epsilon;
  if(isdigit(argv[2][0])) epsilon = atof(argv[2]);
  else
  {
    printf("ERROR: can not convert argument 1 to real number.\n Usage: %s <filename> epsilon minimumClusterSize dimensions dataSize\n", argv[0]);
    return EXIT_FAILURE;
  }
 
  uint minimumClusterSize;
  if(isdigit(argv[3][0])) minimumClusterSize = atoi(argv[3]);
  else
  {
    printf("ERROR: can not convert argument 2 to integer number.\n Usage: %s <filename> epsilon minimumClusterSize dimensions dataSize\n", argv[0]);
    return EXIT_FAILURE;
  }

  uint dimensions;
  if(isdigit(argv[4][0])) dimensions = atoi(argv[4]);
  else
  {
    printf("ERROR: can not convert argument 3 to integer number.\n Usage: %s <filename> epsilon minimumClusterSize dimensions dataSize\n", argv[0]);
    return EXIT_FAILURE;
  }

  uint dataSize;
  if(isdigit(argv[5][0])) dataSize = atoi(argv[5]);
  else
  {
    printf("ERROR: can not convert argument 4 to integer number.\n Usage: %s <filename> epsilon minimumClusterSize dimensions dataSize\n", argv[0]);
    return EXIT_FAILURE;
  }

  Setup<real> setup(minimumClusterSize, epsilon);

  Data<real> hostData(dimensions, dataSize);

  readDataFromFile(filename, hostData);

  Data<uint> cluster(2,0);
  if(gdbscan_cuda(hostData, setup, cluster) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  thrust::host_vector<uint> outlierIndices;
  for(uint number = 0; number < cluster.data[0].size(); ++number)
  {
    if(computeOutlierIndices(cluster, number, minimumClusterSize,
       outlierIndices) != EXIT_SUCCESS) return EXIT_FAILURE;
  }

  printf("#There are %ld outliers and %ld clusters.\n",
	 outlierIndices.size(),
	 cluster.data[1].size() - outlierIndices.size());

  if(printOutliers(outlierIndices, hostData) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  
  return EXIT_SUCCESS;
}
