/**
 * @file Header file containing function definitions.
 * Defines BLOCK_SIZE 265 if unset.
 * 
 * @author (last to touch it) $Author: bv $
 *
 * @version $Revision: 1.0 $
 *
 * @date $Date: 2020/07/24 $
 *
 * Contact: Thomas.Karl@ur.de
 *
 * Created on: Fr Jul 24 00:37:23 2020
 */

#pragma once

#include<cuda_runtime.h>

#include<string>
#include<cassert>
#include<fstream>
#include<cmath>
#include<iostream>
#include<stdio.h>

#include<time.h>

#include<thrust/sort.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/reduce.h>
#include<thrust/count.h>
#include<thrust/execution_policy.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 265
#endif


template<typename T>
__global__ void get_cluster(T *xdata,
			    T *xcentroids,
			    T *ydata,
			    T *ycentroids,
			    size_t c,
			    size_t *cluster,
			    size_t N);

template<typename T>
bool contains(thrust::host_vector<T> &rand,
	      T r,
	      T size);

template<typename T>
int read_data(std::string filename,
	      thrust::host_vector<T> &xdata,
	      thrust::host_vector<T> &ydata);

template<typename T>
void print_vector(thrust::device_vector<T> &v);

template<typename T>
void print_vector(thrust::host_vector<T> &v);

template<typename T>
size_t kmeans(thrust::host_vector<T> &h_xdata,
	      thrust::host_vector<T> &h_ydata,
	      thrust::host_vector<T> &h_xcentroids,
	      thrust::host_vector<T> &h_ycentroids,
	      size_t K,
	      size_t maxiter);

template<typename T>
int seeding(thrust::host_vector<T> &h_xdata,
	    thrust::host_vector<T> &h_ydata,
	    thrust::host_vector<T> &h_xcentroids,
	    thrust::host_vector<T> &h_ycentroids);
