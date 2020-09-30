#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#ifndef CudaTestClass_H
#define CudaTestClass_H
__global__ void createArrays(const int num, double *array0, double *array1);
__global__ void multiplyArrays(const int num, double *array0, double *array1);

#endif