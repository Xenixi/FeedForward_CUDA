#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#ifndef ANNETWORK_H
#define ANNETWORK_H

__global__ void initWeights(float *weightsInputHidden, float *weightsHiddenOutput);


__global__ void trainNetwork(float *inputs, float *targets, int iNodes, int hNodes, int oNodes);
__global__ void queryNetwork(float *inputs);
__device__ float activation(float input);

class NetworkBackbone
{

public:
    struct NodeParams
    {
        int iNodes, oNodes, hNodes;
        int iter;
    };
    NetworkBackbone(int iNodes, int oNodes, int hNodes);
    void train(float *inputs, float *targets);
    void query(float *inputs, float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes);
    void init(float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes);
    int getInputQuantity();
    int getOutputQuantity();
    int getHiddenQuantity();
    int* getDeviceProperties();

    class Utils
    {
        cudaDeviceProp getDeviceProps();

    public:
        int getSMs();
        int getTB();
        int getTMP();
        int getName();
    };
    //   void fetchLast();
};

#endif