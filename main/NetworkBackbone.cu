//#include "Network.cuh"
#include <iostream>
#include <math.h>
#include <random>
#include "NetworkBackbone.cuh"

// https://youtu.be/5IkodnY0PeY?t=592
//runs on GPU
/*__global__ void initWeights(float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes){
    int idx = blockDim.x * blockIdx.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < hNodes*iNodes; i+=stride){
        
    } 
    for(int i = idx; i < hNodes*oNodes; i+=stride){

    }
}
*/
__global__ void trainNetwork(float *inputs, float *targets, int iNodes, int hNodes, int oNodes)
{
}

__global__ void queryNetwork(float *inputs, float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    //CONTINUE HERE PG 132 PYTHON BOOK & http://luniak.io/cuda-neural-network-implementation-part-1/#implementation-plan
    //
    for (int i = idx; i < iNodes; i += stride)
    {
        inputs[i] = inputs[i] * weightsInputHidden[i];
        inputs[i] = activation(inputs[i]);
        inputs[i] = inputs[i] * weightsHiddenOutput[i];
        inputs[i] = activation(inputs[i]);
    }
}

__device__ float activation(float input)
{
    return input / (abs(input) + 1);
}

/////////////////////
int *NetworkBackbone::getDeviceProperties()
{
    Utils u;
    int *props = new int[3];
    props[0] = u.getSMs();
    props[1] = u.getTB();
    props[2] = u.getTMP();

    return props;
}
int NetworkBackbone::Utils::getSMs()
{
    return getDeviceProps().multiProcessorCount;
}

int NetworkBackbone::Utils::getTB()
{
    return getDeviceProps().maxThreadsPerBlock;
}
int NetworkBackbone::Utils::getTMP()
{
    return getDeviceProps().maxThreadsPerMultiProcessor;
}

cudaDeviceProp NetworkBackbone::Utils::getDeviceProps()
{
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    return properties;
}
//CPU
//continued
NetworkBackbone::NodeParams np;
NetworkBackbone::NetworkBackbone(int iNodes, int oNodes, int hNodes)
{
    np.iNodes = iNodes;
    np.oNodes = oNodes;
    np.hNodes = hNodes;

    //initialization
    float *weightsInputsHidden, *weightsHiddenOutput;

    cudaMallocManaged(&weightsInputsHidden, sizeof(float) * np.iNodes * np.hNodes);
    cudaMallocManaged(&weightsHiddenOutput, sizeof(float) * np.hNodes * np.oNodes);

    init(weightsInputsHidden, weightsHiddenOutput, np.iNodes, np.hNodes, np.oNodes);
}
/****************
intiialization
****************/
void NetworkBackbone::init(float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes)
{
    for (int i = 0; i < (iNodes * hNodes); i++)
    {
        std::random_device rnd;
        std::mt19937 mt1(rnd());
        std::uniform_real_distribution<float> dist(-0.5, 0.5);

        weightsInputHidden[i] = dist(mt1);
        std::cout << "val_" << i << ": " << weightsInputHidden[i] << std::endl;
    }
}

void NetworkBackbone::train(float *inputs, float *targets)
{
    //allocate shared memory GPU
    //Maybe move ALL values to GPU ahead of time?
    cudaMallocManaged(&inputs, sizeof(float) * getInputQuantity());
    cudaMallocManaged(&targets, sizeof(float) * getOutputQuantity());

    //kernel call
    int blocks = (getDeviceProperties()[0] * getDeviceProperties()[2] / getDeviceProperties()[1]);
    int blockThreads = getDeviceProperties()[1];

    trainNetwork<<<blocks, blockThreads>>>(inputs, targets, getInputQuantity(), getHiddenQuantity(), getOutputQuantity());

    //free

    cudaFree(inputs);
    cudaFree(targets);
}
void NetworkBackbone::query(float *inputs, float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes)
{
    int blocks = (getDeviceProperties()[0] * getDeviceProperties()[2] / getDeviceProperties()[1]);
    int blockThreads = getDeviceProperties()[1];

    queryNetwork<<<blocks, blockThreads>>>(inputs, weightsInputHidden, weightsHiddenOutput, iNodes, hNodes, oNodes);
}
int NetworkBackbone::getInputQuantity()
{
    return np.iNodes;
}
int NetworkBackbone::getOutputQuantity()
{
    return np.oNodes;
}
int NetworkBackbone::getHiddenQuantity()
{
    return np.hNodes;
}
