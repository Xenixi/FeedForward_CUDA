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
__global__ void trainNetwork(float *inputs, float *targets, float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes, float *learningRate, float *outputErrs, float *finalOutputs, float *hiddenOutputs, float *hiddenErrs)
{
    //  THE INPUT VARIABLE PASSED IN WILL BE REPLACED WITH THE RETURNED ERROR VALUES SO MAKE A COPY OF THE VARIABLE!!!!
    
    //does this work in CUDA GPU kernel function?...
    float *inputsOrig = inputs;



    int idx = blockDim.x * blockIdx.x + threadIdx.x, stride = blockDim.x * gridDim.x;

    for (int i = idx; i < iNodes; i += stride)
    {
        inputs[i] = inputs[i] * weightsInputHidden[i];
        hiddenOutputs[i] = activation(inputs[i]);

    /// final_inputs = numpy.dot(self.who, hidden_outputs)
        inputs[i] = weightsHiddenOutput[i] * hiddenOutputs[i];
    /// final_outputs = self.activation_function(final_inputs)
        finalOutputs[i] = activation(inputs[i]);
    ///output_errors = targets - final_outputs
        outputErrs[i] = targets[i] - finalOutputs[i];
    ///hidden_errors = numpy.dot(self.who.T, output_errors)
        hiddenErrs[i] = weightsHiddenOutput[i] * outputErrs[i];
    //// self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
    //// self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
    //**************************************************************************
    //  IF SOMETHING GOES WRONG IT'S PROBABLY THIS THIS IS PROBABLY BROKEN MAY NEED TO BE FIXED- THE PY ONE USED NUMPY TRANSPOSE AND 2D ARRAYS THIS DOESN'T
       weightsHiddenOutput[i] += (learningRate * (outputErrs[i]*finalOutputs[i]*(1.0-finalOutputs[i])) * hiddenOutputs[i]);
       weightsInputHidden[i] += (learningRate * (hiddenErrs[i]*hiddenOutputs[i]*(1.0-hiddenOutputs[i])) * inputsOrig[i]);

        ///LINE 66 LEFT OFF (PYTHON PROGRAM)
        ///NUMPY.TRANSPOSE LINES AND STUFF FROM THE PYTHON (LINE 71 IN THE OTHER PROGRAM - )
    }
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

void NetworkBackbone::train(float *inputs, float *targets, float *learningRate, float *outputErrs, float *finalOutputs, float *hiddenOutputs, float *hiddenErrs)
{
    //allocate shared memory GPU
    //Maybe move ALL values to GPU ahead of time?
    cudaMallocManaged(&inputs, sizeof(float) * getInputQuantity());
    cudaMallocManaged(&targets, sizeof(float) * getOutputQuantity());

    //only need 1 for the learning rate (not an array-like type)
    cudaMallocManaged(&learningRate, sizeof(float));

    cudaMallocManaged(&outputErrs, sizeof(float) * getOutputQuantity());
    cudaMallocManaged(&finalOutputs, sizeof(float) * getOutputQuantity());
    cudaMallocManaged(&hiddenOutputs, sizeof(float) * getHiddenQuantity());
    cudaMallocManaged(&hiddenErrs, sizeof(float) * getHiddenQuantity());

    //kernel call
    int blocks = (getDeviceProperties()[0] * getDeviceProperties()[2] / getDeviceProperties()[1]);
    int blockThreads = getDeviceProperties()[1];

    trainNetwork<<<blocks, blockThreads>>>(inputs, targets, getInputQuantity(), getHiddenQuantity(), getOutputQuantity(), learningRate, outputErrs, finalOutputs, hiddenOutputs, hiddenErrs);

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
