//#include "Network.cuh"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include <math.h>
#include "NetworkBackbone.cuh"

NetworkBackbone::NodeParams np;
NetworkBackbone::NetworkBackbone(int iNodes, int oNodes, int hNodes)
{
    np.iNodes = iNodes;
    np.oNodes = oNodes;
    np.hNodes = hNodes;
}


void NetworkBackbone::train(float *inputs, float *targets)
{
    
}
void NetworkBackbone::query(float *inputs)
{
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
