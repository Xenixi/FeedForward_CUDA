#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <vector>
#include "Network.cuh"

namespace anncuda0
{
    struct NetworkBackbone
    {
        struct NodeParams
        {
            int iNodes, oNodes, hNodes;
            int iter;
        };
        NodeParams np;
        NetworkBackbone(int iNodes, int oNodes, int hNodes)
        {
            np.iNodes = iNodes;
            np.oNodes = oNodes;
            np.hNodes = hNodes;
        }

        //error " error: invalid redeclaration of type name "Network"
        //D:\REPOLOCAL\Cloned\parallel-computing\FeedForward_CUDA\main\Network.cuh(13): here"
        //---
        void train(float *inputs, float *targets)
        {
        }
        void query(float *inputs)
        {
        }
        int getInputQuantity()
        {
            return np.iNodes;
        }
        int getOutputQuantity()
        {
            return np.oNodes;
        }
        int getHiddenQuantity()
        {
            return np.hNodes;
        }
    };
} // namespace anncuda0

int main(void)
{

    std::cout << "hello" << std::endl;
    anncuda0::NetworkBackbone n(/*Input*/ 784, /*Output*/ 10, /*Hidden*/ 200);

    float *inputs, *targets;
    cudaMallocManaged(&inputs, sizeof(float) * n.getInputQuantity());
    cudaMallocManaged(&targets, sizeof(float) * n.getInputQuantity());

    std::cout << "Reading inputs..." << std::endl;

    std::ifstream inputsFile("dat/trainMNIST.csv");
    std::stringstream bufferStream;
    bufferStream << inputsFile.rdbuf();
    std::string str;

    char delim = ',';

    std::vector<std::string> strings;
    for (int i = 0; std::getline(bufferStream, str, delim); i++)
    {
        strings.push_back(str);
    }

    for (std::string str2 : strings)
    {
        std::cout << "Read: " << str2 << std::endl;
    }

    cudaFree(inputs);
    cudaFree(targets);

    return 0;
}
