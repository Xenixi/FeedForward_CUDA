#include <iostream>
#include <chrono>
#include <math.h>
#include <fstream>
#include <sstream>
#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <vector>
#include "NetworkBackbone.cuh"

int main(void)
{

    std::cout << "hello" << std::endl;
    NetworkBackbone n(/*Input*/ 784, /*Output*/ 10, /*Hidden*/ 200);

    float *inputs, *targets;
    cudaMallocManaged(&inputs, sizeof(float) * n.getInputQuantity());
    cudaMallocManaged(&targets, sizeof(float) * n.getInputQuantity());

    std::cout << "Reading inputs..." << std::endl;

    auto time1 = std::chrono::high_resolution_clock::now();

    std::ifstream inputsFile("dat/mnist_train.csv");
    std::stringstream bufferStream;
    bufferStream << inputsFile.rdbuf();
    std::string str;

    char delim = ',';

    std::vector<std::string> trainStrings;
    int i;
    for (i = 0; std::getline(bufferStream, str, delim); i++)
    {
        trainStrings.push_back(str);
    }
    std::cout << "Train - Total: " << i << std::endl;
    //test file

    std::ifstream testFile("dat/mnist_test.csv");
    std::stringstream bufferStream2;
    bufferStream2 << testFile.rdbuf();
    std::string str2;

    std::vector<std::string> testStrings;

    for (i = 0; std::getline(bufferStream2, str2, delim); i++)
    {
        testStrings.push_back(str2);
    }

    std::cout << "Test - Total: " << i << std::endl;
   
    auto time2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
    std::cout << "Completed.\nTime taken: " << duration << "ms" << std::endl;

    cudaFree(inputs);
    cudaFree(targets);



    return 0;
}
