#ifndef ANNETWORK_H
#define ANNETWORK_H

struct Network
{
    struct NodeParams
    {
        static int iNodes, oNodes, hNodes;
        static int iter;
    };
    Network(int iNodes, int oNodes, int hNodes);
    void train(float *inputs, float *targets);
    void query(float *inputs);
    float getInputQuantity();
    float getOutputQuantity();
    float getHiddenQuantity();
    //   void fetchLast();
};

#endif