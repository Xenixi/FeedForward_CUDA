#ifndef ANNETWORK_H
#define ANNETWORK_H


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
    void query(float *inputs);
    int getInputQuantity();
    int getOutputQuantity();
    int getHiddenQuantity();
    //   void fetchLast();
};


#endif