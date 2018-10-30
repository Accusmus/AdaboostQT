#ifndef ADABOOST_H
#define ADABOOST_H

#include<QString>

class adaboost
{
public:
    adaboost();
    ~adaboost();
    void create_data(int dataset_size);
    QString get_data_as_string();
    QString get_weak_classifiers_as_string();
    QString get_alpha_values_as_string();
    QString get_weights_as_string();
    QString get_err_as_string();

private:
    // Data
    int *x;
    int *y;
    int *cls;

    //number of boosting iterations
    int iterations;
    //size of the data set
    int data_size;
    //number of dimentions of the data set
    int dimensions;

    //will contain 2D array [threshold, dimension, direction]
    //for each boosting round
    float **weak_classifiers;
    //Alpha values for each boosting round
    float *alpha;
    //weight values for each sample
    float *weights;
    //weighted error per boosting iteration
    float *weighted_err;

};

#endif // ADABOOST_H
