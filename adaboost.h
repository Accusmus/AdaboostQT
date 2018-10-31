#ifndef ADABOOST_H
#define ADABOOST_H

#include<QString>
#include<QVector>
#include<QDebug>
#include<math.h>
#include<QFile>
#include<QTextStream>

class adaboost
{
public:
    adaboost(int dataset_size, int boosing_rounds, int max);
    ~adaboost();

    void create_training_data();
    void create_test_data(int testSize);

    QString get_training_data_as_string();
    QString get_test_data_as_string();
    QString get_weak_classifiers_as_string();
    QString get_alpha_values_as_string();
    QString get_weights_as_string();
    QString get_err_as_string();

    void train_adaboost();
    int classify_sample(int *sample);
    double test_training_set();
    double test_test_set();
    void write_classifier_to_file(QString filename);
    void read_classifier_from_file(QString filename);


private:
    // Training Data
    int *x_train;
    int *y_train;
    int *cls_train;

    // Testing Data
    int *x_test;
    int *y_test;
    int *cls_test;
    int test_data_size;

    const int maxXY;

    //number of boosting iterations
    int iterations;
    //size of the data set
    const int data_size;
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

    void find_decision_stump(const int *arr, double &bestThresh, int &bestDirection, double &minError);
    double find_alpha(double weightedErr);
    void update_weights(double alpha, int *classification, int *label);
    void normalise_weights();
    void classify_against_weak_classifier(const int *x_train, double threshold,int direction, int *classifications);

};

#endif // ADABOOST_H
