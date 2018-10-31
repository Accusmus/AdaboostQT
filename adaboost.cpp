#include "adaboost.h"

using namespace std;

adaboost::adaboost(int dataset_size, int boosing_rounds, int max):data_size(dataset_size),iterations(boosing_rounds), maxXY(max)
{

    dimensions = 2; // x and y

    weak_classifiers = new float*[iterations];
    alpha = new float[iterations];
    weighted_err = new float[iterations];

    for(int i = 0; i < iterations; i++){
        weak_classifiers[i] = new float[3];
        for(int j = 0; j < 3; j++){
            weak_classifiers[i][j] = 0.0;
        }

        alpha[i] = 0.0;
    }



    double weight_val = 1.0 / double(data_size);
    weights = new float[data_size];
    for(int i = 0; i < data_size; i++){
        weights[i] = weight_val;
    }

    double inf = numeric_limits<double>::infinity();

    for(int i = 0; i < iterations; i++){
        weighted_err[i] = inf;
    }

    qDebug() << "Set Up finished!";
}

adaboost::~adaboost(){
    free(weak_classifiers);
    free(alpha);
    free(weighted_err);
    free(weights);
}

void adaboost::create_data(){
    x = new int[data_size];
    y = new int[data_size];
    cls = new int[data_size];

    for(int i = 0; i < data_size; i++){
        x[i] = rand() % maxXY;
        y[i] = rand() % maxXY;

        int d = sqrt(double(pow(((maxXY/2)-x[i]), 2) + pow(((maxXY/2) - y[i]), 2)));
        if(abs(d) > (maxXY/3)){
            cls[i] = -1;
        }else{
            cls[i] = 1;
        }
    }
}

void adaboost::train_adaboost(){
    int dim[2] = {0, 1};

    int classification[data_size];

    for(int t = 0; t < iterations; t++){
        for(int feature : dim){

            double bestThresh = 0.0;
            int bestDirection = 0;
            double minError = 0.0;

            if(feature == 0){
                find_decision_stump(x,bestThresh, bestDirection, minError);
            }else{
                find_decision_stump(y,bestThresh, bestDirection, minError);
            }


            if(minError < weighted_err[t]){
                weighted_err[t] = minError;
                weak_classifiers[t][0] = bestThresh;
                weak_classifiers[t][1] = feature;
                weak_classifiers[t][2] = bestDirection;
            }
        }

        alpha[t] = find_alpha(weighted_err[t]);


        if(weak_classifiers[t][1] == 0){
            classify_against_weak_classifier(x, weak_classifiers[t][0], weak_classifiers[t][2], classification);
        }else{
            classify_against_weak_classifier(y, weak_classifiers[t][0], weak_classifiers[t][2], classification);
        }

        update_weights(alpha[t], classification, cls);

        normalise_weights();

    }
}

int max(const int* arr, int size){
    int max = numeric_limits<int>::min();
    for(int i = 0; i < size; i++){
        if(arr[i] > max){
            max = arr[i];
        }
    }
    return max;
}

int min(const int*arr, int size){
    int min = numeric_limits<int>::max();
    for(int i = 0; i < size; i++){
        if(arr[i] < min){
            min = arr[i];
        }
    }
    return min;
}

void adaboost::find_decision_stump(const int* arr, double &bestThresh, int &bestDirection, double &minError){
    //intialise
    minError = numeric_limits<double>::infinity();
    bestThresh = 0.0;
    bestDirection = 0;

    double interval = double(maxXY)/ 100.0;

    //keep track of all of the thresholds
    QVector<double> thresholds;

    for(double i = (min(arr, data_size)-interval*2.0); i < (max(arr, data_size) - interval*2.0); i+= interval){
        thresholds.append(i);
    }

    int directions[2] = {1, -1};
    double tempErr = 0.0;

    for(int d : directions){
        for(double thresh : thresholds){

            int actual[data_size];
            int correct[data_size]; //correct classification = 0 , // incorrect classification = 1

            if(d == 1){
                for(int i = 0; i < data_size; i++){
                    if(arr[i] >= thresh){
                        actual[i] = 1;
                    }else{
                        actual[i] = -1;
                    }
                }
            }else{
                for(int i = 0; i < data_size; i++){
                    if(arr[i] < thresh){
                        actual[i] = 1;
                    }else{
                        actual[i] = -1;
                    }
                }
            }

            for(int i = 0; i < data_size; i++){
                if(actual[i] == cls[i]){
                    correct[i] = 0; //correct
                }else{
                    correct[i] = 1; //incorrect
                }
            }

            double e = 0.0;
            for(int i = 0; i < data_size; i++){
                e += (correct[i] * weights[i]);
            }
            tempErr = e;

            if(tempErr < minError){
                minError = tempErr;
                bestThresh = thresh;
                bestDirection = d;
            }
        }
    }
}

double adaboost::find_alpha(double weightedErr){
    return 0.5 * log((1 - weightedErr)/(weightedErr));
}

void adaboost::update_weights(double alpha, int *classification, int *label){
    for(int i = 0; i < data_size; i++){
        weights[i] = weights[i] * exp((-1.0) * alpha * label[i] * classification[i]);
    }
}

void adaboost::normalise_weights(){
    //Find the sum of all the weights
    double sum = 0;
    for(int i = 0; i < data_size; i++){
        sum += weights[i];
    }

    //normalise the weights according to the sum
    for(int i = 0; i < data_size; i++){
        weights[i] = (weights[i]/sum);
    }
}

void adaboost::classify_against_weak_classifier(const int *x, double threshold,int direction, int *classifications){
    for(int i = 0; i < data_size; i++){
        if(direction == -1){
            if(x[i] < threshold){
                classifications[i] = 1;
            }else{
                classifications[i] = -1;
            }
        }else{
            if(x[i] < threshold){
                classifications[i] = -1;
            }else{
                classifications[i] = 1;
            }
        }
    }
}

QString adaboost::get_data_as_string(){
    QString output;
    for(int i = 0; i < data_size; i++){
        output += "x: " + QString::number(x[i]) + " y: " + QString::number(y[i]) + " cls: " + QString::number(cls[i]) + "\n";
    }
    return output;
}

QString adaboost::get_weak_classifiers_as_string(){
    QString output;
    for(int i = 0; i < iterations; i++){
        output += "itr: " + QString::number(i) + " Thresh: " + QString::number(weak_classifiers[i][0]) + " Dim: " + QString::number(weak_classifiers[i][1]) + " Dir: " + QString::number(weak_classifiers[i][2]) + "\n";
    }
    return output;
}

QString adaboost::get_alpha_values_as_string(){
    QString output;
    for(int i = 0; i < iterations; i++){
        output += "itr: " + QString::number(i) + " a = " + QString::number(alpha[i]) + "\n";
    }
    return output;
}

QString adaboost::get_weights_as_string(){
    QString output;
    for(int i = 0; i < data_size; i++){
        output += "data number: " + QString::number(i) + " weight: " + QString::number(weights[i]) + "\n";
    }
    return output;
}

QString adaboost::get_err_as_string(){
    QString output;
    for(int i = 0; i < iterations; i++){
       output +=  "itr: " + QString::number(i) + " err: " + QString::number(weighted_err[i]) + "\n";
    }
    return output;
}
