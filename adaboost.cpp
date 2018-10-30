#include "adaboost.h"

using namespace std;

adaboost::adaboost()
{
    dimensions = 2; // x and y
    iterations = 30;

    weak_classifiers = new float*[iterations];
    alpha = new float[iterations];
    //weighted_err = new float[iterations];

    for(int i = 0; i < iterations; i++){
        weak_classifiers[i] = new float[3];
        for(int j = 0; j < 3; j++){
            weak_classifiers[i][j] = 0.0;
        }

        alpha[i] = 0.0;
    }

    int weight_val = 1 / data_size;
    weights = new float[data_size];
    for(int i = 0; i < data_size; i++){
        weights[i] = weight_val;
    }


}

adaboost::~adaboost(){
    free(weak_classifiers);
    free(alpha);
    free(weighted_err);
    free(weights);
}

void adaboost::create_data(int dataset_size){
    data_size = dataset_size;
    x = new int[dataset_size];
    y = new int[dataset_size];
    cls = new int[dataset_size];

    for(int i = 0; i < dataset_size; i++){
        x[i] = rand() % 100;
        y[i] = rand() % 100;

        int d = sqrt(double(pow((50-x[i]), 2) + pow((50 - y[i]), 2)));
        if(abs(d) > 30){
            cls[i] = -1;
        }else{
            cls[i] = 1;
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
        output += "itr: " + QString::number(i) + " weight: " + QString::number(weights[i]) + "\n";
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
