#include "adaboost.h"

using namespace std;

adaboost::adaboost(int dataset_size)
{
    data_size = dataset_size;

    dimensions = 2; // x and y
    iterations = 30;

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

        int d = sqrt(double(pow((50-x[i]), 2) + pow((50 - y[i]), 2)));
        if(abs(d) > 30){
            cls[i] = -1;
        }else{
            cls[i] = 1;
        }
    }
}

void adaboost::find_decision_stump(double &bestThresh, int &bestDirection, double &minError){
    //intialise
    minError = numeric_limits<double>::infinity();
    bestThresh = 0.0;
    bestDirection = 0;

    double interval = double(maxXY)/ 100.0;

    //keep track of all of the thresholds
    QVector<double> thresholds;

    for(double i = 0; i < maxXY; i+= interval){
        thresholds.append(i);
    }

    int directions[2] = {1, -1};

    for(int d : directions){
        for(auto thresh : thresholds){

            int actual[data_size];
            int correct[data_size]; //correct classification = 0 , // incorrect classification = 1

            if(d == 1){
                for(int i = 0; i < data_size; i++){
                    if(x[i] >= thresh){
                        actual[i] = 1;
                    }else{
                        actual[i] = -1;
                    }
                }
            }else{
                for(int i = 0; i < data_size; i++){
                    if(x[i] < thresh){
                        actual[i] = 1;
                    }else{
                        actual[i] = -1;
                    }
                }
            }

            for(int i = 0; i < data_size; i++){
                qDebug() << "actual " << actual[i] << " exp: " << cls[i];
                if(actual[i] == cls[i]){
                    correct[i] = 0; //correct
                }else{
                    correct[i] = 1; //incorrect
                }
            }

            double tempErr = 0.0;
            for(int i = 0; i < data_size; i++){
                tempErr += correct[i] * weights[i];
            }

            if(tempErr < minError){
                minError = tempErr;
                bestThresh = thresh;
                bestDirection = d;
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
