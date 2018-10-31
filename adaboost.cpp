#include "adaboost.h"

using namespace std;

adaboost::adaboost(int dataset_size, int boosing_rounds, int max):
    data_size(dataset_size),
    iterations(boosing_rounds),
    maxXY(max)
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

void adaboost::create_training_data(){
    x_train = new int[data_size];
    y_train = new int[data_size];
    cls_train = new int[data_size];

    for(int i = 0; i < data_size; i++){
        x_train[i] = rand() % maxXY;
        y_train[i] = rand() % maxXY;

        int d = sqrt(double(pow(((maxXY/2)-x_train[i]), 2) + pow(((maxXY/2) - y_train[i]), 2)));
        if(abs(d) > (maxXY/3)){
            cls_train[i] = -1;
        }else{
            cls_train[i] = 1;
        }
    }
}

void adaboost::create_test_data(int testSize){
    test_data_size = testSize;
    x_test = new int[testSize];
    y_test = new int[testSize];
    cls_test = new int[testSize];

    for(int i = 0; i < test_data_size; i++){
        x_test[i] = rand() % maxXY;
        y_test[i] = rand() % maxXY;

        int d = sqrt(double(pow(((maxXY/2)-x_test[i]),2) + pow(((maxXY/2)-y_test[i]),2)));
        if(abs(d) > (maxXY/3)){
            cls_test[i] = -1;
        }else{
            cls_test[i] = 1;
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
                find_decision_stump(x_train,bestThresh, bestDirection, minError);
            }else{
                find_decision_stump(y_train,bestThresh, bestDirection, minError);
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
            classify_against_weak_classifier(x_train, weak_classifiers[t][0], weak_classifiers[t][2], classification);
        }else{
            classify_against_weak_classifier(y_train, weak_classifiers[t][0], weak_classifiers[t][2], classification);
        }

        update_weights(alpha[t], classification, cls_train);

        normalise_weights();

    }
}


int adaboost::classify_sample(int *sample){
    double sum = 0;

    for(int i = 0; i < iterations; i++){
        double thresh = weak_classifiers[i][0];
        int feat = weak_classifiers[i][1];
        int direction = weak_classifiers[i][2];
        double alp = alpha[i];

        int ht = sample[feat] - thresh;
        int sign_res= 0;

        if(ht >0 ){
            sign_res = 1 * direction;
        }else{
            sign_res = -1 * direction;
        }

        sum += alp * sign_res;
    }
    if(sum >= 0){
        return 1; // inside
    }else{
        return -1; // outside
    }
}

double adaboost::test_training_set(){
    int sum = 0;
    for(int i = 0; i < data_size; i++){
        int a[2];
        a[0] = x_train[i];
        a[1] = y_train[i];
        int res = classify_sample(a);
        if(res == cls_train[i]){
            sum++;
        }
    }

    return (double(sum)/double(data_size))*100.0;
}

double adaboost::test_test_set(){
    int sum = 0;
    for(int i = 0; i < test_data_size; i++){
        int a[2];
        a[0] = x_test[i];
        a[1] = y_test[i];
        int res = classify_sample(a);
        if(res == cls_test[i]){
            sum++;
        }
    }

    return (double(sum)/double(test_data_size))*100.0;
}

void adaboost::write_classifier_to_file(QString filename){
    filename = "../data/" + filename;

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly| QIODevice::Text)) {
        qDebug() << "Unable to Write to file!" + filename;
        return;
    }

    QTextStream out(&file);

    out << iterations << "\n";
    for(int i = 0; i < iterations; i++){
        out << weak_classifiers[i][0] << " " << weak_classifiers[i][1] << " " << weak_classifiers[i][2] << " " << alpha[i] << "\n";
    }
    file.close();

    qDebug() << "File Written Successfully";
}

void adaboost::read_classifier_from_file(QString filename){
    filename = "../data/" + filename;

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly| QIODevice::Text)) {
        qDebug() << "Unable to Read from file!" + filename;
        return;
    }

    QTextStream in(&file);
    QString line;

    int count = 0;
    while(!in.atEnd()){
        line = in.readLine();
        if(count == 0){
            iterations  = line.toInt();
            qDebug() << iterations;
        }else{
            weak_classifiers[count-1][0] = line.split(" ")[0].toDouble();
            weak_classifiers[count-1][1] = line.split(" ")[1].toDouble();
            weak_classifiers[count-1][2] = line.split(" ")[2].toDouble();
            alpha[count-1] = line.split(" ")[3].toDouble();
        }
        count++;
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
                if(actual[i] == cls_train[i]){
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

QString adaboost::get_training_data_as_string(){
    QString output;
    for(int i = 0; i < data_size; i++){
        output += "x: " + QString::number(x_train[i]) + " y: " + QString::number(y_train[i]) + " cls: " + QString::number(cls_train[i]) + "\n";
    }
    return output;
}

QString adaboost::get_test_data_as_string(){
    QString output;
    for(int i = 0; i < data_size; i++){
        output += "x: " + QString::number(x_test[i]) + " y: " + QString::number(y_test[i]) + " cls: " + QString::number(cls_test[i]) + "\n";
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
