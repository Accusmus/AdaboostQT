#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_create_data_clicked()
{
    int dataset_size =  ui->spinBox_size_dataset->value();
    int boosing_rounds = ui->spinBox_boosing_rounds->value();
    int maxXY = ui->spinBox_max_X_Y->value();

    adb = new adaboost(dataset_size, boosing_rounds, maxXY);

    adb->create_training_data();

    QString output;
    output = "Size of Dataset: " + QString::number(dataset_size) + "\nNumber of boosing rounds: " + QString::number(boosing_rounds) + "\nMax x and y: " + QString::number(maxXY) + "\n";
    output += adb->get_training_data_as_string();
    ui->textEdit_data->setText(output);
}

void MainWindow::on_pushButton_train_clicked()
{
    if(adb == NULL){
        QString output = "Error please create data first!";
        qDebug() << output;
        ui->textEdit_messages->setText(output);
        return;
    }

    adb->train_adaboost();
    QString output;
    output = adb->get_weak_classifiers_as_string();
    ui->textEdit_classifiers->setText(output);
}

void MainWindow::on_pushButton_classify_clicked()
{
    if(adb == NULL){
        ui->textEdit_messages->setText("Please initialise Adaboost first!");
        return;
    }

    int x = ui->spinBox_sample_x->value();
    int y = ui->spinBox_sample_y->value();
    int cls = ui->spinBox_class_sample->value();

    int a[2];
    a[0] = x;
    a[1] = y;

    int res = adb->classify_sample(a);

    QString string;
    if(cls == res){
        string = "predicted: " + QString::number(res) + " actual: " + QString::number(cls) + " Correct Classification!";
        ui->lineEdit_result->setText(string);
    }else{
        string = "predicted: " + QString::number(res) + " actual: " + QString::number(cls) + " incorrect Classification!";
        ui->lineEdit_result->setText(string);
    }
}

void MainWindow::on_pushButton_classify_training_clicked()
{
    if(adb == NULL){
        ui->textEdit_messages->setText("Please initialise Adaboost first!");
        return;
    }
    ui->lineEdit_result_training->setText(QString::number(adb->test_training_set()) + "%");
}

void MainWindow::on_pushButton_generate_test_clicked()
{
    int testDataSize = ui->spinBox_sample_test->value();
    if(adb == NULL){
        ui->textEdit_messages->setText("Please initialise Adaboost first!");
        return;
    }
    adb->create_test_data(testDataSize);
    QString output = adb->get_test_data_as_string();

    ui->textEdit_test_data->setText(output);
}

void MainWindow::on_pushButton_classify_test_clicked()
{
    if(adb == NULL){
        ui->textEdit_messages->setText("Please initialise Adaboost first!");
        return;
    }

    double result = adb->test_test_set();

    ui->textEdit_test_result->setText(QString::number(result)+ "%");
}

void MainWindow::on_pushButton_write_classifier_clicked()
{
    QString filename = ui->lineEdit_filename->text();

    if(adb != NULL){
        adb->write_classifier_to_file(filename);
    }else{
        ui->textEdit_messages->setText("Please initialise Adaboost by pressing Create dataset");
    }
}

void MainWindow::on_pushButton_read_classifer_clicked()
{
    QString filename = ui->lineEdit_filename->text();

    if(adb != NULL){
        adb->read_classifier_from_file(filename);
    }else{
        ui->textEdit_messages->setText("Please initialise Adaboost by pressing Create dataset");
    }
}
