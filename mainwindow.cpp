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

    adb = new adaboost(dataset_size);
    adb->create_data();

    QString output = adb->get_data_as_string();
    output += adb->get_weak_classifiers_as_string();
    output += adb->get_alpha_values_as_string();
    output += adb->get_weights_as_string();
    output += adb->get_err_as_string();
    ui->textEdit_data->setText(output);

    double bt = 0.0;
    int dir = 0;
    double er = 0.0;

    adb->find_decision_stump(bt, dir, er);
    qDebug() << "thresh: " << bt << " dir: " << dir << " er " << er;
}
