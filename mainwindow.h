#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <cstdlib>
#include <qmath.h>
#include "adaboost.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_create_data_clicked();

    void on_pushButton_train_clicked();

    void on_pushButton_classify_clicked();

    void on_pushButton_classify_training_clicked();

    void on_pushButton_generate_test_clicked();

    void on_pushButton_classify_test_clicked();

private:
    Ui::MainWindow *ui;
    adaboost *adb;
};

#endif // MAINWINDOW_H
