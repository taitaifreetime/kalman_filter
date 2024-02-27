/**
 * @file kalman_filter.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-02-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <Eigen/Dense>
using namespace Eigen;

class KalmanFilter{
    public:
        KalmanFilter();
        ~KalmanFilter();
    private:
        VectorXd x_pri_, x_pos_; // prior state estimate, poterior state estimate
        MatrixXd P_pri_, P_pos_, KG_; // prior covarance matrix, posterior covariance matrix, kalman gain
        MatrixXd A_, B_, C_; // matrix for system model and obsrvation model
        MatrixXd Q_, R_; // system noise covariance matrix, observation noise covarinace matrix 
        MatrixXd I_; // 
        void Initialization(VectorXd x0, MatrixXd P0, MatrixXd A, MatrixXd B, MatrixXd C){
            this->x_pos_=x0;
            this->P_pos_=P0;
            this->A_=A;
            this->B_=B;
            this->C_=C;
            this->I_=MatrixXd::Zero(x0.size())
        };
        void Prediction(VectorXd v){
            this->Q_=v*v->transpose();
            this->x_pri_=this->A_*this->x_pos_;
            this->P_pri_=this->A_*this->P_pos_*this->A_->transpose()+this->B_*this->Q_*this->B_->transpose();
        };
        void Correction(VectorXd w, VectorXd y){
            this->R_=w*w->transpose();
            this->KG_=P_pri_*this->C_->transpose()*(this->C_*this->P_pri_*this->C_->transpose()+this->R_)->inverse();
            this->x_pos_=this->x_pri_+this->KG_*(y-this->C_*this->x_pri_);
            this->P_pos_=(this->I_-this->KG_*this->C)*this->P_pri_;
        };
        // void CorrectionNonObservation(VectorXd w);
};

#endif // KALMAN_FILTER_H_