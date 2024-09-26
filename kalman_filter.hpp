/**
 * @file kalman_filter.hpp
 * @author 
 * @brief Kalman filter class
 * @version 0.1
 * @date 2024-09-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include "system.hpp"
#include <memory>

namespace state_estimation
{


class KalmanFilter
{
    public:
        KalmanFilter(
            std::shared_ptr<System> system, 
            const Eigen::VectorXf &initial_state, 
            const Eigen::MatrixXf &initial_covariance
        ) : system_(system), state_(initial_state), covariance_(initial_covariance)
        {}

        ~KalmanFilter(){}

        Eigen::VectorXf getState() const {return state_;}

        Eigen::MatrixXf getCovariance() const {return covariance_;}

        /**
         * @brief prediction step
         * 
         * @param dt time interval
         * @param control control vector
         */
        void predict(const double &dt, const Eigen::VectorXf &control)
        {
            // compute priori estimate
            state_ = system_->predictState(dt, state_, control);

            // predict error covariance matrix
            Eigen::MatrixXf A = system_->getSystemMatrix();
            Eigen::MatrixXf R = system_->getSystemNoise();
            covariance_ = A*covariance_*A.transpose() + R;
        }

        /**
         * @brief correction step
         * 
         * @param observation 
         */
        void correct(const Eigen::VectorXf &observation)
        {
            // compute kalman gain
            Eigen::MatrixXf C = system_->getObservationMatrix();
            Eigen::MatrixXf K = computeKalmanGain(C);

            // update based on innovation
            state_ = state_ + K*(observation - system_->predictObservation(state_));
            Eigen::MatrixXf I = Eigen::MatrixXf::Identity(K.rows(), K.rows());
            covariance_ = (I - K*C)*covariance_;

            // store observation
            observation_ = observation;
        }

    private:
        /**
         * @brief compute kalman gain
         * 
         * @param C observation matrix
         * @return Eigen::MatrixXf 
         */
        Eigen::MatrixXf computeKalmanGain(const Eigen::MatrixXf &C)
        {
            Eigen::MatrixXf C_T = C.transpose();
            return covariance_*C_T*(C*covariance_*C_T + system_->getObservationNoise()).inverse();
        }

        std::shared_ptr<System> system_;
        Eigen::VectorXf state_;
        Eigen::MatrixXf covariance_;
        Eigen::VectorXf observation_;
};


} // namespace state_estimation

#endif // KALMAN_FILTER_HPP