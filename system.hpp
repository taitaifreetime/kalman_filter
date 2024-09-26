/**
 * @file system.hpp
 * @author 
 * @brief system to be estimated
 * @version 0.1
 * @date 2024-09-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <Eigen/Dense>

namespace state_estimation
{


class System
{
    public:
        System(
            const Eigen::MatrixXf &process_noise, 
            const Eigen::MatrixXf &observation_noise, 
            const Eigen::MatrixXf &system_model, 
            const Eigen::MatrixXf &control_model, 
            const Eigen::MatrixXf &observation_model 
        );
        ~System();

        // state model
        Eigen::VectorXf predictState(const double &dt, const Eigen::VectorXf &state, const Eigen::VectorXf &control);

        // observation model
        Eigen::VectorXf predictObservation(const Eigen::VectorXf &state);

        Eigen::MatrixXf getSystemNoise() const {return process_noise_;}
        Eigen::MatrixXf getObservationNoise() const {return observation_noise_;}
        Eigen::MatrixXf getSystemMatrix() {return A_;}
        Eigen::MatrixXf getControlMatrix() {return B_;}
        Eigen::MatrixXf getObservationMatrix() {return C_;}
    
    private:
        Eigen::MatrixXf process_noise_;
        Eigen::MatrixXf observation_noise_;
        Eigen::MatrixXf A_, B_, C_;
};


} // namespace state_estimation

#endif // SYSTEM_HPP