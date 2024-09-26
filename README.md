# Kalman Filter in C++

## Usage
- git clone this in you project and it would be as follows
```
your_project
    ├─kalman_filter
    │   ├── kalman_filter.hpp
    │   ├── README.md
    │   └── system.hpp
    │
    └─your_code.cpp
```
- include in your_code.cpp 
```
#include "kalman_filter/kalman_filter.hpp"
#include <Eigen/Dense>

using namespace state_estimation;
```
- define the state of the system
```
std::shared_ptr<System> system_;
system_.reset(
    new System(
        process_noise, 
        observation_noise, 
        system_model, 
        control_model, 
        observation_model
    )
);

std::shared_ptr<KalmanFilter> kf_;
kf_.reset(new KalmanFilter(system, observation, initial_covariance));
```
- start kalman filter loop
```
// fix your vector dimension
// Eigen::VectorXf control;
// Eigen::VectorXf observation;
kf_->predict(dt, control);
kf_->correct(observation);
```