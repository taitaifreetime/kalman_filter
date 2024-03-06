import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.append(os.getcwd())
from script.kalman_filter import KalmanFilter
from script.kf_example.print_data import print_data

class Const2dAccRobotTracking(KalmanFilter):
    def __init__(self, x0, P0, A, B, C, sigma_sys, sigma_obs):
        super().__init__(x0, P0, A, B, C, sigma_sys, sigma_obs)
        self.i=0
        self.N=300
        self.t=[t for t in range(self.N)]
        self.tp_x=[10]
        self.tp_y=[10]
        self.tv_x=[0]
        self.tv_y=[0]
        self.ta_x=[10]*self.N # constant acceleration
        self.ta_x[3]=-30 # suddenly deceleration
        self.ta_y=[10]*self.N # constant acceleration
        self.ta_y[3]=-30 # suddenly deceleration
        for t in range(1,len(self.t)-1):
            self.tv_x.append(self.tv_x[t-1]+self.ta_x[t]*self.dt_)
            self.tv_y.append(self.tv_y[t-1]+self.ta_y[t]*self.dt_)
            self.tp_x.append(self.tp_x[t-1]+self.tv_x[t-1]*self.dt_+1/2*self.dt_*self.dt_*self.ta_x[t])
            self.tp_y.append(self.tp_y[t-1]+self.tv_y[t-1]*self.dt_+1/2*self.dt_*self.dt_*self.ta_y[t])

        self.o_x=[]
        self.o_y=[]
        self.estimate=[]

        print("\nWe are about to estimate the person position given that we can observe the position with sensor noise. \
True position, velocity, and acceleration were defined based on constant acceleration equation. \
We assume that a robot stops suddenly. And we defined a control vector. \
We aim here to see the impact of the control.\n")
        self.fig = plt.figure()

    def track(self, sigma_obs):
        self.sigma_obs=sigma_obs
        self.ani = FuncAnimation(self.fig, self.plot, interval=100, blit=True)
        plt.show()

        print_data(self.t[0:self.i],
                self.tp_x[0:self.i],
                [x[0] for x in self.estimate[0:self.i]],
                self.tp_y[0:self.i],
                [x[1] for x in self.estimate[0:self.i]],
                self.tv_x[0:self.i],
                [x[2] for x in self.estimate[0:self.i]],
                self.tv_y[0:self.i],
                [x[3] for x in self.estimate[0:self.i]])
        return

    def plot(self,data):
        try:
            xlim = [0,500]
            ylim = [0,500]
            plt.cla()
            
            z=np.array([self.tp_x[self.i]+np.random.normal(scale=self.sigma_obs),self.tp_y[self.i]+np.random.normal(scale=self.sigma_obs)])
            self.o_x.append(z[0])
            self.o_y.append(z[1])
            self.prediction(np.array([self.ta_x[self.i],self.ta_y[self.i]]))
            self.correction(z)
            self.estimate.append(kf.x_pos_)
            if self.i > self.N or \
                self.estimate[self.i][0]>xlim[1] or \
                self.estimate[self.i][1]>ylim[1] or \
                self.estimate[self.i][0]<0 or\
                self.estimate[self.i][1]<0:
                print("\nWe found that the estimate has followed the true value ven though the velocity of the robot suddenly changed. \
That is because we add the acceleration as the control. \
When a robot is about to bump into an obstacle and needs to stop, we can track the robot if we know the acceleration as a control. \
In real situation, we need to consider processing time and the difference between the time that we got sensor data and current time. \n")
                self.ani.event_source.stop()

            plt.ylim(ylim[0],ylim[1])
            plt.xlim(xlim[0],xlim[1])

            # observation, true value, estimate
            p_err=np.sqrt(np.power(self.tp_x[self.i]-kf.x_pos_[0],2)+np.power(self.tp_y[self.i]-kf.x_pos_[1],2))
            v_err=np.sqrt(np.power(self.tv_x[self.i]-kf.x_pos_[2],2)+np.power(self.tv_y[self.i]-kf.x_pos_[3],2))
            print(z,[self.tp_x[self.i], self.tp_y[self.i],self.tv_x[self.i], self.tv_y[self.i],self.ta_x[self.i], self.ta_y[self.i]],kf.x_pos_)
            # print(p_err,v_err)
            tru = plt.plot(self.tp_x[self.i], self.tp_y[self.i], 'x', label="true position")
            est = plt.plot(kf.x_pos_[0], kf.x_pos_[1], 'o', label="estimate")
            # obs = plt.plot(self.o_x[self.i], self.o_y[self.i], '*', label="observation")

            plt.title("Object Tracking with wrong model (true: const acc, model: const vel)")
            plt.legend()
            
            self.i+=1
        except KeyboardInterrupt:
            exit()
        return plt.gca().lines

dt=1
x0=np.array([0,0,0,0])
A=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
B=np.array([[0.5*dt*dt,0],[0,0.5*dt*dt],[dt,0],[0,dt]])
print(B)
C=np.eye(2,4)
P0=np.identity(4)
sigma_sys=1
sigma_obs=1
kf=Const2dAccRobotTracking(x0, P0, A, B, C, sigma_sys, sigma_obs)
kf.track(sigma_obs)
    
