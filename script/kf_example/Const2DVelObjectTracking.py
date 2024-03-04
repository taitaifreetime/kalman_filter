import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.append(os.getcwd())
from script.kalman_filter import KalmanFilter
from script.kf_example.print_data import print_data

class Const2dAccObjectTracking(KalmanFilter):
    def __init__(self, x0, P0, A, B, C):
        super().__init__(x0, P0, A, B, C)
        self.i=0
        self.N=300
        self.t=[t for t in range(self.N)]
        self.tp_x=[10]
        self.tp_y=[10]
        self.tv_x=[10]
        self.tv_y=[10]
        self.ta_x=[0]*self.N # constant velocity = acceleration is 0
        self.ta_y=[0]*self.N # constant velocity = acceleration is 0
        for t in range(len(self.t)-1):
            self.tv_x.append(self.tv_x[t]+self.ta_x[t]*self.dt_)
            self.tv_y.append(self.tv_y[t]+self.ta_y[t]*self.dt_)
            self.tp_x.append(self.tp_x[t]+self.tv_x[t]*self.dt_+1/2*self.dt_*self.dt_*self.ta_x[t]*self.dt_)
            self.tp_y.append(self.tp_y[t]+self.tv_y[t]*self.dt_+1/2*self.dt_*self.dt_*self.ta_y[t]*self.dt_)

        self.o_x=[]
        self.o_y=[]
        self.estimate=[]

        print("\nWe are about to estimate the person position given that we can observe the position with sensor noise. \
In case that an object that will be estimated is human, we do not consider control vector. \
True position, velocity, and acceleration were defined based on constant velocity equation. \
Also we defined a transition model considering constant velocity. \
We will see the estimate follows the true position.\n")
        self.fig = plt.figure()

    def track(self, sigma_sys, sigma_obs):
        self.sigma_sys=sigma_sys
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
            self.prediction(np.array([np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys)]),np.array([0,0,0,0]))
            self.correction(np.array([np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys)]),z)
            self.estimate.append(kf.x_pos_)
            if self.i > self.N or \
                self.estimate[self.i][0]>xlim[1] or \
                self.estimate[self.i][1]>ylim[1] or \
                self.estimate[self.i][0]<0 or\
                self.estimate[self.i][1]<0:
                print("\nWe found that the estimate has followed the true position properly. \
In addition that, we can also look at the output log and know that there is a little difference between true and estimate. \
This difference is caused by a sensor noise by gaussian. \
Why we estimate with a bit difference is because we defined the correct models in accordance with the object's motion.\n")
                self.ani.event_source.stop()

            plt.ylim(ylim[0],ylim[1])
            plt.xlim(xlim[0],xlim[1])

            # observation, true value, estimate
            print(z,[self.tp_x[self.i], self.tp_y[self.i],self.tv_x[self.i], self.tv_y[self.i]],kf.x_pos_)
            tru = plt.plot(self.tp_x[self.i], self.tp_y[self.i], 'x', label="true position")
            est = plt.plot(kf.x_pos_[0], kf.x_pos_[1], 'o', label="estimate")
            # obs = plt.plot(self.o_x[self.i], self.o_y[self.i], '*', label="observation")

            plt.title("Object Tracking with const vel model")
            plt.legend()
            
            self.i+=1
        except KeyboardInterrupt:
            exit()
        return plt.gca().lines

dt=1
x0=np.array([0,0,0,0])
A=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
B=np.identity(4)
C=np.eye(2,4)
P0=np.identity(4)
kf=Const2dAccObjectTracking(x0, P0, A, B, C)
kf.track(1,1)
    
