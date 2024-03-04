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
        self.tv_x=[0]
        self.tv_y=[0]
        self.ta_x=[10]*self.N # constant acceleration
        self.ta_y=[10]*self.N # constant acceleration
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
True position, velocity, and acceleration were defined based on constant acceleration equation. \
However, we defined a transition model considering constant velocity actually. \
We aim here to see the estimate if we defined the \"wrong\" model by mistake.\n")
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
                print("\nWe found that the estimate has been delay as time has gone on according to output logs. \
That is because we defined the transition model \"incorrectly\", and then acceleration has not updated according to the wrong model. \
But estimation has been computed and updated by observation. So it appeared to have followed the true position with a bit late. \
In real situation, if models had been wrong, there would be a possibility that we cannot observe objects because of out of sensor range. \
Even if we had observed the position again, which means the object exists wthin the sensor range, we would see a large difference between true and estimate because of the wrong models.\n")
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
B=np.identity(4)
C=np.eye(2,4)
P0=np.identity(4)
kf=Const2dAccObjectTracking(x0, P0, A, B, C)
kf.track(1,1)
    
