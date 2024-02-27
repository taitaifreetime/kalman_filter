import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class KalmanFilter:
    def __init__(self, x0, P0, A, B, C):
        self.x_pos_=x0
        self.P_pos_=P0
        self.A_=A
        self.B_=B
        self.C_=C
        self.I_=np.identity(len(x0))

    def prediction(self, v):
        self.Q_=np.dot(v,v.T)
        self.x_pri_=np.dot(self.A_,self.x_pos_)
        self.P_pri_=np.dot(np.dot(self.A_,self.P_pos_),self.A_.T)+np.dot(np.dot(self.B_,self.Q_),self.B_.T)

    def correction(self, w, y):
        self.R_=np.dot(w,w.T)
        if self.P_pos_.ndim==1:
            self.KG_=np.dot(self.P_pri_,self.C_.T)/(np.dot(np.dot(self.C_,self.P_pri_),self.C_.T)+self.R_)
        else:
            self.KG_=np.dot(np.dot(self.P_pri_,self.C_.T),np.linalg.inv((np.dot(np.dot(self.C_,self.P_pri_),self.C_.T)+self.R_)))
        innovation=(y-np.dot(self.C_,self.x_pri_))
        self.x_pos_=self.x_pri_+np.dot(self.KG_,innovation)
        self.P_pos_=np.dot((self.I_-np.dot(self.KG_,self.C_)),self.P_pri_)

class Const1dVelObjectTracking(KalmanFilter):
    def __init__(self, x0, P0, A, B, C):
        super().__init__(x0, P0, A, B, C)
        N=300
        self.t=np.arange(N)
        self.true_position=np.arange(N)
        self.observation=[]
        self.estimate=[]

    def track(self, sigma_sys, sigma_obs):
        for p in self.true_position:
            self.estimate.append(self.x_pos_)
            z=p+np.random.normal(scale=sigma_obs)
            self.observation.append(z)
            self.prediction(np.array([np.random.normal(scale=sigma_sys)]))
            self.correction(np.array([np.random.normal(scale=sigma_obs)]),np.array([z]))

        plt.plot(self.t, self.true_position, label="true")
        plt.plot(self.t, self.observation, label="observation")
        plt.plot(self.t, self.estimate, label="estimate")
        plt.title("1D Object Tracking By Kalman Filter")
        plt.legend()
        plt.show()

x0=np.array([0,])
P0=np.array([1,])
A=np.array([1,])
B=np.array([1,])
C=np.array([1,])
kf=Const1dVelObjectTracking(x0, P0, A, B, C)
# kf.track(1,2)

class Const2dVelObjectTracking(KalmanFilter):
    def __init__(self, x0, P0, A, B, C):
        super().__init__(x0, P0, A, B, C)
        self.i=0
        self.true_position=[]
        self.observation=[]
        self.estimate=[]
        self.N=300
        self.fig = plt.figure()

    def track(self, sigma_sys, sigma_obs):
        self.sigma_sys=sigma_sys
        self.sigma_obs=sigma_obs
        ani = FuncAnimation(self.fig, self.plot, interval=30, blit=True)
        plt.show()

    def plot(self,data):
        try:
            xlim = [0,self.N]
            ylim = [0,self.N]
            plt.cla()
            
            self.true_position.append([self.i,self.i])
            self.estimate.append(kf.x_pos_)
            z=self.true_position[self.i]+np.array([np.random.normal(scale=self.sigma_obs),np.random.normal(scale=self.sigma_obs)])
            self.observation.append(z)
            self.prediction(np.array([np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys)]))
            self.correction(np.array([np.random.normal(scale=self.sigma_sys),np.random.normal(scale=self.sigma_sys)]),z)
            
            if self.i > self.N:
                exit(1)

            plt.ylim(ylim[0],ylim[1])
            plt.xlim(xlim[0],xlim[1])
            tru = plt.plot(self.true_position[self.i][0], self.true_position[self.i][1], 'x', label="true")
            est = plt.plot(self.estimate[self.i][0], self.estimate[self.i][1], 'o', label="estimate")
            obs = plt.plot(self.observation[self.i][0], self.observation[self.i][1], '*', label="observation")

            plt.title("2D Object Tracking By Kalman Filter")
            plt.legend()
            
            self.i+=1
        except KeyboardInterrupt:
            exit(1)
        return plt.gca().lines

x0=np.array([0,0,1,1])
A=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
B=np.identity(4)
C=np.eye(2,4)
P0=np.identity(4)
kf=Const2dVelObjectTracking(x0, P0, A, B, C)
kf.track(1,2)
    

    

