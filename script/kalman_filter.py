import numpy as np

class KalmanFilter:
    def __init__(self, x0, P0, A, B, C, sigma_sys, sigma_obs):
        self.x_pos_=x0
        self.P_pos_=P0
        self.A_=A
        self.B_=B
        self.C_=C
        self.I_=np.identity(len(x0))
        self.Q_=np.identity(B.shape[1])*sigma_sys
        self.R_=np.identity(C.shape[0])*sigma_obs
        self.dt_=1

    def prediction(self, u):
        self.x_pri_=np.dot(self.A_,self.x_pos_)+np.dot(self.B_, u)
        self.P_pri_=np.dot(np.dot(self.A_,self.P_pos_),self.A_.T)+np.dot(np.dot(self.B_,self.Q_),self.B_.T)

    def correction(self, y):
        if self.P_pos_.ndim==1:
            self.KG_=np.dot(self.P_pri_,self.C_.T)/(np.dot(np.dot(self.C_,self.P_pri_),self.C_.T)+self.R_)
        else:
            self.KG_=np.dot(np.dot(self.P_pri_,self.C_.T),np.linalg.inv((np.dot(np.dot(self.C_,self.P_pri_),self.C_.T)+self.R_)))
        innovation=(y-np.dot(self.C_,self.x_pri_))
        self.x_pos_=self.x_pri_+np.dot(self.KG_,innovation)
        self.P_pos_=np.dot((self.I_-np.dot(self.KG_,self.C_)),self.P_pri_)
