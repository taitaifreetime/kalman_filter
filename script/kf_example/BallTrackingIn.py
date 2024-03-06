# Togarashi lid tracking
# we assume constant velocity model
# estimate = [cx,cy,vx,vy,r]
# = [x of center,
#    y of center,
#    x vel,
#    y vel,
#    radius]
# observation = [cx, cy, r]

import numpy as np
import os
import sys
import cv2
sys.path.append(os.getcwd())
from script.kalman_filter import KalmanFilter
import time

dt=0.5 # you need to set an appropriate interval for here 
x0=np.array([426, 416,0,0,45])
A=np.array([[1,0,dt,0 ,0],
            [0,1,0 ,dt,0],
            [0,0,1 ,0 ,0],
            [0,0,0 ,1 ,0],
            [0,0,0 ,0 ,1],])
B=np.identity(5)
C=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1]])
P0=np.identity(5)
u=np.array([0,0,0,0,0])
sigma_sys=1
sigma_obs=10
kf=KalmanFilter(x0, P0, A, B, C, sigma_sys, sigma_obs)

video_path = 'togarashi.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = 'estimate.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

# observation (hugh circle detection params)
dp = 1
minDist = width//2
param1 = 50
param2 = 30
minRadius = 40
maxRadius = 60
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            z=np.array([i[0],i[1],i[2]])
            # cv2.circle(frame, (z[0],z[1]), z[2], (0, 255, 0), 2)
            kf.prediction(u)
            kf.correction(z)
            
    else:
        kf.prediction(u)
        kf.P_pos_=kf.P_pri_
        kf.x_pos_=kf.x_pri_
    if int(kf.x_pos_[4])<0: continue
    cv2.circle(frame, (int(kf.x_pos_[0]),int(kf.x_pos_[1])), int(kf.x_pos_[4]), (255, 255, 255), 2)
    print(np.round(kf.x_pos_))


    out.write(frame)
    cv2.imshow('Circle Detection', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


