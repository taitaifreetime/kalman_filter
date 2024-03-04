import matplotlib.pyplot as plt

def print_data(t,tx,x,ty,y,tvx,vx,tvy,vy):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t,tx)
        plt.plot(t,x)
        plt.title('position_x')
        plt.subplot(2, 1, 2)
        plt.plot(t,ty)
        plt.plot(t,y)
        plt.title('position_y')
        plt.show()

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t,tvx)
        plt.plot(t,vx)
        plt.title('velocity_x')
        plt.subplot(2, 1, 2)
        plt.plot(t,tvy)
        plt.plot(t,vy)
        plt.title('velocity_y')
        plt.show()