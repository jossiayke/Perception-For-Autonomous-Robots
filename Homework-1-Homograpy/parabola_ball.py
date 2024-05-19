#!/usr/bin/python3

# ========================================
# ENPM673 Spring 2022: Perception for Autonomous Robotics
# HW 1: Question 2
#
# Author: Yoseph Kebede
# ========================================
# Run as 'python3 parabola_ball.py'
# Press ESC for exit


from ast import Num
import matplotlib.pyplot as plt
import numpy as np
import cv2

def lstSq(myCrd_top, myCrd_mean, myCrd_bottom, res, num):
    # Displaying the list of top, mean, and bottom
    #print("\n Array for top points \n")
    #print(myCrd_top)
    #print("\n Array for mean/middle points \n")
    #print(myCrd_mean)
    #print("\n Array for bottom points \n")
    #print(myCrd_bottom)
        
    # Plotting the three regions of the ball for entire video
    for i in range(len(myCrd_top)):
        plt.scatter(myCrd_top[i][0], myCrd_top[i][1], s=5)
    for j in range(len(myCrd_mean)):
        plt.scatter(myCrd_mean[j][0], myCrd_mean[j][1], s=5)
    for k in range(len(myCrd_bottom)):
        plt.scatter(myCrd_bottom[k][0], myCrd_bottom[k][1],s=5)
    
    x = np.vstack(myCrd_top)
    x1 = np.vstack(myCrd_mean)
    x2 = np.vstack(myCrd_bottom)

    plt.title("Tracking ball top middle bottom points for: Video %d" % num)
    plt.xlabel("Pixels in X-direction")
    plt.ylabel("Pixels in Y-direction")
    plt.axis([0, res[1], 0, res[0]])
    plt.xlim([x2[:,0].min()-100,x2[:,0].max()+100])
    plt.ylim([x2[:,1].min()-100,x[:,1].max()+100])
    plt.grid(visible=True, which='both',axis='both',color='k',linestyle='--',linewidth='0.8')
    plt.grid(which='minor', color='k', linestyle=':',linewidth='0.4')
    plt.minorticks_on()    

    # To approximate the plot using least squares for quadratic path
    # we have ax^2 + bx + c = y

    # Setup A v = y quadratic curve fit for parabola
    A = np.zeros([len(myCrd_top),3], dtype=int)
    A1 = np.zeros([len(myCrd_mean),3], dtype=int)
    A2 = np.zeros([len(myCrd_bottom),3], dtype=int)

    # A is [x^2 x 1], so let's create A from datasets top
    A[:,0] = np.square(x[:,0])
    A[:,1] = x[:,0]
    A[:,2] = 1
    #print(A)

    A1[:,0] = np.square(x1[:,0])
    A1[:,1] = x1[:,0]
    A1[:,2] = 1

    
    A2[:,0] = np.square(x2[:,0])
    A2[:,1] = x2[:,0]
    A2[:,2] = 1

    # Evaluate v using v = (A^T A)^-1 A^T* y
    y= np.vstack(x[:,1])
    y1= np.vstack(x1[:,1])
    y2 = np.vstack(x2[:,1])

    #print(A.transpose())
    #print(np.vstack(np.transpose(A), ))
    pInv_0 = np.matmul(np.transpose(A), A)
    pInv_1 = np.linalg.inv(pInv_0)
    pInv_2 = np.matmul(np.transpose(A), y)
    v = np.matmul(pInv_1, pInv_2)
    #print(v)

    pInv_10 = np.matmul(np.transpose(A1), A1)
    pInv_11 = np.linalg.inv(pInv_10)
    pInv_12 = np.matmul(np.transpose(A1), y1)
    v1 = np.matmul(pInv_11, pInv_12)

    pInv_20 = np.matmul(np.transpose(A2), A2)
    pInv_21 = np.linalg.inv(pInv_20)
    pInv_22 = np.matmul(np.transpose(A2), y2)
    v2 = np.matmul(pInv_21, pInv_22)
    # Find the coefficients of the quadratic a, b and c
    # For top
    a=v[0] 
    b=v[1]
    c=v[2]

    # For middle
    a1=v1[0] 
    b1=v1[1]
    c1=v1[2]
    
    # For bottom
    a2=v2[0] 
    b2=v2[1]
    c2=v2[2]

    #Plot the curve fitting
    plt.plot(x[:,0], a*np.square(x[:,0])+b*x[:,0]+c,label='Ball Topmost',color='green',linestyle='--', linewidth=1.3)
    plt.plot(x1[:,0], a1*np.square(x1[:,0])+b1*x1[:,0]+c1,label='Ball middle',color='black',linestyle='-', linewidth=1.3)
    plt.plot(x2[:,0], a2*np.square(x2[:,0])+b2*x2[:,0]+c2,label='Ball Bottommost',color='red',linestyle=':', linewidth=1.3)
    plt.legend()
    
    #plt.show() - Uncomment to have plot disaplay at this point

    # Plots saved in same directory
    plt.savefig("Video-%d_LSQ.png" %num)
    plt.clf()    

def main():
    """
    Reads a video frame by frame, filters the red channel of ball
    in video, and stores in X and Y coordinates. Information then 
    used to plot curves

    """
    print ("Using OpenCV", cv2.__version__)
    
    # Read videos frame
    vidObj_1 = cv2.VideoCapture('ball_video1.mp4')
    vidObj_2 = cv2.VideoCapture('ball_video2.mp4')

    success, image = vidObj_1.read()
    count = 0

    success1, image1 = vidObj_2.read()
    second = False
   
    # Storing (x,y) coordinates of tuples of top, mean and bottom of ball in frame
    # First Video
    myCrd_top = []
    myCrd_mean = []
    myCrd_bottom = []

    # Second Video
    myCrd_top1 = []
    myCrd_mean1 = []
    myCrd_bottom1 = []

    res = [image.shape[0], image.shape[1]]
    res1 = [image1.shape[0], image1.shape[1]]

    # Resolution of First and Second Video
    #print(res, res1)

    while success:

        # Mask image to only show red ball in grayscale
        image_hsv= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Applying gaussian blur to clean out image
        image_hsv = cv2.GaussianBlur(image_hsv, (5, 5), 0)
        image_hsv = image_hsv.astype(np.uint8)
 
        # Mask image to only color ball in black
        lower_grey = np.array([128])
        upper_grey = np.array([255])
        mask1 = cv2.inRange(image_hsv, lower_grey, upper_grey)
        image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask = mask1)
        image_hsv[np.where(mask1==0)] = 0
        
        # Obtain coordinates (y, x) of black ball
        y,x = np.where(image_hsv==0)

        # Flipping the y coordinates so that origin is left bottom corner
        y = res[0] - y

        # Obtain list of indices with min and max pixel locations
        y_list = np.where(y == y.min())
        y_m = np.where(y == y.max())

        # Take mean value of x and y coordinates for topmost and bottommost region
        top = [round(np.mean(x[y_m])), round(np.mean(y[y_m]))]
        bottom =  [round(np.mean(x[y_list])), round(np.mean(y[y_list]))]
        mean = [round((top[0]+bottom[0])/2),round((top[1]+bottom[1])/2)]
                
        # Uncomment below to display masked frame
        # cv2.imshow('Masked frames', image_hsv)
        # cv2.waitKey(0)
        
        if not second:
            # Append coorinate of frame in list for each frame
            myCrd_top.append(top)
            myCrd_mean.append(mean)
            myCrd_bottom.append(bottom)

            #Save frame as JPEG file inside directory
            cv2.imwrite("frame%d.jpg" % count, image_hsv)

            success, image = vidObj_1.read()
            
            if not success:
                success = success1
                second = True
                image = image1
                count = 0
                res = res1
                continue
        else:
            # Append coorinate of frame in list for each frame
            myCrd_top1.append(top)
            myCrd_mean1.append(mean)
            myCrd_bottom1.append(bottom)

            #Save frame as JPEG file inside directory
            cv2.imwrite("frame1%d.jpg" % count, image_hsv)
            
            success, image = vidObj_2.read()
            
        print("New frame read (1) True: ", success)
        count +=1

    # For first video
    lstSq(myCrd_top=myCrd_top, myCrd_mean=myCrd_mean, myCrd_bottom=myCrd_bottom, res=res, num=1)        
    
    #For second video
    lstSq(myCrd_top=myCrd_top1, myCrd_mean=myCrd_mean1, myCrd_bottom=myCrd_bottom1, res=res1, num=2)

    
if __name__ == '__main__':
    main()

