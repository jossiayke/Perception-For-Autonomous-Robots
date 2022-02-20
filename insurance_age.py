#!/usr/bin/python3

# ========================================
# ENPM673 Spring 2022: Perception for Autonomous Robotics
# HW 1: Question 3
#
# Author: Yoseph Kebede
# ========================================
# Run as 'python3 parabola_ball.py'
# Press ESC for exit


from ast import Num
from tkinter import N
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
import random
import cv2

def ran(e, p, s):
	"""
	purpose: Computes number of samples N using RANSAC algorithm
	param: e - 'float' outlier probability
		   p - 'float' desired probability to get a good sample
		   s - 'int' number of points in sample
	return: N - 'int' number of samples close to regression line
	"""
	num = m.log(1-p)
	den = m.log(1-(1-e)**s)
	print(den)
	N = int(float(num / den))

	return N

def cov_eigV(data, x, y):
	
	A = np.zeros([len(data),2], dtype=int)
	for i in range(len(data)):
		A[i,0] = data[i,0]
		A[i,1] = data[i,1]

	# Compute covariance matrix (S), for array - x
	""" Do something """
	# S = sum((x - x _mean) * (x - x_mean)^T) / len(x)
	# x = X - 1*1' X * (1/n)
	x_mean, y_mean = np.mean(x), np.mean(y)
	#A = np.column_stack(((A[:,0]-x_mean), (A[:,1] - y_mean)))#[x-x_mean, y-y_mean]
	A[:,0]=A[:,0]-x_mean
	A[:,1]=A[:,1]-y_mean
	
	#xn, yn = [x-x_mean, y-y_mean]
	A=[A[:,0],A[:,1]]
	A_T = np.transpose(A)
	mult = np.matmul(A,A_T)
	
	# Co-variance matrix is computed below as: S = x * x' (1/n)
	S = mult / len(A)
	
	# Uncomment to see covariance matrix on terminal
	#print(S, "\n")
	
	# Compute eigenvalues and eigenvectors

	""" Do something """
	# |S - L*I| v = 0
	# [x1-L1 x2] [v1] = 0
	# [x3 x4-L2] [v2]

	origin = [0,0]
	eigV, eigVt = np.linalg.eig(S)
	
	# Plot data and eigenvectors on same graph
	""" Do something """
	x = A[0][:]; y=A[1][:]
	for i in range(len(x)):
		#plt.scatter(x[i], y[i], s=5)
		plt.scatter(x[i],y[i], s=5)

	#xx= eigVt[:,0]*eigV[0]
	xx = [eigV[0],0]

	#xy = eigVt[:,1]*eigV[1]
	xy = [0,eigV[1]]
	# Norm of the eigen Vector
	# norm =  np.linalg.norm(eigVt[:,0])
	# norm1 =  np.linalg.norm(eigVt[:,1])
	
	plt.title("Person's Age vs Health Insurance Cost ")
	plt.xlabel("Person's age")
	plt.ylabel("Charging")
	plt.quiver(*origin, *xx, angles='xy',scale_units='xy',label='Eig Vector 1', color=['r'],scale=2)
	plt.quiver(*origin, xy[0],xy[1], angles='xy',scale_units='xy',label='Eig Vector 2', color=['b'],scale=2)
	plt.grid(visible=True, which='both',axis='both',color='k',linestyle='--',linewidth='0.8')
	plt.grid(which='minor', color='k', linestyle=':',linewidth='0.4')
	plt.xlim([x.min()-5,x.max()+5])
	plt.ylim([y.min()-1000,y.max()+1000])
	plt.minorticks_on()    
	plt.legend()
	#plt.show() #- uncomment to see plot in code
	
	# Plots saved in same directory
	plt.savefig("Problem-3-part1.png")
	plt.clf()

def leastSquare(input, x, y):
	""" Do something """
	B = np.zeros([len(input),2], dtype=int)
	B[:,0] = (np.vstack(x))[:,0]
	B[:,1] = 1
	y = np.vstack(y)
	# Evaluate v using v = (A^T A)^-1 A^T* y
	pInv_0 = np.matmul(np.transpose(B), B)
	pInv_1 = np.linalg.inv(pInv_0)
	pInv_2 = np.matmul(np.transpose(B), y)
	v = np.matmul(pInv_1, pInv_2)

	x_mean, y_mean = np.mean(x), np.mean(y)

	 # Find the coefficients of the quadratic a, b and c
	m=v[0] 
	b=v[1]

	#Plot the curve fitting
	for i in range(len(x)):
		plt.scatter(x[i], y[i], s=5)

	plt.plot(x,m*x+b,
		label='Linear Least Square Method',color='green',linestyle='--', linewidth=1.3)
	plt.title("Person's Age vs Health Insurance Cost w/ Linear LSq Fitting")
	plt.xlabel("Person's age")
	plt.ylabel("Charging")
	plt.grid(visible=True, which='both',axis='both',color='k',linestyle='--',linewidth='0.8')
	plt.grid(which='minor', color='k', linestyle=':',linewidth='0.4')
	plt.xlim([x.min()-5,x.max()+5])
	plt.ylim([y.min()-1000,y.max()+1000])
	plt.minorticks_on()   
	plt.legend()
	#plt.show() - uncomment to see plot when run

	# Plots saved in same directory
	plt.savefig("Problem-3-part2-LSq.png")
	plt.clf() 
	
def totalLeastSq(data,x,y):
	U = np.zeros([len(data),2], dtype=int)
	for i in range(len(data)):
		U[i,0] = data[i,0]
		U[i,1] = data[i,1]
	U = np.vstack(U)

	mult = np.matmul(np.transpose(U),U)

	eigV, eigVt = np.linalg.eig(mult)

	x_mean, y_mean = np.mean(x), np.mean(y)

	#Plot the curve fitting
	xd = U[:,0] ; yd =U[:,1]
	for i in range(len(xd)):
		plt.scatter(xd[i], yd[i], s=5)
	
	# eigenVector associated with smallest eigenValue is eigVt[0]
	# d= a* x_mean + b*y_mean
	a = eigVt[0,0] ; b = eigVt[1,0]
	d = a*x_mean + b*y_mean
	
	plt.plot(xd,(d-a*xd)/b,
		label='Linear Total Least Square Method',color='orange',linestyle='--', linewidth=1.3)
	plt.title("Person's Age vs Health Insurance Cost w/ Linear Total LSq Fitting")
	plt.xlabel("Person's age")
	plt.ylabel("Charging")
	plt.grid(visible=True, which='both',axis='both',color='k',linestyle='--',linewidth='0.8')
	plt.grid(which='minor', color='k', linestyle=':',linewidth='0.4')
	plt.xlim([x.min()-5,x.max()+5])
	plt.ylim([y.min()-1000,y.max()+1000])
	plt.minorticks_on()   
	plt.legend()
	#plt.show() #- uncomment to see plot when run

	# Plots saved in same directory
	plt.savefig("Problem-3-part2-TLSq.png")
	plt.clf() 

def ransac(data,x,y):
	# Randomly select two points
	# Hypothesize a model, use TLSq
	# Compute error Fn, E = a*x + b*y - d whre d = a*x_mean+b*y_mean
	# Select points Consistent with model, points +/- regressio line
	# Repeat process and verify loop

	# Number of samples
	s = len(data)
	# probability for outlier, inspected visually
	e = 0.2
	# desired probability for good sample
	p = 0.95
	# Number of iterations: 2 points will be selected per iteration
	# so no. of iteration will be half of number of samples
	nIt = s/2

	# Due to large sample size ran() function not runnable to get N 
	#N = ran(e=e, p=p, s=s); Let N = 1000 samples
	N=1000 

	# while percentage of number of inliers vs total points is not in 
	# desired probability range, loop through regression plotting
	count = 0
	visited = []
	v_lines = []
	# Now to see how many points in the data fall within this threshold
	# Let the threshold be 10 upon visual inspection of graph
	tH = 2000

	mB = 0	# slope of best fit line
	bB = 0  # intercept of best fit line

	A = np.zeros([len(data),2], dtype=int)
	for i in range(len(data)):
		A[i,0] = data[i,0]
		A[i,1] = data[i,1]
	A = np.vstack(A)

	xd = A[:,0] ; yd =A[:,1]
	x_mean, y_mean = np.mean(x), np.mean(y)

	while count < N:
		idx = random.randint(0,s-1)   # Find index for first random point
		pt1 = [xd[idx],yd[idx]]
		jdx = random.randint(0,s-1)   # Find index for second random point
		while jdx == idx:	  # Double check two indeces are not identical
			jdx = random.randint(0,s-1)
		pt2 = [xd[jdx],yd[jdx]]
		line = [pt1,pt2]
		v_lines.append(line)
		while line in v_lines:
			idx = random.randint(0,s-1)   # Find index for first random point
			pt1 = [xd[idx],yd[idx]]
			jdx = random.randint(0,s-1)   # Find index for second random point
			while jdx == idx:	  # Double check two indeces are not identical
				jdx = random.randint(0,s-1)
			pt2 = [xd[jdx],yd[jdx]]
			line = [pt1,pt2]
		xP = [xd[idx],xd[jdx]]
		yP = [yd[idx],yd[jdx]]
		M = [xP,yP]
		M = np.vstack(M)
		Mx = M[:,0]
		My = M[:,1]
		# to simiplify error calculation later on, yP is reduced by 1000
		#yP = [yP[0]/1000,yP[1]/1000]
		yP = [yP[0],yP[1]]
		num = (yP[1] - yP[0])
		den = float((xP[1]-xP[0]))
		if den == 0:
			den = 0.1
			
		m = num / den
		b = (yP[0]) - m* xP[0]
		
		# y = (d-a0*x)/b0 using TLSq
		# d = a0* x_mean + b0 * y_mean
		# b0 = (1/(m**2+1))**(1/2)
		# a0 = -1*b0*m
		# d = a0*x_mean + b0*y_mean
		
		# mx - y + b = 0

		# To simplify Error computation, Y is divided by 1000
		#yd = yd/1000
		
		E = np.zeros([len(data),1], dtype=int) 
		for i in range(len(x)):
			#E[i] = (a0*xd[i] + b*yd[i] - d)**2
			#E[i] = abs(m*xd[i]-yd[i]+b)/(m**2+1)**(1/2)
			E[i] = (m*(xd[i]-x_mean)-(yd[i]-y_mean))**2
		#E = E/1000000000000 # reduced by factor of 10^12 
		E = E/1000
		# print(yd[0])
		# print(E,'\n')
		# print(E.min(), E.max())
		
		# Counting for points that fall within error range
		#for idx in range(len(x)):
		inPoints = 0
		for i in range(len(E)):
			if E[i] <= tH:
				inPoints+=1
		
		# percentage of inliers in data
		p = float(inPoints / len(E))
		visited.append(float(p))
		#print(visited)
		
		if p == max(visited): #save the slope and intercept for best fit curve
			mB = m
			bB = b
		count+=1

	for i in range(len(x)):
		plt.scatter(xd[i], yd[i], s=5)
	#Mx,m*Mx+b
	yR = mB*xd+bB

	plt.plot(xd,yR,
	label='RANSAC Method',color='orange',linestyle='-', linewidth=1.5)

	# xL = xd ; yL= yd
	# np.delete(xL,idx) ; np.delete(yL,idx)
	# np.delete(xL,jdx) ; np.delete(yL,jdx)
	# for j in range(len(xL)):
	# 	plt.plot(xL[j],(d - a0*xL[j])/b0,color='green',linestyle='-', linewidth=0.5)

	plt.title("Person's Age vs Health Insurance Cost w/ RANSAC Fitting: trial %d" % count)
	plt.xlabel("Person's age")
	plt.ylabel("Charging")
	plt.grid(visible=True, which='both',axis='both',color='k',linestyle='--',linewidth='0.8')
	plt.grid(which='minor', color='k', linestyle=':',linewidth='0.4')
	plt.xlim([xd.min()-5,xd.max()+5])
	plt.ylim([yd.min()-20000,yd.max()+10000])
	plt.minorticks_on()   
	plt.legend()
	plt.show()

	# Explain draw back/advantages for each in report
	# Explain Solution steps and explain better choice 
		# for outlier rejection for each case

def main():
	"""	Read a csv file to relate a person's age with insurance cost.
	Compute covariance matrix, eigenvalues, and eigenvectors 
	plot eigegnvectors. 
	Fit a line to data via least square method, total least square 
	method and RANSAC. Plot result for each method, explain drawback
	and advantage for each in report.
	Explain steps of each solution and discuss which is better 
	to reject outliers on report.  """

	# Read CSV columns age and charging into arrays
	# x for age and y for charging

	
	file = pd.read_csv('ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
	x= file.age.tolist()
	y = file.charges.tolist()
	
	data = []
	
	for i in range(len(x)):
		data.append([x[i],y[i]])
	
	data = np.vstack(data) 
	
	# Problem 3: part a)		
	# Plot data with eigenvectors
	"""uncomment the line below to run part a"""
	#cov_eigV(data,x,y)		

	# Problem 3: part b)
	"""uncomment the lines below that call functions to run the methods
	in parts in b"""

	# Method 1: Fit a line using linear least square method	
	#leastSquare(data,x,y)
	
	# Method 2: Fit a line using Total least square method
	#totalLeastSq(data,x,y)
	
	# Method 3: Fit a line using RANSAC
	#ransac(data,x,y)
	  

if __name__ == '__main__':
    main()
