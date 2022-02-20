#!/usr/bin/python3

# ========================================
# ENPM673 Spring 2022: Perception for Autonomous Robotics
# HW 1: Question 4
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

def main():
	"""Compute the SVD given the matrix"""
	x = [0,5,150,150,5]
	y= [0,5,5,150,150]
	xp = [0,100,200,220,100]
	yp = [0,100,80,80,200]

	A = [[-x[1],-y[1],-1,0,0,0,x[1]*xp[1],y[1]*xp[1],xp[1]],
		 [0,0,0,-x[1],-y[1],-1,x[1]*yp[1],y[1]*yp[1],yp[1]],
		 [-x[2],-y[2],-1,0,0,0,x[2]*xp[2],y[2]*xp[2],xp[2]],
		 [0,0,0,-x[2],-y[2],-1,x[2]*yp[2],y[2]*yp[2],yp[2]],
		 [-x[3],-y[3],-1,0,0,0,x[3]*xp[3],y[3]*xp[3],xp[3]],
		 [0,0,0,-x[3],-y[3],-1,x[3]*yp[3],y[3]*yp[3],yp[3]],
		 [-x[4],-y[4],-1,0,0,0,x[4]*xp[4],y[4]*xp[4],xp[4]],
		 [0,0,0,-x[4],-y[4],-1,x[4]*yp[4],y[4]*yp[4],yp[4]]]
	A = np.vstack(A)
	# SVD(A) = U E V.T
	
	U = np.matmul(A,np.transpose(A))
	V = np.matmul(np.transpose(A),A)
	eV0, eUt = np.linalg.eig(U)
	eV1,eVt = np.linalg.eig(V)
	#U = np.vstack(eUt)
	E = eV1
	E = np.vstack(E)
	I = np.identity(len(E))
	
	eVt =np.vstack(eVt)
	vT = np.transpose(eVt)
	#vT = np.vstack(vT)
	#print(E[:,0])
	E = np.diag(E[:,0])

	# SVD can now be computed as the last row of diagonal matrix and V.T
	#prt1 = np.matmul(E,vT)
	#svdA = np.matmul(U,prt1)
	a = np.vstack(E[:,len(E[0])-1])
	b = np.vstack(vT[:,len(V[0])-1])
	svdA = [a[:,0],b[:,0]]
	#svdA = np.vstack(svdA)
	print(svdA)

if __name__ == '__main__':
    main()
