"""Taking fixed mean/varance noise from an input file,
   Uses this to nudge parameters in time
   Integrate an ODE with these fluctuating parameters
   N>>1 times with normally distributed initial conditions
   Plot a density estimate from the histogram."""


import time
import math
import gc
import numpy as np
import cupy as cp
import cv2
from scipy.interpolate import interp1d

def eval_dr(r_t, params, timestep):
    """evaluate pathwise increment at r=r(t), with params p=p(t)"""

    dr_dt = cp.array([ r_t[:, 0]*params[0] - r_t[:, 1],\
    	               r_t[:, 0] - r_t[:, 2],\
    	               r_t[:, 1] - r_t[:, 3],\
    	               r_t[:, 2] - r_t[:, 4],\
    	               r_t[:, 3] - r_t[:, 5],\
    	               r_t[:, 4] - r_t[:, 6],\
    	               r_t[:, 5] - r_t[:, 7],\
    	               r_t[:, 6] - r_t[:, 8],\
    	               r_t[:, 8] * (r_t[:, 7] - params[2]) * params[1] + params[3]])


    return cp.transpose(timestep*dr_dt)


def pdf_cp(r_x, r_y, fft_kernel):
    """evaluate probability density function via fft method"""
    res_k = fft_kernel.shape[0]
    half_res = int(res_k/2)

    scaled_x, scaled_y = (res_k*r_x/cp.amax(r_x)).astype(int), (res_k*r_y/cp.amax(r_y)).astype(int)
    fft_xy = cp.fft.fft2(cp.histogram2d(scaled_x-half_res,scaled_y-half_res,bins=(res_k, res_k))[0])

    return 256*cp.abs(cp.fft.ifftshift(cp.fft.ifft2(cp.multiply(fft_xy, fft_kernel))))/len(r_x)


def rk4_cp(r_t, params, tau, timestep, f_t, fft_kernel):
    """runge kutta scheme"""

    for j in range(tau):
        iterate = 2*j
        dr_1 = eval_dr(r_t, params, timestep) + f_t[iterate, :] * r_t

        iterate += 1
        dr_2 = eval_dr(r_t+0.5*dr_1, params, timestep) + f_t[iterate, :] * r_t
        dr_3 = eval_dr(r_t+0.5*dr_2, params, timestep) + f_t[iterate, :] * r_t

        iterate += 1
        dr_4 = eval_dr(r_t+dr_3, params, timestep) + f_t[iterate, :] * r_t
        r_t +=  (dr_1 + dr_4 + 2.0*(dr_2 + dr_3))/6.0

        pdf = pdf_cp(r_t[:,0]-cp.amin(r_t[:,0]), r_t[:,4 ]-cp.amin(r_t[:, 4]), fft_kernel)
        cv2.imshow('',cp.asnumpy(pdf))
        cv2.waitKey(1)

        time.sleep(0.02)

    return cp.asnumpy(r_t)



if __name__ == '__main__':
    gc.enable()

    #load the data from file
    dat1 = np.loadtxt('setOf3_3.txt')
    dat2 = np.loadtxt('setOf3_2.txt')
    dat3 = np.loadtxt('setOf3_1.txt')

    print("Integrating trajectories")

    #dynamical system parameters
    PARAMS = [0.3, 4.0, 2.0, 0.1]

    N_DAT = 8000
    TAU = 8*N_DAT
    DELTA_T = 0.004
    #keep below 0.005
    SIGMA = 0*0.1
    DELTA_0 = 0.05
    N_TRAJ = 2**18
    BINS=512
    BW=8

    #calculate and scale random forcing
    DAT=np.zeros((N_DAT*8,9))
    DAT[:,:3]  = dat1[1:N_DAT*8+1,:3]-dat1[:N_DAT*8,:3]
    DAT[:,3:6] = dat2[1:N_DAT*8+1,:3]-dat2[:N_DAT*8,:3]
    DAT[:,6:9] = dat3[1:N_DAT*8+1,:3]-dat3[:N_DAT*8,:3]


    DAT = DAT/np.std(DAT)


    T_LR=np.linspace(0,DELTA_T*TAU,N_DAT)
    T_HR=np.linspace(0,DELTA_T*TAU,2*TAU)


    F_T = np.zeros((2*TAU, 9))

    #interpolate data
    for i in range(8):
        LIN_D = interp1d(T_LR, DAT[N_DAT * i: N_DAT* (i+1),1], kind='linear')
        F_T[:,i] = SIGMA*math.sqrt(DELTA_T)*LIN_D(T_HR)

    T_F = 32000
    INIT = np.zeros((N_TRAJ, 9))
    INIT[:,-1] = 10.0
    INIT +=  DELTA_0 * np.random.randn(N_TRAJ, 9)

    GRID = np.mgrid[-BINS/2:BINS/2,-BINS/2:BINS/2]
    KERNEL = cp.asarray(np.exp(-(GRID[0,:]**2+GRID[1,:]**2) / BW**2))
    FFT_K = cp.fft.fft2(KERNEL)



    TRAJ = rk4_cp(cp.asarray(INIT), PARAMS, T_F, DELTA_T, cp.asarray(F_T[:2*T_F+1,:]), FFT_K)
