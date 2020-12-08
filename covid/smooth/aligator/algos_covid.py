#ref: http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/

import time
import pickle

#import pywt
import numpy as np
import matplotlib as mpl
mpl.rc('xtick', labelsize=30)     
mpl.rc('ytick', labelsize=30)
mpl.rcParams.update({'errorbar.capsize': 2})
import matplotlib.pyplot as plt
import sys

import math
from math import fabs
from scipy.stats import truncnorm

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random
from patsy import dmatrix

#import arrows
#import covid
#import covid_lin
#import fusedlasso

sim_steps=0
sigma = 0
haar_coeff = []
thresholded_coeff = []
thresholded_norm_squared = 0.
uthresh = 0
first = True

#my_file = open('algosnorm.txt','w') 

def calculate_tv(theta):
    sim_steps = len(theta)
    C = theta[1:]
    C = np.fabs(theta[:sim_steps-1] - C)
    return np.sum(C)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10, steps=1):
    return truncnorm.rvs(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size = steps)

def wv_smooth(y,sigma,wavelet='haar',mod="reflect"):
    coeff = pywt.wavedec( y, wavelet, mod )
    uthresh = sigma * np.sqrt( 2*np.log( len( y ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    wv = pywt.waverec( coeff, wavelet, mod )
    return wv

def set_epoch_and_sigma(a,b):
    global sim_steps
    global sigma
    global haar_coeff
    global thresholded_coeff
    global thresholded_norm_squared
    global uthresh
    
    sim_steps = a
    sigma = b
    #haar_coeff = np.array([0.]*sim_steps,)
    #thresholded_coeff = np.array([0.]*sim_steps)

    #thresholded_norm_squared = 0.
    uthresh = sigma*np.sqrt(20*math.log(sim_steps))



def up_sample(x,length):
    rep = int(length/len(x))
    ctr = 0
    y = np.array([x[-1]]*length)
    
    while ctr*rep < length:
        try:
            y[ctr*rep:(ctr*rep)+rep] = np.array([x[ctr]]*rep)
            ctr = ctr+1
        except:
            break

    return y

def soft_threshold(x):
    global uthresh
    if(np.fabs(x) > uthresh):
        if x < 0:
            return x + uthresh
        else:
            return x - uthresh
    else:
        return 0


def constant_signal_source(mean, num_steps):
    return np.array([mean]*num_steps)

def bump_source(means, reps):
    y = np.array([])
    for i in range(len(means)):
        y = np.append(y,constant_signal_source(means[i],reps[i]))
    return y

def doppler(x, epsilon):
    #return np.sqrt(x*sim_steps*(sim_steps-x*sim_steps))*\
    return np.sin(2*np.pi*(1+epsilon)/(x+epsilon))
           #/(sim_steps*0.5)
    
def heavisine(x):
    return 4*np.sin(4*np.pi*x) - np.sign(x-0.3) - np.sign(0.72-x)

def blocks(t):
    t_j = [0.1,0.13,0.15,0.23,0.25,0.40,0.44,0.65,0.76,0.78,0.81]
    h_j = [4,-5,3,-4,5,-4.2,2.1,4.3,-3.1,5.1,-4.2]
    
    y = np.zeros(len(t))
    
    
    for j in range(len(t_j)):
        d = t - t_j[j]
        k = 0.5*(1+np.sign(d))
        y = y + h_j[j]*k
        
    return y

    
def spline(x):
    y = dmatrix("bs(x, knots=(10,30,50,70,75,80,85,90,95), degree=3, df = 12, include_intercept=False)", {"x": x})
    b = np.array([-1, -1, -1, -0.5, 0, -5,5,-5,5,-5,5,-5,-5])
    z = np.dot(y,b)
    z = -1*z
    i = np.where(z > 3.6)[0]
    z[i] = 3.6
    return z
    
def pad(x):
   t = len(x)
   deficit = int((2**np.ceil(np.log2(t))) - t)
   y = np.pad(x,(0,deficit), mode='constant')
   return y


def demo_filtered_results(theta,sigma,sob=False):
    global uthresh
    global thresholded_norm_squared
    global haar_coeff
    global thresholded_coeff

       

    steps = len(theta)
    tv = calculate_tv(theta)
    uthresh = sigma*np.sqrt(2*np.log(steps))
    y = theta + np.random.normal(0, sigma, steps)

    haar_coeff = haar_coeff * 0
    thresholded_coeff = thresholded_coeff * 0
    thresholded_norm_squared = 0.
    
    sobolev = calculate_sobolev(theta) #* math.sqrt(steps)
    if sob:
        tv = math.sqrt(steps) * calculate_sobolev(theta)
    

    if sob:
        width_ogd = min(int(np.ceil((steps*np.log(steps)) ** (1 / 3) * sigma ** (2 / 3) / sobolev ** (2 / 3))), steps)
        width_ma = min(int(np.ceil(steps ** (1 / 3) * sigma ** (2 / 3) / sobolev ** (2 / 3))), steps)
    else:
        width_ogd = min(int(np.ceil(np.sqrt(steps*np.log(steps))*sigma/tv)),steps)
        width_ma =  min(int(np.ceil(np.sqrt(steps)*sigma/tv)),steps)
    #    width_ma = width_ogd
    #width_ma = min(int(np.ceil(np.sqrt(steps) * sigma / tv)),
                   #steps)  # MA if I use sigma here, then it will give n^1/3 for bump


    ogd_est = ogd.ogd(y,width_ogd)
    ma_est = movingmean.MA(y,width_ma)
    
    print('error ogd: '+str(np.sum((ogd_est-theta)**2)))
    print('error ma: '+str(np.sum((ma_est-theta)**2)))
    
    return arrows.shoot_arrows(y,sigma,tv, uconst=2, rconst=1),\
           ogd_est,\
           ma_est


def generate_and_run_trials(theta, tv, sigma,B):
    global uthresh
    global thresholded_norm_squared
    global haar_coeff
    global thresholded_coeff
    
    
    steps = len(theta)
    uthresh = sigma*np.sqrt(2*np.log(steps))

    num_trials = 5
    
    error_ofs = 0
    error_alig = 0
    error_wv = 0


    for i in range(num_trials):
        y = theta + get_truncated_normal(0, sigma, -1*sigma, sigma, steps)

        
        
        print('trial: '+str(i+1))
        
        #alig1 = aligator.run_aligator(steps,y,0,B,pow(10,-4))
        #alig2 = aligator.run_aligator(steps,np.flip(y),0,B,pow(10,-4))

        #alig = (alig1 + np.flip(alig2))/2
        

        e1 = np.sum((aligator.run_aligator(steps,y,np.arange(0,steps),0,B,pow(10,-4))-theta)**2) ## original
        #e1 = np.sum((alig-theta)**2)
        
                
        
        error_alig = error_alig + e1
        
        e2 = np.sum((arrows.shoot_arrows(y,sigma,tv, uconst=2, rconst=1)-theta)**2)
        error_ofs = error_ofs + e2
        
        e3 = np.sum((wv_smooth(y,sigma)-theta)**2)
        error_wv = error_wv + e3

        print('****************')

    return error_alig/num_trials, error_ofs/num_trials, error_wv/num_trials

def tune_lambda(theta,y,sim_steps):
    grid = [0.125,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,10,12,14,16]
    
    best_lamb = 0.125;
    minim = 999999999;
    
    for lamb in grid:
        z = fusedlasso.run_fusedlasso(sim_steps,y,lamb)
        error = np.mean((z-theta)**2)
        
        if error < minim:
            minim = error
            best_lamb = lamb
    
    return best_lamb


'''def tune_lambda(theta,y,sim_steps):
    lamb = pow(2,-5)
    
    ctr = 0;
    b_ctr = 0;
    minim = 99999999;
    
    while (lamb <= sim_steps):
        z = fusedlasso.run_fusedlasso(sim_steps,y,lamb)
        error = np.mean((z-theta)**2)
        
        if error < minim:
            minim = error
            b_ctr = ctr

        ctr = ctr+1
        lamb = lamb * 2

    lamb = pow(2,-5) * pow(2,b_ctr)
    return lamb'''
    

def generate_and_run_trials2(theta, tv, sigma,B,z=0):
    global uthresh
    global thresholded_norm_squared
    global haar_coeff
    global thresholded_coeff
    
    
    steps = len(theta)
    uthresh = sigma*np.sqrt(2*np.log(steps))

    num_trials = 5
    
    error_ofs = 0
    error_alig = 0
    error_wv = 0
    error_fl = 0

    #y = theta + get_truncated_normal(0, sigma, -1*sigma, sigma, steps)
    y = theta + get_truncated_normal(0, sigma, -3*sigma, 3*sigma, steps)
    #y = theta + np.random.normal(0,sigma,steps)
    lam = tune_lambda(theta,y, steps)
    
    print("optimal lambda = "+str(lam))

    for i in range(num_trials):
        #y = theta + get_truncated_normal(0, sigma, -1*sigma, sigma, steps)
        y = theta + get_truncated_normal(0, sigma, -3*sigma, 3*sigma, steps)

        print('trial: '+str(i+1))

        #e1 = np.sum((aligator.run_aligator(steps,y,0,B,pow(10,-4))-theta)**2)
        #error_alig = error_alig + e1
        
        num_perm = pow(2,8)
        alig1 = aligator.run_aligator(steps,y,np.arange(0,steps),z,B,pow(10,-4))
        alig2 = aligator.run_aligator(steps,y,np.flip(np.arange(0,steps)),z,B,pow(10,-4))

        alig = (alig1 + alig2)/2

        
        '''for k in range(num_perm):
            index = np.random.permutation(np.arange(0,steps))
            alig = alig + aligator.run_aligator(steps,y,index,0,B,pow(10,-4))
        
        alig = alig/(num_perm+1)'''
        
        e1 = np.sum((alig-theta)**2)
        error_alig = error_alig + e1
        
        e2 = np.sum((arrows.shoot_arrows(y,sigma,tv, uconst=2, rconst=1)-theta)**2)
        error_ofs = error_ofs + e2
        
        e3 = np.sum((wv_smooth(y,sigma)-theta)**2)
        error_wv = error_wv + e3
        
        e4 = np.sum((fusedlasso.run_fusedlasso(steps,y,lam)-theta)**2)
        error_fl = error_fl + e4
        

        print('****************')

    return error_alig/num_trials, error_ofs/num_trials, error_wv/num_trials, error_fl/num_trials


def calculate_tv(theta):
    sim_steps = len(theta)
    C = theta[1:]
    C = np.fabs(theta[:sim_steps-1] - C)
    return np.sum(C)

def discretize(theta,n):
    length = len(theta)
    lc = int(np.floor(length/n))
    sub_theta = []
    for i in range(n):
        sub_theta.append(theta[((i+1)*lc)-1])
    sub_theta = np.array(sub_theta)
    tv = calculate_tv(sub_theta)
    return sub_theta, tv


def sub_sample_and_run(theta,sigma,B):
    error_alig = []
    error_arr = []
    error_wv = []
    i = 128
    n = len(theta)
    tv0 = calculate_tv(theta)
    
    while i<=n:
        print('sampling level: 2^'+str(np.log2(i)))
        sub_theta,tv = discretize(theta,i)
        tv = tv0 # using the end TV
        print('-------------------------------------------')
        
        ali, arr, wv = generate_and_run_trials(sub_theta,tv,sigma,B)

        error_alig.append(ali)
        error_arr.append(arr)
        error_wv.append(wv)
        i = i*2

    return np.array(error_alig),np.array(error_arr),np.array(error_wv)

def sub_sample_and_run2(theta,sigma,B,z=0):
    error_alig = []
    error_arr = []
    error_wv = []
    error_fl = []
    i = 128
    n = len(theta)
    tv0 = calculate_tv(theta)
    
    while i<=n:
        print('sampling level: 2^'+str(np.log2(i)))
        sub_theta,tv = discretize(theta,i)
        tv = tv0 # using the end TV
        print('-------------------------------------------')
        
        ali, arr, wv, fl = generate_and_run_trials2(sub_theta,tv,sigma,B,z)

        error_alig.append(ali)
        error_arr.append(arr)
        error_wv.append(wv)
        error_fl.append(fl)
        i = i*2

    return np.array(error_alig),np.array(error_arr),np.array(error_wv), np.array(error_fl)
