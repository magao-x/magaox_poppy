'''
zernike.py
This modular function file has the following goals:
* [calc_zernike_phase] Calculates Zernike phase based on circular aperture inserted
* [calc_zernike_proj] Calculates the projection weight of a Zernike on a surface. Phase only.
* [remove_ptt] Insert data, returns new data with PTT removed (and the projection weights)
'''

import numpy as np
import copy
from astropy import units as u
from scipy.special import factorial

######################################

def calc_zernike_phase(n, m, mask):
    # sets up Normalized Zernike function. 
    # Code heavily influenced from Michael Hart's OPTI 528 course.
    ma = np.absolute(m)
    if (n<ma) or (n<0): # zernike integers
        raise Exception('Try again. n must be non-negative and n>=|m|.\nGiven: n={0}, m={1}'.format(n,m))
    
    s = (n-ma)/2 # radial polynomial upper bound
    if (s-np.floor(s) != 0):
        raise Exception('Try again. n-m must be even.\nGiven: n={0}, m={1}'.format(n,m))
    
    # Set up the grid to calculate rho and theta
    ap_diam = np.shape(mask)[0]
    c1 = -((ap_diam-1)/2)
    c2 = ap_diam+c1-1
    x = np.linspace(c1, c2, ap_diam)
    y = np.linspace(c1, c2, ap_diam)
    xv, yv = np.meshgrid(x,y)
    yv = yv[::-1]
    rho = np.sqrt((xv**2) + (yv**2))/ap_diam*2
    theta = np.arctan2(yv,xv)

    # Calculate the radial polynomial
    R = 0
    for k in range(0,np.int(s)+1): # need the +1 to include the upper bound sum number
        mult_num = ((-1)**k) * factorial(n-k) * (rho**(n-(2*k)))
        mult_den = factorial(k) * factorial( ((n+ma)/2) - k) * factorial( ((n-ma)/2) - k)
        R = R + (mult_num/mult_den)

    # Calculate the Zernike polynomial
    if m>0: # "even" zernike
        zern = np.sqrt(2*(n+1)) * R * mask * np.cos(ma*theta)
    elif m<0: # odd zernike
        zern = np.sqrt(2*(n+1)) * R * mask * np.sin(ma*theta)
    else: # m==0
        zern = np.sqrt(n+1) * R * mask
        
    # No need to tighten the matrix because passed in the mask
    return zern * u.radian

def calc_zernike_proj(data, mask, zn, zm):
    # check the units of data that it is in phase otherwise this doesn't work
    if data.unit != u.radian:
        raise Exception('Data units must be in phase (radians)')
    # calculate the zernike surface
    zern = calc_zernike_phase(zn, zm, mask)
    # do the dot product
    vec1d = np.product(data.shape)
    dp_num = np.dot(np.reshape(data.value, (vec1d)), np.reshape(zern.value,(vec1d)))
    dp_den = np.dot(np.reshape(zern.value, (vec1d)), np.reshape(zern.value,(vec1d)))
    return (zern, dp_num/dp_den)

def remove_ptt(data, mask):
    zn = [0, 1, 1]
    zm = [0, -1, 1]
    w_ptt = []
    rem_ptt = copy.copy(data)
    for i in range(0, len(zn)):
        zern, zproj = calc_zernike_proj(data=data, mask=mask, zn=zn[i], zm=zm[i])
        rem_ptt = rem_ptt - (zern*zproj)
        w_ptt.append(zproj)
    return (rem_ptt, w_ptt)


