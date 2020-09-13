
#load modules
import numpy as np
import cupy as cp
np.set_printoptions(suppress=True) # scientific notation gets annoying
from astropy import units as u
from astropy.io import fits
import time
from skimage.draw import draw

# PSD code
import h5py
import copy
import os

# import personal code
from model_kit import datafiles as dfx
from model_kit import dust
from model_kit import psd_functions as psd

# choose the files here
nflat = 0
nstep = 0

# open up all the data
opt_parms = {'ca' : 80, # of the 80% CA given
             'ovs': 4096,
             'surf_units': u.micron,
             'ring_width': 5,
             'diam_ca100': 25*u.mm,
             'label': '1in'} # useful for PSD

fm_list = [1, 2, 3]
tot_step=6
# rewrite this folder name
fits_folder = 'flat_mirrors/fixed_fits/'

surf_cube = [] # order: mirror num, step num, (data shape)

for nf in range(0, len(fm_list)):
    fm_num = fm_list[nf]
    for ns in range(0, tot_step):
        fits_file = fits_folder+'flat_{0}_ca{1}_n{2}_step{3}'.format(opt_parms['label'],
                                                                     opt_parms['ca'], fm_num, ns)

        if ns==0: # initialize first time
            mask_fits = fits.open(fits_file+'_mask.fits')[0]
            mask = mask_fits.data
            opt_parms['wavelen'] = mask_fits.header['wavelen']*u.m
            opt_parms['latres'] = mask_fits.header['latres']*u.m/u.pix
            data_set = np.zeros((tot_step, np.shape(mask)[0], np.shape(mask)[0]))

        # open data
        sd = fits.open(fits_file+'_surf.fits')[0].data
        # write data to matrix
        data_set[ns] = sd

    # apply matrix units
    surf_cube.append(data_set*opt_parms['surf_units'])

print('Data all loaded')

# load the dust
# center of dust based on n8 flat
dcen_x = ([[ 0,  81, 117, 153,  0,   0],
           [91, 127, 163,   0,  0,   0],
           [ 0,   0,  14,  49, 86, 122]])
dcen_y = [13, 27, 97]
dust_radius=10 # oversize for everyone, not many pixels to lose
x_offset = [5, -3, 4, 2, 2, 1, -1, 0]
y_offset = [0,  2, 2, 1, 2, 1, 1, 0]

# step 0 big dust mask
box_corner_c = [210, 202, 208, 206, 206, 206, 202, 202]
box_corner_r = [114, 114, 114, 114, 114, 114, 114, 114]
box_rsize =    [28,   28,  28,  28,  28,  28,  28,  28]
box_csize =    [ 8,   16,  10,  12,  12,  12,  16,  16]

# marked optics
mark_optic = [2,4]
# build dust lists
dust_all_set = []

# This loop masks the dust per step for all the optics
for nf in range(0, len(fm_list)): # choose optic
    fm_num = fm_list[nf]
    
    # build all dust mask
    all_dust = np.ones((np.shape(dcen_x)[1], mask.shape[0], mask.shape[1]))
    
    # analyzing per step
    for ns in range(0, np.shape(dcen_x)[1]):
        mask_step = np.ones_like(mask).astype(float) # initialize step mask
        
        # Apply specific dust mask present in all steps at same place
        if fm_num == 2:
            mark_cen_x = 197
            mark_cen_y = 138
            mark_radius = 8
            mark_coord = draw.circle(r=mark_cen_y, c=mark_cen_x, radius=mark_radius)
            mask_step[mark_coord] = np.nan
        elif fm_num == 4:
            mark_cen_x = 49
            mark_cen_y = 96
            mark_radius = 8
            mark_coord = draw.circle(r=mark_cen_y, c=mark_cen_x, radius=mark_radius)
            mask_step[mark_coord] = np.nan
        
        # for first step mask ONLY, need to apply a box to clean up a single dust piece.
        if ns==0:
            mask_step[box_corner_r[nf]:box_corner_r[nf]+box_rsize[nf], 
                      box_corner_c[nf]:box_corner_c[nf]+box_csize[nf]] = np.nan
        
        # build each dust location at the step
        for nd in range(0, np.shape(dcen_x)[0]): 
            dmc = np.zeros_like(mask)
            if dcen_x[nd][ns] != 0: # make mask if value present. Otherwise, skip.
                dm_coord = draw.circle(r=dcen_y[nd]+y_offset[nf], 
                                       c=dcen_x[nd][ns]+x_offset[nf],
                                       radius=dust_radius)
                dmc[dm_coord] = True
            mask_step[dmc==True] = np.nan
        all_dust[ns] = mask_step
    print('All dust masked for fm{0}'.format(fm_num))
    
    # add all the dust objects to the lists
    dd = dust.dust_map(name='fm{0}_dust_all'.format(fm_num))
    dd.load_mask(mask_set=all_dust)
    dust_all_set.append(dd)

print('Dust maps all set up')

# Now, let's begin the Lomb-Scargle. (scargle noises begin)
print('Initializing variables for Lomb-Scargle')
# set the data up for the lomb-scargle
surf_data = surf_cube[nflat][nstep]
# write the hann window
data_side = np.shape(mask)[0]
surf_hann = psd.han2d((data_side, data_side)) * surf_data

# build spatial matrix
cen = int(data_side/2)
my, mx = np.mgrid[-cen:cen, -cen:cen]
tm = mx*opt_parms['latres'].value
tn = my*opt_parms['latres'].value

# create the vectors for the data
# see the nflat and nstep value at start of script
mt = dust_all_set[nflat].mask[nstep]*mask
mask_filter = np.where(mt==1)
tmv = tm[mask_filter]
tnv = tn[mask_filter]
yv = surf_hann[mask_filter].value

# Build the wkl matrices
k_side = 300
dk = 1/(k_side * opt_parms['latres'])
k_cen = int(k_side/2)
yy, xx = np.mgrid[-k_cen:k_cen, -k_cen:k_cen]
wk = xx*dk.value
wl = yy*dk.value

# vectorize wk and wl to make the math easier
wkv = np.reshape(wk, k_side**2)
wlv = np.reshape(wl, k_side**2)

# start the timer
print('Variables ready, begin scargling (and timing)...')
start_time = time.process_time()
# Calculate tau parameter for Lomb-Scargle
k_tot = k_side**2
tau = np.zeros((k_tot)) # initialize matrix
for nk in range(0, k_tot):
    wkn = wkv[nk]
    wln = wlv[nk]
    tau_num = 0 # initialize
    tau_denom = 0
    for p in range(0, tmv.shape[0]):
        inner_calc = (wkn*tmv[p]) + (wln*tnv[p])
        tau_num = tau_num + cp.cos(2*inner_calc)
        tau_denom = tau_denom + cp.sin(2*inner_calc)
    tau[nk] = 0.5 * cp.arctan(tau_num/tau_denom)
    
# initialize the variables for the PSD
akl = np.zeros((k_tot))
bkl = np.zeros((k_tot))
for nk in range(0, k_tot):
    wkn = wkv[nk]
    wln = wlv[nk]
    taun = tau[nk]
    ak_num = 0
    ak_denom = 0
    bk_num = 0
    bk_denom = 0
    for p in range(0, tmv.shape[0]):
        inner_calc = (wkn*tmv[p]) + (wln*tnv[p]) - taun
        akcos = cp.cos(inner_calc)
        ak_num = ak_num + (yv[p]*akcos)
        ak_denom = ak_denom + (akcos**2)
        bksin = cp.sin(inner_calc)
        bk_num = bk_num + (yv[p]*bksin)
        bk_denom = bk_denom + (bksin**2)
    akl[nk] = ak_num/ak_denom
    bkl[nk] = bk_num/bk_denom

# Calculate the PSD
psd_ls = (akl**2) + (bkl**2)
psd_ls = np.reshape(psd_ls, (k_side, k_side))

# calculate the time
tot_time = time.process_time() - start_time
print('Scargling and PSD completed')
hr1 = 60*60
hrs = int(tot_time/hr1)
mins = int((tot_time%hr1) / 60)
secs = (tot_time%hr1)%60
print('Lomb-Scargle time: {0:2}h {1:2}m {2:.2f}s'.format(hrs, mins, secs))

# Save the PSD as a FITS file
hdr = fits.Header()
hdr['name'] = ('fm{0}_n{1}_s{2} PSD LS'.format(opt_parms['label'], nflat, nstep),
               'test file')
hdr['dx'] =  (opt_parms['latres'].value, 
              'optic spatial spacing, [{0}]'.format(opt_parms['latres'].unit))
hdr['dk'] = (dk.value, 
             'Spatial frequency spacing, [{0}]'.format(dk.unit))
hdr['surfunit'] = (opt_parms['surf_units'],
                   'surface units of data')
hdr['ca'] = (opt_parms['ca'], 'clear aperture percent')
hdr['diam_ca'] = (opt_parms['diam_ca100'].value*opt_parms['ca']/100,
                  'clear aperture diameter')
fits.writeto(os.getcwd()+'/LSPSD_test_fm{0}_n{1}_s{2}.fits'.format(opt_parms['label'], nflat, nstep),
             psd_ls, hdr, overwrite=True)


