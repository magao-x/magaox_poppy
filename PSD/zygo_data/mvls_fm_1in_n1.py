#load modules
import numpy as np
from astropy import units as u
from astropy.io import fits
import time
from skimage.draw import draw

# PSD code
from model_kit import psd_functions as psd

file_dir = 'flat_mirrors/fixed_fits/'
nfm = 1
tot_step = 5
k_side = 300 # bigger than this optic but could help fill in missing info?

for nstep in range(0, tot_step+1):
    # open the data file
    opt_fits = fits.open(file_dir+'flat_1in_ca80_n{0}_step{1}_surf.fits'.format(nfm, nstep))[0]
    opt_data = (opt_fits.data * u.micron).to(u.nm)
    dx = (opt_fits.header['LATRES'] * u.m).to(u.mm)
    opt_side = np.shape(opt_data)[0]
    
    # open the associated dust mask
    opt_mask = fits.open(file_dir+'flat_1in_ca80_n{0}_step{1}_mask_LS.fits'.format(nfm, nstep))[0].data
    
    # do the scargle
    psd_name = 'lspsd_fm_1in_n{0}_step{1}'.format(nfm, nstep)
    print('Optic test: {0}'.format(psd_name))
    lspsd, lspsd_parms = psd.mvls_psd(data=opt_data, mask=opt_mask, dx=dx,
                                      k_side=k_side, print_update=True,
                                      write_psd=True, psd_name=psd_name)
    print('----------') # print breaker between surfaces

print('All FM 1in n1 steps tested with 2D Lomb-Scargle.')
