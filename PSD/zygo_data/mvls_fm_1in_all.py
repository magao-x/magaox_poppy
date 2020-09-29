#load modules
import numpy as np
from astropy import units as u
from astropy.io import fits
import time
from skimage.draw import draw

# PSD code
from model_kit import psd_functions as psd

file_dir = 'flat_mirrors/fixed_fits/'
tot_fm = 8
tot_step = 6
flat_label = '1in'

for nfm in range(0, tot_fm):
    for nstep in range(0, tot_step):
        # open the data file
        opt_fits = fits.open(file_dir+'flat_{0}_n{1}_80CA_step{2}_zern_surf.fits'.format(flat_label, nfm+1, nstep))[0]
        opt_data = (opt_fits.data * u.micron).to(u.nm)
        dx = (opt_fits.header['LATRES'] * u.m).to(u.mm)
        opt_side = np.shape(opt_data)[0]
        
        # open the associated dust mask
        opt_mask = fits.open(file_dir+'flat_{0}_n{1}_80CA_step{2}_dust_mask.fits'.format(flat_label, nfm+1, nstep))[0].data
        
        # do the scargle
        psd_name = 'lspsd_fm_{0}_n{1}_step{2}_zc'.format(flat_label, nfm+1, nstep)
        print('Optic test: {0}'.format(psd_name))
        lspsd, lspsd_parms = psd.mvls_psd(data=opt_data, mask=opt_mask, dx=dx,
                                        k_side=opt_side, print_update=True,
                                        write_psd=True, psd_name=psd_name)
        print('----------') # print breaker between surfaces

print('All FM {0} steps tested with 2D Lomb-Scargle.'.format(flat_label))

