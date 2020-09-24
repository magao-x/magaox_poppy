#load modules
import numpy as np
from astropy import units as u
from astropy.io import fits
import time
from skimage.draw import draw

# PSD code
from model_kit import psd_functions as psd

# open the files
# import the OAP data
file_dir = 'oaps/oap_coated/'
n_test = 4

oap_fits = fits.open(file_dir+'oap{0}_centered_80CA_surf.fits'.format(n_test))[0]
oap_data = (oap_fits.data * u.micron).to(u.nm)
dx = (oap_fits.header['LATRES'] * u.m).to(u.mm)
oap_side = np.shape(oap_data)[0]
oap_mask = fits.open('oaps/oap_coated/oap{0}_centered_80CA_mask_LS.fits'.format(n_test))[0].data

k_side = oap_side
psd_name = 'lspsd_oap{0}'.format(n_test)
print('Optic test: {0}'.format(psd_name))
lspsd, lspsd_parms = psd.mvls_psd(data=oap_data, mask=all_dust, dx=dx, 
                                  k_side=k_side, print_update=True, 
                                  write_psd=True, psd_name=psd_name)
