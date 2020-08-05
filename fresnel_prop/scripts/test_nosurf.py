import numpy as np
from astropy import units as u
from astropy.io import fits
import copy

import poppy
from model_kit import magaoxFunctions as mf
from skimage.draw import draw
from datetime import date


# declare MagAO-X variables
fr_parm = {'wavelength': 0.9e-6 * u.m,
           'npix': 512, # sample size
           'beam_ratio': 0.25, # oversample
           'leak_mult': 0.01,
           'surf_off': True}

wavelen = np.round(fr_parm['wavelength'].to(u.nm).value).astype(int)
br = int(1/fr_parm['beam_ratio'])
parm_name = '{0:3}_{1:1}x_{2:3}nm'.format(fr_parm['npix'], br, wavelen)

# load the CSV prescription values
home_dir = '/home/jhen/XWCL/code/MagAOX/' # change for exao0
data_dir = home_dir + 'data/'
rx_loc = data_dir+'rxCSV/rx_magaox_NCPDM_sci_{0}.csv'.format(parm_name)
rx_sys = mf.makeRxCSV(rx_loc)

# acquiring csv numerical values for specifically named optics
# same for all csv prescription files based on v11.1
for t_optic, test_opt in enumerate(rx_sys):
    if fr_parm['surf_off'] == True and test_opt['Optical_Element_Number'] > 1:
        test_opt['surf_PSD_filename'] = 'none'
    if test_opt['Name'] == 'Tweeter':
        tweeter_num = test_opt['Optical_Element_Number']
    elif test_opt['Name'] == 'vAPP-trans':
        vappTRANS_num = test_opt['Optical_Element_Number']
        vapp_diam = test_opt['Radius_m']*2*u.m
    elif test_opt['Name'] == 'vAPP-opd':
        vappOPD_num = test_opt['Optical_Element_Number'] 

# vAPP files (rewrite this section with better vAPP files)
vAPP_pixelscl = vapp_diam.value/fr_parm['npix'] # more direct
vAPP_folder = data_dir+'coronagraph/'
vAPP_trans_filename = 'vAPP_trans_2PSF_{0}'.format(parm_name)
vAPP_posOPD_filename = 'vAPP_opd_2PSF_{0}_posPhase'.format(parm_name)
vAPP_negOPD_filename = 'vAPP_opd_2PSF_{0}_negPhase'.format(parm_name)

# generate the PSFs
print('[FRESNEL] Calculating all PSFs at wavelength = {0}'.format(fr_parm['wavelength']))

# Build the leakage term
rx_sys['surf_PSD_filename'][vappTRANS_num] = 'none'
rx_sys['surf_PSD_filename'][vappOPD_num] = 'none'
magaox = mf.csvFresnel(rx_sys, fr_parm['npix'], fr_parm['beam_ratio'], 'F69Sci')
leak_psf = magaox.calc_psf(wavelength=fr_parm['wavelength'].value)[0] # much faster for crunching to intensity!
print('Leakage PSF (0 phase) complete')

# bottom PSF (positive phase coronagraph)
rx_sys['surf_PSD_filename'][vappTRANS_num] = vAPP_trans_filename
rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_posOPD_filename
magaox = mf.csvFresnel(rx_sys, fr_parm['npix'], fr_parm['beam_ratio'], 'F69Sci')
pos_psf = magaox.calc_psf(wavelength=fr_parm['wavelength'].value)[0]
print('Bottom PSF (+phase) complete')

# top PSF (negative phase coronagraph)
rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_negOPD_filename
magaox = mf.csvFresnel(rx_sys, fr_parm['npix'], fr_parm['beam_ratio'], 'F69Sci')
neg_psf = magaox.calc_psf(wavelength=fr_parm['wavelength'].value)[0]
print('Top PSF (-phase) complete')

# sum the PSFs
tot_psf = pos_psf.data + neg_psf.data + (leak_psf.data * fr_parm['leak_mult'])
contr_psf = tot_psf/tot_psf.max()

# Write the PSF to file to analyze with ds9
output_dir = home_dir + 'fresnel_prop/output/'
psf_filename = 'psf_mwfs_nosurf_contrast_{0:3}nm.fits'.format(wavelen)
hdr = copy.copy(pos_psf.header)
del hdr['HISTORY']
hdr.set('PIX_SAMP', fr_parm['npix'], 'initial pixel sampling size')
hdr.set('LEAKMULT', fr_parm['leak_mult'], 'Multiplier for leakage term intensity')
hdr.set('DIFFMODE', 'Fresnel', 'Diffraction Proagation mode')
fits.PrimaryHDU(contr_psf, header=hdr).writeto(psf_loc, overwrite=True)

