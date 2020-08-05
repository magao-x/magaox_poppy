import numpy as np
from astropy import units as u
from astropy.io import fits
import copy
import pickle
import random

import poppy
from model_kit import magaoxFunctions as mf
from skimage.draw import draw
from datetime import date


# declare MagAO-X variables
fr_parm = {'wavelength': 0.9e-6 * u.m,
           'npix': 512, # sample size
           'beam_ratio': 0.25, # oversample
           'leak_mult': 0.01,
           'surf_off': True,
           'n_tests': 10}

wavelen = np.round(fr_parm['wavelength'].to(u.nm).value).astype(int)
br = int(1/fr_parm['beam_ratio'])
parm_name = '{0:3}_{1:1}x_{2:3}nm'.format(fr_parm['npix'], br, wavelen)

# load the CSV prescription values
home_dir = '/home/jhen/XWCL/code/MagAOX/' # change for exao0
data_dir = home_dir + 'data/'
rx_loc = data_dir+'rxCSV/rx_magaox_NCPDM_sci_{0}.csv'.format(parm_name)
rx_sys = mf.makeRxCSV(rx_loc)

# acquiring csv numerical values for specifically named optics
for t_optic, test_opt in enumerate(rx_sys):
    if test_opt['Optical_Element_Number'] > 3 and test_optic['type'] == 'mirror':
        if test_opt['focal_length_m'] == 0:
            test_opt['surf_PSD_filename'] = 'psd_fm_pdr'
        else:
            test_opt['surf_PSD_filename'] = 'psd_oap_pdr'
    if test_opt['Name'] == 'Tweeter':
        tweeter_num = test_opt['Optical_Element_Number']
    elif test_opt['Name'] == 'vAPP-trans':
        vappTRANS_num = test_opt['Optical_Element_Number']
    elif test_opt['Name'] == 'vAPP-opd':
        vappOPD_num = test_opt['Optical_Element_Number']

# vAPP files (rewrite this section with better vAPP files)
vAPP_pixelscl = vapp_diam.value/fr_parm['npix'] # more direct
vAPP_folder = data_dir+'coronagraph/'
vAPP_trans_filename = 'vAPP_trans_2PSF_{0}'.format(parm_name)
vAPP_posOPD_filename = 'vAPP_opd_2PSF_{0}_posPhase'.format(parm_name)
vAPP_negOPD_filename = 'vAPP_opd_2PSF_{0}_negPhase'.format(parm_name)

# load the PSD pickles to get the PSD parameters
psd_label = ['pdr', 'm2m3']
psd_dict = {}
for n in range(0, len(psd_label)):
    psd_filename = home_dir + 'PSD/psd_parms_{0}.pickle'.format(psd_label[n])
    with open(psd_filename, 'rb') as psd_data_file:
        psd_data = pickle.load(psd_data_file)
    psd_dict.update(psd_data)
    
# build the random seed arrays
n_rand = rx_sys.shape[0] * fr_parm['n_tests']
seed_set = random.sample(range(n_rand*5), n_rand)
seed_set = np.reshape(seed_set, (fr_parm['n_tests'], rx_sys.shape[0]))
# need to save this seed array as a numpy array file!!!

# generate the PSFs
print('[FRESNEL] Calculating all PSFs at wavelength = {0}'.format(fr_parm['wavelength']))

for n in range(0, fr_parm['n_tests']):
    # Build the leakage term
    rx_sys['surf_PSD_filename'][vappTRANS_num] = 'none'
    rx_sys['surf_PSD_filename'][vappOPD_num] = 'none'
    magaox = mf.csvFresnel(rx_csv=rx_sys, samp=fr_parm['npix'], oversamp=fr_parm['beam_ratio'],
                           break_plane='F69Sci', psd_dict=psd_dict, seed=seed_set[n])
    leak_psf = magaox.calc_psf(wavelength=fr_parm['wavelength'].value)[0] 
    #print('Leakage PSF (0 phase) complete')

    # bottom PSF (positive phase coronagraph)
    rx_sys['surf_PSD_filename'][vappTRANS_num] = vAPP_folder+vAPP_trans_filename
    rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_folder+vAPP_posOPD_filename
    magaox = mf.csvFresnel(rx_csv=rx_sys, samp=fr_parm['npix'], oversamp=fr_parm['beam_ratio'],
                           break_plane='F69Sci', psd_dict=psd_dict, seed=seed_set[n])
    pos_psf = magaox.calc_psf(wavelength=fr_parm['wavelength'].value)[0]
    #print('Bottom PSF (+phase) complete')

    # top PSF (negative phase coronagraph)
    rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_folder+vAPP_negOPD_filename
    magaox = mf.csvFresnel(rx_csv=rx_sys, samp=fr_parm['npix'], oversamp=fr_parm['beam_ratio'],
                           break_plane='F69Sci', psd_dict=psd_dict, seed=seed_set[n])
    neg_psf = magaox.calc_psf(wavelength=fr_parm['wavelength'].value)[0]
    #print('Top PSF (-phase) complete')

    # sum the PSFs
    tot_psf = pos_psf.data + neg_psf.data + (leak_psf.data * fr_parm['leak_mult'])
    contr_psf = tot_psf/tot_psf.max()

    # Write the PSF to file to analyze with ds9
    output_dir = home_dir + 'fresnel_prop/output/'
    psf_filename = 'psf_mwfs_pdrsurf_contrast_{0:3}nm_n{1}.fits'.format(wavelen, n)
    hdr = copy.copy(pos_psf.header)
    del hdr['HISTORY']
    hdr.set('PIX_SAMP', fr_parm['npix'], 'initial pixel sampling size')
    hdr.set('LEAKMULT', fr_parm['leak_mult'], 'Multiplier for leakage term intensity')
    hdr.set('DIFFMODE', 'Fresnel', 'Diffraction Proagation mode')
    fits.PrimaryHDU(contr_psf, header=hdr).writeto(psf_loc, overwrite=True)

print('[FRESNEL] All PSFs complete')
