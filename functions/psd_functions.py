'''
psd_functions.py
This modular function file has the following goals:
* [surfPSD] class that calculates PSD stuff for single surface
* [MODELING] Support functions for modeling a PSD
* [INTERPOLATION] The function in here needs to be checked.
* [SUPPORT] Dumping ground for functions running specific tasks in bigger functions
'''

import numpy as np
import copy
from scipy import interpolate

# for calling in data
from astropy.io import fits
from astropy import units as u

# drawing for the apertures
from skimage.draw import draw

##########################################
# surfPSD Class definition

class surfPSD:
    
    def __init__(self, surf_name, ca_percent, oversamp, **kwargs):
        self.surf_name = surf_name
        self.ca_percent = ca_percent
        self.oversamp = oversamp
        
    def open_surf(self, fileloc, surf_units):
        # to use if opening the data from a FITS file
        # assumes the data is efficiently filled (no zero col/row)
        self.data = (fits.open(fileloc)[0].data*surf_units).to(u.mm) # convert to mm internally
        # unload data from header
        hdr = fits.open(fileloc)[0].header
        self.wavelen = hdr['WAVELEN'] * u.m
        self.latres = hdr['LATRES'] * u.m / u.pix
        self.rpix_full = hdr['NAXIS1'] / 2 # full diameter is 1 side of the data
    
    def open_mask(self, fileloc):
        mask = fits.open(fileloc)[0].data
        if mask.shape != self.data.shape:
            raise Exception('Mask and data are not compatiable (shape)')
        else:
            self.mask = mask.astype(bool)
            self.npix_diam = int(np.sum(mask[int(mask.shape[0]/2)]))
            self.diam_ca = (self.npix_diam * u.pix * self.latres).to(u.mm)
    
    def load_surf(self, data, wavelen, latres):
        # to use if data has already been loaded into environment
        if hasattr(data, 'unit'): # set to data if there are units
            self.data = data
        else: # exit if there are no units
            raise Exception('Data needs units')
        self.wavelen = wavelen
        self.latres = latres
        self.rpix_full = np.shape(data)[0]/2
    
    def load_mask(self, mask):
        if mask.shape != self.data.shape:
            raise Exception('Mask and data are not compatiable (shape)')
        else:
            self.mask = mask.astype(bool)
            self.npix_diam = int(np.sum(mask[int(mask.shape[0]/2)]))
            self.diam_ca = (self.npix_diam * u.pix * self.latres).to(u.mm)
        
    def calc_psd(self, var_unit = u.nm, save_psd=False):
        # note - data MUST be even, square, and efficiently filled.
        optic = self.data.value

        # calculate the mean and variance of the active region of data
        ap_active = optic * self.mask
        ap_active_coords = optic[self.mask==True]
        ap_avg = np.mean(ap_active_coords)
        self.var = (np.var(ap_active_coords)*(self.data.unit**2)).to(var_unit**2)

        # Subtract the mean from the data
        ap_submean = (ap_active - ap_avg) * self.mask

        # build the Hann 2D window
        hannWin = han2d((self.npix_diam, self.npix_diam)) * self.mask
        
        # zero pad to oversample then start taking FT's
        pad_side = int((self.oversamp - self.mask.shape[0])/2)
        optic_ovs = np.pad(hannWin*ap_submean, pad_side, pad_with)
        FT_wf = np.fft.fftshift(np.fft.fft2(optic_ovs)) # this comes out unitless
        self.psd_raw = np.real(FT_wf*np.conjugate(FT_wf))/(self.data.unit**2)
        
        # The raw power is uncalibrated, need to normalize then multiply by variance.
        self.delta_k = 1/(self.oversamp*self.diam_ca/self.npix_diam)
        self.psd_norm = self.psd_raw / (np.sum(self.psd_raw)*(self.delta_k**2))
        self.psd_cal = self.psd_norm * self.var
        
        # save the PSD file
        if save_psd == True:
            print('Saving the calibrated PSD into FITS file is not a ready made feature, please check later.')

    def check_normpsd(self):
        var_verify = np.sum(self.psd_norm) * (self.delta_k**2) # unitless and 1
        psd_verify = np.allclose(1, var_verify)
        if psd_verify==True:
            print('PSD normalized: var={0:.3f}'.format(var_verify))
        else:
            print('PSD not normalized: var={0:.3f}. What happened?'.format(var_verify))
    
    def mask_psd(self, center, radius):
        mask = np.ones((self.oversamp, self.oversamp))
        
        # fill in the mask
        for nc in range(0, np.shape(center)[0]):
            mc = np.zeros_like(mask)
            mc_coords = draw.circle(center[nc][0], center[nc][1], radius=radius[nc])
            mc[mc_coords] = True
            mask[mc==True] = np.nan
        
        self.psd_raw *= mask
        self.psd_norm *= mask
        self.psd_cal *= mask
    
    def calc_psd_radial(self, ring_width):
        self.k_min = 1/self.diam_ca
        self.k_max = 1/(2*self.diam_ca / self.npix_diam)
        
        # make grid for average radial power value
        shift = np.int(self.oversamp/2)
        maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
        
        # Set full radial frequency range
        samp_space = self.diam_ca / self.npix_diam
        ft_freq = np.fft.fftfreq(n=self.oversamp, d=samp_space)
        self.radialFreq = ft_freq[0:shift]
        
        # set up ring parameters
        if ring_width % 2 == 0:
            ring_width += 1 # increase by 1 to make it odd
        self.ring_width = ring_width
        r_halfside = np.int((ring_width-1)/2)
        r = 1
        
        # initialize content
        mean_psd = [] # initialize empty list of mean PSD values
        mean_psd2 = []
        k_val = [] # initialize empty list of spatial frequencies
        
        # chug along through the radial frequency values
        while((r+r_halfside)<shift): # while inside the region of interest
            ri = r - r_halfside # inner radius of ring
            if self.radialFreq[r].value < self.k_min.value: # verify that position r is at the low limit
                #print('test k-value too small, iterate to next')
                r+=1
            else:
                if ri > 0:
                    radial_mask = makeRingMask(maskY, maskX, ri, ring_width)
                    radial_bin = makeRingMaskBin(self.psd_norm.value,radial_mask)
                    mean_psd.append(np.mean(radial_bin))
                    radial_bin2 = makeRingMaskBin(self.psd_cal.value, radial_mask)
                    mean_psd2.append(np.mean(radial_bin2))
                    k_val.append(self.radialFreq[r].value)
                r+=ring_width # iterate to the next r value in the loop

        self.k_radial = k_val * self.radialFreq.unit
        self.psd_radial_norm = mean_psd * self.psd_norm.unit
        self.psd_radial_cal = mean_psd2 * self.psd_cal.unit
    
    def calc_psd_rms(self, pwr_opt, tgt_low, tgt_high, print_rms=False):
        # find the locations for k_low and k_high:
        (bin_low, k_low) = k_locate(self.radialFreq, tgt_low)
        (bin_high, k_high) = k_locate(self.radialFreq, tgt_high)
        ring_width = bin_high - bin_low
        
        # make a grid for the average radial power value
        shift = np.int(self.oversamp/2)
        maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
        
        # make the mask
        radial_mask = makeRingMask(maskY, maskX, bin_low, ring_width)
        radial_bin = makeRingMaskBin(pwr_opt.value,radial_mask) * pwr_opt.unit
        
        # calculate the rms
        rms_val = np.sqrt(np.sum(radial_bin * (self.delta_k**2)))
        if print_rms==True:
            print('Target range - k_low: {0:.3f} and k_high: {1:.3f}'.format(k_low, k_high))
            print('RMS value: {0:.4f}'.format(rms_val))
        return rms_val

##########################################
# MODELING
'''
Separate from the main class because this is applied only to the average PSD.
Class is applied to individual surface PSD.

Assumptions on the units:
alpha           - unitless
beta            - nm^2 mm^(-alpha+2). It washes out at the end.
L0              - mm (1/spatial frequency = (1/(1/mm) = mm)
lo              - unitless
k_min, k_max    - 1/mm (spatial frequency)
rms_sr          - nm (surface roughness)
rms_surf        - nm (total surface)
radial_k        - 1/mm (spatial frequency)
radial_psd      - mm^2 nm^2 (1/spatial frequency ^2 * variance = mm^2 nm^2)
bsr        - mm^2 nm^2 (same units as psd)
'''
def calc_beta(alpha, k_min, k_max, rms_surf):
    # unit check and fix
    if k_min.unit != (1/u.mm):  k_min.to(1/u.mm)
    if k_max.unit != (1/u.mm):  k_max.to(1/u.mm)
    if rms_surf.unit != u.nm:   rms_surf.to(u.nm)
    
    # calculate beta
    if alpha==2:
        beta = (rms_surf**2) / (2*np.pi*np.log(k_max/k_min))
    else: # when alpha not 2
        beta = (rms_surf**2) * (alpha - 2) / (2*np.pi*( (k_min**(2-alpha)) - (k_max**(2-alpha)) ) )
    return beta # units safe

def calc_bsr(rms_sr, k_min, k_max):
    # unit check and fix
    if k_min.unit != (1/u.mm):  k_min.to(1/u.mm)
    if k_max.unit != (1/u.mm):  k_max.to(1/u.mm)
    if rms_sr.unit != u.nm:     rms_sr.to(u.nm)

    return (rms_sr**2) / (np.pi * (k_max**2 - k_min**2)) # units safe

def solve_model_lsf(radial_psd, radial_k):
    # unit check and fix
    if radial_psd.unit != (u.mm**2 * u.nm**2):  radial_psd.to(u.mm**2 * u.nm**2)
    if radial_k.unit != (1/u.mm):               radial_k.to(1/u.mm)
    
    # linearized equation from PSD in form y = mx + c
    y = np.log10(radial_psd.value)
    x = np.log10(radial_k.value)
    
    # linear least square fit
    A = np.vstack([x, np.ones(len(x))]).T
    m,c = np.linalg.lstsq(A, y)[0] # y = mx + c linear equation
    alpha_lsf = -1*m # unitless naturally
    beta_lsf = 10**(c) * (u.nm**2) * (u.mm**(-alpha_lsf+2))
    
    return (alpha_lsf, beta_lsf) #units applied

def calc_model_simple(radial_k, alpha, beta):
    # for OAPs and flat mirrors
    # unit check and fix
    if radial_k.unit != (1/u.mm):   radial_k.to(1/u.mm)
    if beta.unit != (u.nm**2 * u.mm**(-alpha+2)):   beta.to(u.nm**2 * u.mm**(-alpha+2))
        
    return beta/(radial_k**alpha) # units safe

def calc_model_full(radial_k, alpha, beta, L0, lo, bsr = 0 * u.nm**2 * u.mm**2):
    # only useful for OAPs
    # unit check and fix
    if radial_k.unit != (1/u.mm):   radial_k.to(1/u.mm)
    if beta.unit != (u.nm**2 * u.mm**(-alpha+2)):   beta.to(u.nm**2 * u.mm**(-alpha+2))
    if L0.unit != u.mm: L0.to(u.mm)
    if bsr.unit != (u.nm**2 * u.mm**2): bsr.to(u.nm**2 * u.mm**2)
        
    # exponential cannot handle units, so calculate without units then apply at the end.
    pk = (beta.value * np.exp(-(radial_k.value*lo)**2) / ( ( (L0.value**-2) + (radial_k.value**2) ) ** (alpha*.5))) + bsr.value
    return pk * bsr.unit

###########################################
# INTERPOLATION

def k_interp(oap_label, kval, npix_diam, norm_1Dpsd, cal_1Dpsd, k_npts):
    kmin = []
    kmax = []
    ntot = np.shape(npix_diam)[0]
    
    # calculate the min and max
    for no in range(0,ntot):
        oap_name = oap_label[no]
        kmax.append(np.amax(kval[oap_name]).value)
        kmin.append(np.amin(kval[oap_name]).value)
        
    kmin_interp = np.amax(kmin)*kval[oap_name].unit
    kmax_interp = np.amin(kmax)*kval[oap_name].unit
    print('kmin = {0:.4f}, kmax = {1:.4f}'.format(kmin_interp, kmax_interp))
    
    # check if interpolation needs to happen:
    if (np.unique(npix_diam).size==1) and (np.unique(kmax).size==1) and (np.unique(kmin).size==1):
        psd1d_interp = False
        print('1D-PSD interpolation does not need to occur; #pts of spatial freq: {0}'.format(k_npts))
        
    else:
        psd1d_interp = True
        k_new = np.linspace(kmin_interp, kmax_interp, num=k_npts) # new range of spatial frequencies
        print('1D-PSD interpolation does needs to occur; #pts of spatial freq: {0}'.format(np.shape(k_new)[0]))

    # Write the matrices for the data
    psd1D_data = np.zeros((ntot, k_npts)) # all k_val is same size
    #print('size of psd1D_data: {0}'.format(np.shape(psd1D_data)))
    psd1D_norm = np.zeros((ntot, k_npts)) # all k_val is same size
    #print('size of psd1D_norm: {0}'.format(np.shape(psd1D_norm)))
    k_out = np.zeros_like((psd1D_data))
    #print('size of k_out: {0}'.format(np.shape(k_out)))
    
    # fill in the data matricies from the PSD simulation and interpolate
    for no in range(0,ntot):
        oap_name = oap_label[no]
        k_orig = kval[oap_name]
        psd_data = cal_1Dpsd[oap_name]
        psd_norm = norm_1Dpsd[oap_name]

        if psd1d_interp == True:
            # calculate the interpolation function based on the data
            f_data = interpolate.interp1d(k_orig, psd_data)
            f_norm = interpolate.interp1d(k_orig, psd_norm)
            # fill in the interpolation for specific spatial frequencies
            psd1D_data[no,:] = f_data(k_new)
            psd1D_norm[no,:] = f_norm(k_new)
            k_out[no, :] = k_new
            
        else:
            psd1D_data[no, :] = psd_data
            psd1D_norm[no, :] = psd_norm
            k_out[no, :] = k_orig

    # apply units
    psd1D_data *= cal_1Dpsd[oap_name].unit
    psd1D_norm *= norm_1Dpsd[oap_name].unit
    
    return(psd1D_data, psd1D_norm, k_out)


###########################################
# SUPPORT

def han2d(shape, fraction=1./np.sqrt(2), normalize=False):
    '''
    Code generated by Kyle Van Gorkom.
    Radial Hanning window scaled to a fraction 
    of the array size.
    
    Fraction = 1. for circumscribed circle
    Fraction = 1/sqrt(2) for inscribed circle (default)
    '''
    #return np.sqrt(np.outer(np.hanning(shape[0]), np.hanning(shape[0])))

    # get radial distance from center
    radial = get_radial_dist(shape)

    # scale radial distances
    rmax = radial.max() * fraction
    scaled = (1 - radial / rmax) * np.pi/2.
    window = np.sin(scaled)**2
    window[radial > fraction * radial.max()] = 0.
    return window

def get_radial_dist(shape, scaleyx=(1.0, 1.0)):
    '''
    Code generated by Kyle Van Gorkom.
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def zeroPadOversample(optic_data,oversamp):
    '''
    Makes zero pad based on oversampling requirements
    Input:
    optic_data  - 2D array of data
    oversamp    - oversampling multiplier
    Output:
    zp_wf       - zero padded wavefront
    '''
    n_row = np.shape(optic_data)[0]
    n_col = np.shape(optic_data)[1]
    
    if n_row != n_col: # check if a symmetric matrix is being passed in
        # zero pad data into symmetric matrix
        data = zeroPadSquare(optic_data)
        # recalibrate the number of rows and columns
        n_row = np.shape(data)[0]
        n_col = np.shape(data)[1]
    else:
        data = np.copy(optic_data)
    # Sample the matrix as some 2-factor value
    samp = getSampSide(data)
    
    # This is the oversampled side size
    side = samp * oversamp
    # NOTE: This will not work for an odd symmetric matrix! If you get an error, this is why.
    row_pad = np.int((side - n_row)/2)
    zp_wf = np.pad(data, (row_pad,row_pad), 'constant')
                
    return zp_wf

def getSampSide(optic_data):
    '''
    Calculates the sample side based on the largest side of image
    Input:
    optic_data  - 2D array of the data
    Output:
    samp        - sample side value
    '''
    # Choose the larger side first
    if np.shape(optic_data)[0] > np.shape(optic_data)[1]:
        samp_side = np.shape(optic_data)[0]
    else:
        samp_side = np.shape(optic_data)[1]
    
    # Choosing a sample size
    if samp_side < 512:
        if samp_side < 256:
            samp = 256
        else:
            samp = 512
    else:
        samp = 1024
    
    return samp

def makeRingMask(y,x,inner_r,r_width):
    '''
    Makes radial median mask... that looks like a ring.
    Input:
    y        - meshgrid vertical values (pixel count units)
    x        - meshgrid horizontal values (pixel count units)
    inner_r  - inner radial value (pixel count units)
    dr       - ring thickness (pixel count units)
    Output:
    ringMask - ring mask (boolean type)
    '''
    inside_mask = x**2+y**2 <= inner_r**2
    outside_mask = x**2+y**2 <= (inner_r+r_width)**2
    ringMask = outside_mask ^ inside_mask # use xor, such that so long as one is true then it will make ring.
    return ringMask
    
def makeRingMaskBin(power_data, ringMask):
    '''
    Returns bin values of the ring mask
    Input:
    data        - wavefront power data, must be square matrix
    ringmask    - ring mask, must be same size as data and boolean type
    Output:
    ringmaskbin - vector of values passed through mask
    '''
    ringMaskBin = np.extract(ringMask, power_data)
    ringbin = ringMaskBin[~np.isnan(ringMaskBin)]
    return ringbin

def k_locate(freqrange, k_tgt, print_change=False):
    # given a target spatial frequency, find the index bin and value closest to target.
    kscale = np.abs(freqrange.value - k_tgt.value)
    bb = np.where(kscale==np.amin(kscale))
    if freqrange[bb][0] != k_tgt and print_change == True:
        print('Target: {0:.2f}; changing to closest at {1:.3f}'.format(k_tgt, freqrange[bb][0]))
    return (bb[0][0], freqrange[bb][0])

# verbatim taken from numpy.pad website example
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder',0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
