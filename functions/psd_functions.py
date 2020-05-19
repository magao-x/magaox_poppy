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
    
    def __init__(self, surf_name, **kwargs):
        self.surf_name = surf_name
        
    def open_surf(self, fileloc, surf_units):
        # to use if opening the data from a FITS file
        # assumes the data is efficiently filled (no zero col/row)
        self.data = (fits.open(fileloc)[0].data*surf_units).to(u.mm) # convert to mm internally to match with the fourier transform units
        # unload data from header
        hdr = fits.open(fileloc)[0].header
        self.wavelen = hdr['WAVELEN'] * u.m
        self.latres = hdr['LATRES'] * u.m / u.pix
    
    def load_surf(self, data, wavelen, latres):
        # to use if data has already been loaded into environment
        if hasattr(data, 'unit'): # set to data if there are units
            self.data = data
        else: # exit if there are no units
            raise Exception('Data needs units')
        if hasattr(wavelen, 'unit'):
            self.wavelen = wavelen
        else:
            raise Exception('Wavelength needs units')
        if hasattr(latres, 'unit'):
            self.latres = latres
        else:
            raise Exception('Lateral Resolution needs units')
        
    def open_mask(self, fileloc):
        mask = fits.open(fileloc)[0].data
        self.load_mask(mask)
    
    def load_mask(self, mask):
        if mask.shape != self.data.shape:
            raise Exception('Mask and data are not compatiable (shape)')
        else:
            self.mask = mask.astype(bool)
            self.npix_diam = int(np.sum(mask[int(mask.shape[0]/2)]))
            self.diam_ca = (self.npix_diam * u.pix * self.latres).to(u.mm)
            
    def open_psd(self, psd_fileloc, psd_type, var_unit = u.nm):
        psd_fits = fits.open(psd_fileloc)[0]
        hdr = psd_fits.header
        self.oversamp = hdr['oversamp']
        self.diam_ca = hdr['diam_ca']*u.mm
        self.npix_diam = int(hdr['diam_pix'])
        self.var = hdr['var_tot']*(var_unit**2)
        self.calc_psd_parameters()
        
        # set the PSD variables
        if psd_type == 'norm':
            self.psd_norm = psd_fits.data * (self.diam_ca.unit**2)
            self.psd_cal = self.psd_norm * self.var
        elif psd_type == 'cal':
            self.psd_cal = psd_fits.data * (self.diam_ca.unit**2) * (var_unit**2)
            
    def load_psd(self, psd_data, psd_type, var):
        if hasattr(var, 'unit'):
            self.var = var
        else:
            raise Exception('Variance needs units')
        if hasattr(psd_data, 'unit'):
            if psd_type=='norm':
                self.psd_norm=psd_data
                self.psd_cal = self.psd_norm * var 
            elif psd_type=='cal':
                self.psd_cal=psd_data
                self.psd_norm = self.psd_cal / var
            elif psd_type=='raw':
                self.psd_raw=psd_raw
            else:
                raise Exception('Type does not match cal, norm, raw; type sent: {0}'.format(psd_type))
        else:
            raise Exception('PSD needs units')
        
    def load_psd_parameters(self, oversamp, diam_ca, npix_diam, wavelen):
        self.oversamp = oversamp
        self.diam_ca = diam_ca
        self.npix_diam = npix_diam
        self.wavelen = wavelen
        self.calc_psd_parameters() # calculate other necessary parameters
    
    def calc_psd_parameters(self):
        self.k_min = 1/self.diam_ca
        self.k_max = 1/(2*self.diam_ca / self.npix_diam)
        self.delta_k = 1/(self.oversamp*self.diam_ca/self.npix_diam)
        
        # Set full radial frequency range
        samp_space = self.diam_ca / self.npix_diam
        ft_freq = np.fft.fftfreq(n=self.oversamp, d=samp_space)
        self.radialFreq = ft_freq[0:np.int(self.oversamp/2)]
        
    def calc_psd(self, oversamp, kmid_ll = 0.1/u.mm, khigh_ll=1/u.mm, var_unit = u.nm):
        self.oversamp = oversamp
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
        self.calc_psd_parameters() # calculate some basic parameters needed for PSDs and everything
        self.psd_norm = self.psd_raw / (np.sum(self.psd_raw)*(self.delta_k**2))
        self.psd_cal = self.psd_norm * self.var
        
        # Calculate the RMS based on the k-parameter limits
        self.calc_rms_set(kmid_ll=kmid_ll, khigh_ll=khigh_ll, pwr_opt=self.psd_cal)
        
    def check_normpsd(self, psd_norm):
        var_verify = np.sum(psd_norm) * (self.delta_k**2) # unitless and 1
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
        # shortcut version for basic code analysis
        (self.k_radial, self.psd_radial_cal) = self.do_psd_radial(ring_width=ring_width,
                                                                  psd_data = self.psd_cal)

    def do_psd_radial(self, ring_width, psd_data):
        # generic version of code if inputting a different ring width and PSD data set
        # make grid for average radial power value
        shift = np.int(self.oversamp/2)
        maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
        
        # set up ring parameters
        if ring_width % 2 == 0:
            ring_width += 1 # increase by 1 to make it odd
        r_halfside = np.int((ring_width-1)/2)
        r = 1
        
        # initialize content
        mean_bin = [] # initialize empty list of mean PSD values
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
                    radial_bin = makeRingMaskBin(psd_data.value,radial_mask)
                    mean_bin.append(np.mean(radial_bin))
                    k_val.append(self.radialFreq[r].value)
                r+=ring_width # iterate to the next r value in the loop
                
        k_radial = k_val * self.radialFreq.unit
        psd_radial = mean_bin * psd_data.unit
        
        return (k_radial, psd_radial)
    
    def calc_rms_set(self, kmid_ll, khigh_ll, pwr_opt):
        # Calculate the RMS based on the k-parameter limits
        self.kmid_ll = kmid_ll
        self.khigh_ll = khigh_ll
        self.rms_tot = self.calc_psd_rms(tgt_low=self.k_min, tgt_high=self.k_max,
                                     pwr_opt=pwr_opt)
        self.rms_l = self.calc_psd_rms(tgt_low=self.k_min, tgt_high=kmid_ll,
                                     pwr_opt=pwr_opt)
        self.rms_m = self.calc_psd_rms(tgt_low=kmid_ll, tgt_high=khigh_ll,
                                     pwr_opt=pwr_opt)
        self.rms_h = self.calc_psd_rms(tgt_low=khigh_ll, tgt_high=self.k_max,
                                        pwr_opt=pwr_opt)
        self.rms_mh = self.calc_psd_rms(tgt_low=kmid_ll, tgt_high=self.k_max,
                                        pwr_opt=pwr_opt)
    
    def calc_psd_rms(self, tgt_low, tgt_high, pwr_opt, print_rms=False, print_kloc = False):       
        # find the locations for k_low and k_high:
        (bin_low, k_low) = k_locate(self.radialFreq, tgt_low, print_change=print_kloc)
        (bin_high, k_high) = k_locate(self.radialFreq, tgt_high, print_change=print_kloc)
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
    
    def write_psd_file(self, filename, psd_data, single_precision=True):
        # Write header and cards for FITS
        hdr = fits.Header()
        hdr['name'] = (self.surf_name + ' PSD', 'filename')
        hdr['psd_unit'] = (str(psd_data.unit), 'Units for PSD data')
        hdr['wavelen'] = (self.wavelen.value, 'Wavelength used for optical test [{0}]'.format(self.wavelen.unit))
        hdr['diam_ca'] = (self.diam_ca.value, 'Physical diameter for clear aperture [{0}]'.format(self.diam_ca.unit))
        hdr['diam_pix'] = (self.npix_diam, 'Pixel diameter for clear aperture')
        hdr['oversamp'] = (self.oversamp, 'Oversampled array size')
        hdr['delta_k'] = (self.delta_k.value, 'Spatial frequency lateral resolution [{0}]'.format(self.delta_k.unit))
        hdr['k_min'] = (self.k_min.value, 'Minimum spatial frequency boundary [{0}]'.format(self.k_min.unit))
        hdr['k_max'] = (self.k_max.value, 'Maximum spatial frequency boundary [{0}]'.format(self.k_max.unit))
        hdr['rms_tot'] = (self.rms_tot.value, 'Total RMS based on kmin and kmax [{0}]'.format(self.rms_tot.unit))
        hdr['var_tot'] = (self.var.value, 'Total variance for optical surface [{0}]'.format(self.var.unit))
        
        if single_precision==True:
            write_data = np.single(psd_data.value)
        else:
            write_data = psd_data.value
        
        # Write to FITS file
        fits.writeto(filename, write_data, hdr, overwrite=True)
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
        print('Target: {0:.4f}; changing to closest at {1:.4f}'.format(k_tgt, freqrange[bb][0]))
    return (bb[0][0], freqrange[bb][0])

# verbatim taken from numpy.pad website example
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder',0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
