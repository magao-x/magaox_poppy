import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from scipy import interpolate

# for calling in data
from astropy.io import fits
from astropy import units as u
import h5py

# for windowing
from scipy import signal

# MagAO-X function definitions, you'll need at least one
import magaoxFunctions as mf

# drawing for the apertures
from skimage.draw import draw


class surfPSD:
    
    def __init__(self, surf_name, fileloc, ca_percent, diam_100ca, oversamp, **kwargs):
        self.surf_name = surf_name
        self.fileloc = fileloc
        self.oversamp = oversamp
        
        self.ca_percent = ca_percent
        self.diam_100ca = diam_100ca
        
        # unpack the fits file content
        #self.data = fits.open(self.fileloc)[0].data * u.micron
        self.data = fits.open(self.fileloc)[0].data * 1e-3 * u.mm 
        # original data is in microns, convert to mm
        
        # unload data from header
        hdr = fits.open(self.fileloc)[0].header
        self.wavelen = hdr['WAVELEN'] * u.m
        self.latres = hdr['LATRES'] * u.m / u.pix
        self.rpix_full = hdr['NAXIS1'] / 2 # full diameter is 1 side of the data
        #self.ca_data = hdr['CA_DATA']
        #self.diam_full = hdr['']
    
    def calc_psd(self, save_psd=False):
        # calculate the power optic
        # note - data MUST be even and square.
        optic = self.data.value

        # Calculate the number of pixel diameter from CA
        #data_side = np.shape(optic_data)[0]
        npix_diam = np.int(self.data.shape[0]*self.ca_percent/100) # integer required for hannWin
        # npix_diam must be even to get the padding to work correctly
        if npix_diam %2 != 0: # if odd
            npix_diam -= 1 # decrease by 1 to force to even
        self.npix_diam = npix_diam
        self.diam_ca = (npix_diam * u.pix) * self.latres * (1e3 * u.mm / u.m) 
        
        # Create the clear aperture mask based on the clear aperture percent
        ap_clear = np.zeros_like(optic, dtype=np.uint8)
        ca_coords = draw.circle(self.rpix_full, self.rpix_full, radius=(npix_diam/2))
        ap_clear[ca_coords] = True

        # calculate the mean and variance of the active region of data
        ap_active = optic * ap_clear
        ap_active_coords = optic[ap_clear==True]
        ap_avg = np.mean(ap_active_coords)
        self.var = np.var(ap_active_coords)*(self.data.unit**2)

        # Subtract the mean from the data
        ap_submean = (ap_active - ap_avg) * ap_clear

        # build the Hann 2D window
        hannWin = han2d((npix_diam, npix_diam))
        # zero-pad the window to fit with the full aperture dimensions
        pad_side = np.int((self.data.shape[0] - npix_diam)/2)
        pad_hann = np.pad(hannWin, pad_side, mf.pad_with) * ap_clear

        # oversample the windowed data and take FFT for the raw power
        optic_ovs = zeroPadOversample(ap_submean * pad_hann, 
                                          np.int(self.oversamp/getSampSide(ap_clear)))*self.data.unit
        FT_wf = np.fft.fftshift(np.fft.fft2(optic_ovs)) 
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
        mask = np.ones_like((self.oversamp, self.oversamp))
        
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
        self.k_lowlim = 1/self.diam_ca
        
        # make grid for average radial power value
        #shift = np.int(ft_freq.size/2)
        #self.shift = shift
        #maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
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
            if self.radialFreq[r].value < self.k_lowlim.value: # verify that position r is at the low limit
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
    
    def calc_psd_rms(self, pwr_opt, tgt_low, tgt_high):
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
        print('Target range - k_low: {0:.3f} and k_high: {1:.3f}'.format(k_low, k_high))
        print('RMS value: {0.4f}'.format(rms_val)
        return rms_val

##########################################
# Model Calculation Functions

def calc_beta(alpha, k_min, k_max, rms_val):
    var_val = rms_val**2
    if alpha==2:
        beta = var_val / (2*np.pi*np.log(k_max/k_min))
    else: # when beta not 2
        beta = var_val * (alpha - 2) / (2*np.pi*( (k_min**(2-alpha)) - (k_max**(2-alpha)) ) )
    return beta

def solve_model_lsf(radial_psd, radial_k):
    y = np.log10(radial_psd.value)
    x = np.log10(radial_k)
    
    # linear least square fit
    A = np.vstack([x, np.ones(len(x))]).T
    m,c = np.linalg.lstsq(A, y)[0]
    alpha_lsf = -1*m
    beta_lsf = 10**(c)
    
    return (alpha_lsf, beta_lsf)

def calc_model_simple(radial_k, alpha, beta):
    return beta/(radial_k**alpha)

def calc_model_full(radial_k, alpha, beta, L0, lo):
    return beta * np.exp(-(radial_k*lo)**2) / ( ( (L0**-2) + (radial_k**2) ) ** (alpha*.5))

###########################################
# Other calculation functions that are important

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
    
    #k_npts = np.shape(kval[oap_name])[0]
    #print('spatial freq pts: {0}'.format(k_npts))
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
# Supporting functions

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

def k_locate(freqrange, k_tgt):
    # given a target spatial frequency, find the index bin and value closest to target.
    kscale = np.abs(freqrange.value - k_tgt.value)
    bb = np.where(kscale==np.amin(kscale))
    if freqrange[bb][0] != k_tgt:
        print('Target: {0}; changing to closest at {1}'.format(k_tgt, freqrange[bb][0]))
    return (bb[0][0], freqrange[bb][0])

############################################
# DATX AND FITS FILE HANDLING

def doCenterCrop(optic_data,shift):
    side = np.shape(optic_data)[0]
    center = np.int(side/2)
    crop_data = optic_data[center-shift:center+shift,center-shift:center+shift]
    return crop_data

def show2plots(supertitle, data1, plot1_label, data2, plot2_label, set_dpi=150):
    # Shortcut function for 2 plot drawing
    fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=set_dpi)
    fig.suptitle(supertitle)
    
    if hasattr(data1, 'unit'):
        show_data = data1.value
        cb_label = str(data1.unit)
    else:
        show_data = data1
        cb_label = ''
    img1 = ax1.imshow(show_data, origin='bottom')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("bottom", size="5%", pad=0.25)
    ax1.set_title(plot1_label)
    fig.colorbar(img1, cax=cax1, orientation='horizontal').set_label(cb_label)

    if hasattr(data2, 'unit'):
        show_data = data2.value
        cb_label = str(data2.unit)
    else:
        show_data = data2
        cb_label = ''
    img2 = ax2.imshow(show_data, origin='bottom')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.25)
    ax2.set_title(plot2_label)
    fig.colorbar(img2, cax=cax2, orientation='horizontal').set_label(cb_label)

def get_map_datx(datx_file_loc):
    # opens a .datx file and gets the surface. Adapted from Kyle Van Gorkom.
    h5file = h5py.File(datx_file_loc) # must be .datx filetype for hdf5
    
    # Get surface and attributes
    surface_snum = list(h5file['Data']['Surface'])[0]
    surface = h5file['Data']['Surface'][surface_snum].value
    surface_attrs = h5file['Data']['Surface'][surface_snum].attrs
    
    # compile parameters of the surface data
    surf_parms = {'wavelength': surface_attrs['Wavelength'][0],
                  'lateral_res': surface_attrs['X Converter'][0][2][1], # m/pix
                  'scale_factor': surface_attrs['Interferometric Scale Factor'][0],
                  'units': surface_attrs['Unit'][0]}
    surf_parms['diameter_mm'] = surf_parms['lateral_res']*np.shape(surface)[0]*1e3 * u.mm
    
    # Build mask from "No Data" attributes
    mask = np.ones_like(surface).astype(bool)
    mask[surface == surface_attrs['No Data']] = 0
    
    # Apply mask (will always be true)
    surface[~mask] = 0
    
    # Mask and scale the surface data to convert to microns
    #if do_scale:
    if surf_parms['units'] == b'Fringes':
        surf_out = surface * surf_parms['scale_factor'] * surf_parms['wavelength'] * 1e6 * u.micron
        surf_parms['units'] = str(u.micron)
        # convert to surface waves, multiply by wavelength [m], convert to microns
    else:
        surf_out = surface
    
    return surf_out, mask, surf_parms

def adjustData(optic):
    # Adjusts data so it becomes an even, square matrix.
    # fix the size to set to a square matrix
    # if not same shape, cut down the edges of where there would be extra zeros.
    '''
    if np.shape(optic)[0] < np.shape(optic)[1]:
        diff = np.int(np.abs(np.shape(optic)[0] - np.shape(optic)[1]))
        if diff == 1: # if too short by 1 row, add a column of zeros.
            zero_col = np.zeros((np.shape(optic)[1]))
            optic_data = np.vstack((optic,zero_col))
        else:
            if diff % 2 == 0:
                optic_data = optic[:,np.int(diff/2):np.shape(optic)[1]-np.int(diff/2)]
            else: # odd and 3+, untested
                h_optic = optic[:,np.int((diff-1)/2):np.shape(optic)[1]-np.int((diff-1)/2)]
                zero_col = np.zeros((np.shape(h_optic)[1]))
                optic_data = np.vstack((h_optic,zero_col))
                
    elif np.shape(optic)[0] > np.shape(optic)[1]:
        diff = np.int(np.abs(np.shape(optic)[0] - np.shape(optic)[1]))
        if diff == 1:
            zero_row = np.zeros((np.shape(optic)[0],1))
            optic_data = np.hstack((optic,zero_row))
        else:
            if diff % 2 == 0:
                optic_data = optic[np.int(diff/2):np.shape(optic)[0]-np.int(diff/2),:]
            else: # odd and 3+, untested
                h_optic = optic[np.int((diff-1)/2):np.shape(optic)[0]-np.int((diff-1)/2),:]
                zero_row = np.zeros((np.shape(h_optic)[0],1))
                optic_data = np.hstack((h_optic,zero_row))
    
    else:
        optic_data = optic
    '''
    # make the matrix square, turns out I had old code I wrote in the past
    if np.shape(optic)[0] != np.shape(optic)[1]:
        optic_data = zeroPadSquare(optic)
    else:
        optic_data = optic
    
    # at this point, the data should be square, but is it an odd or even size?
    # All the code works only for even.
    if np.shape(optic_data)[0] % 2 != 0: # if odd, add a row and column of zeros.
        z_col = np.zeros((np.shape(optic_data)[0]))
        nt = np.vstack((optic_data,z_col))
        z_row = np.zeros((np.shape(nt)[0],1))
        surf_data = np.hstack((nt,z_row))
    else: # if even, then let it be
        surf_data=optic_data
    #print(np.shape(optic_data))
    
    return surf_data

def sn_map(surface, mask):
    # produce the surface map with nans outside mask
    mask_bool = copy.copy(mask).astype(bool) # in case mask pushed in is not boolean
    surf_nan = copy.copy(surface)
    for r in range(0, np.shape(surface)[0]):
        for c in range(0,np.shape(surface)[1]):
            if mask_bool[r][c] == False:
                surf_nan[r][c] = np.nan
    return surf_nan

def interp_mask(ap_coord, mask_data):
    # Makes an interpolation mask when needing to set an active aperture region
    imask = np.ones(np.shape(mask_data))
    ind = []
    for n in range(0, np.shape(ap_coord)[1]):
        yy = ap_coord[0][n]
        xx = ap_coord[1][n]
        if mask_data[yy][xx] == 0:
            ind.append([yy, xx])
            imask[yy][xx] = 0
    return ind, imask

def fill_surface(surface, mask_data, ap_clear, ap_coords, method='cubic'):
    # fill in the missing bad pixels using cubic interpolation
    side_len = np.shape(surface)[0]
    ind, imask = interp_mask(ap_coords, mask_data)
    working_data = imask*surface*ap_clear
    data_values = working_data[imask.astype(bool)==True] # pick where the data is good
    yy,xx = np.indices(working_data.shape)
    mask_points = np.argwhere(imask==1) # indices where the data is good
    grid_z2 = interpolate.griddata(mask_points, data_values, (yy,xx), method=method)
    return grid_z2
    
def write2fits(surface, mask, surf_parms, filename, surf_nan=False):
    # write specific data to fits file
    header = fits.Header()
    header['diameter'] = (surf_parms['diameter_mm'].value, 'full measurement diameter [mm]')
    header['latres'] = (surf_parms['lateral_res'], 'meters/pixel')
    header['wavelen'] = (surf_parms['wavelength'], 'Zygo wavelength [meters]')
    # write mask file
    fits.writeto(filename + '_mask.fits', mask.astype(int), header, overwrite=True)
    # write surface file
    if surf_nan:  # if the surface to be written to FITS should be the masked nan version
        surface = sn_map(surface,mask)
        header['maskVal'] = ('nan', 'mask units')
        surf_filename = filename+'_surf_nan'
    else: # by default the surface written does not have the nan mask
        header['maskVal'] = ('zeros', 'mask units')
        surf_filename = filename+'_surf'
    header['UNITS'] = (surf_parms['units'], 'surface units')
    if hasattr(surface, 'unit'):
        surf_val = surface.value
    else:
        surf_val = surface
    fits.writeto(surf_filename + '.fits', surf_val, header, overwrite=True)

def datx2fits(datx_file_loc, filename, surf_nan=False):
    # Automatically write the .datx file into fits for the surface and mask
    surface, mask, surf_parms = get_map_datx(datx_file_loc)
    write2fits(surface, mask, surf_parms, filename, surf_nan=surf_nan)
