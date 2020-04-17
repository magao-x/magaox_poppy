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
    
def calc_PSD2D(folder_loc, surf_data_loc, opt_parms, savePSD = True):
    n_oap = opt_parms['oap_name']
    
    # Open up the data. Assumes the data is clean and square before this point.
    optic_data, optic_header = fits.getdata(folder_loc+surf_data_loc, header=True)
    opt_parms['latres'] = optic_header['latres']
    opt_parms['diam_full'] = optic_header['latres']*opt_parms['rpix_full']*2*1e3*u.mm # this is the full diameter
    
    # convert the surface units from micron to mm (assumes data coming in is in micron)
    surf_data_mm = optic_data * u.um * u.mm / (1e3*u.um)
    print('surf unit check: {0}'.format(surf_data_mm.unit))
    
    # calculate the power optic data
    (psd2d_opt, d_npix, ap_var) = calc_surfPower(surf_data_mm, opt_parms)
    opt_parms['npix_diam_ca'] = d_npix
    opt_parms['diam_ca'] = d_npix*opt_parms['latres']*1e3*u.mm
    
    print('test diameter: {0:.3f} ({1:.2f}% CA)'.format(opt_parms['diam_ca'], opt_parms['diam_ca']/opt_parms['diam_100CA']*100))

    # convert ap_var from mm^2 to nm^2
    ap_var *= 1e6*(u.nm/u.mm) * 1e6*(u.nm/u.mm) # convert variance mm2 to nm2
    opt_parms['var'] = ap_var
    print('var: {0:.3f}'.format(ap_var))

    # calculate the spatial frequency content
    (del_k, FT_freq) = calc_spatialFreq(opt_parms)
    opt_parms['del_k'] = del_k
    #print('delta k: {0:.5f}'.format(del_k))

    # Normalized the PSD
    norm_psd = calc_normPSD(psd2d_opt, del_k)
    print('post-normalization var check: {0:.3f} (should match 1.000)'.format(np.sum(norm_psd) * (del_k**2)))
    #cal_psd = norm_psd * ap_var # there is function calc_calPSD but that's for a different time
    #print('calibrated var check: {0:.3f} (should match {1:.3f})'.format(np.sum(cal_psd) * (del_k**2), ap_var))
    
    # SPECIAL CASE
    if (opt_parms['surf_type']=='coated') and (n_oap == 'n2'):
        print('psd for {0} needs masking to eliminate blips'.format(n_oap))
        n2_mask_centers = [(3107, 1922), (990, 2175)]
        n2_radius = [28, 28]
        mask = psd_mask(n2_mask_centers, n2_radius, opt_parms)
        psd2d_opt *= mask
        norm_psd *= mask
    
    if savePSD == True: # save the 2DPSD data
        header = fits.Header()
        fits_data = np.single(psd2d_opt)
        header['datatype'] = str(fits_data.dtype)
        fits.writeto(folder_loc+'psd2d_{0}_{1}_{2}.fits'.format(opt_parms['surf_type'], n_oap, header['datatype']), fits_data.value, header, overwrite=True)
    
    return(surf_data_mm, psd2d_opt, norm_psd, FT_freq, opt_parms)

#def calc_surfPower(surf_data, D_full_1, ca_new, oversamp_size):
def calc_surfPower(optic, opt_parms):
    # calculate the surface power
    surf_units = optic.unit
    optic = optic.value
    full_diam = opt_parms['diam_full'] # must be passed in mm units
    ca_percent = opt_parms['ca'] # something like 98.5%
    ovs = opt_parms['ovs']
    # check if the optics are same shape. 
    ''' 
    # if not same shape, cut down the edges of where there would be extra zeros.
    if np.shape(optic)[0] < np.shape(optic)[1]:
        diff = np.int(np.abs(np.shape(optic)[0] - np.shape(optic)[1]))
        optic_data = optic[:,np.int(diff/2):np.shape(optic)[1]-np.int(diff/2)]
    elif np.shape(optic)[0] > np.shape(optic)[1]:
        diff = np.int(np.abs(np.shape(optic)[0] - np.shape(optic)[1]))
        optic_data = optic[np.int(diff/2):np.shape(optic)[0]-np.int(diff/2),:]
    else: # if they are the same size, check if it's odd or even.
        if np.shape(optic)[0] % 2 != 0: # if odd, add a row and column of zeros.
            zero_col = np.zeros((np.shape(optic)[0]))
            nt = np.vstack((optic,zero_col))
            zero_row = np.zeros((np.shape(nt)[0],1))
            optic_data = np.hstack((nt,zero_row))
        else: # if even, then let it be
            optic_data=optic
    '''
    '''
    # If the data is not square, pick the shortest side for npix_diam
    if np.shape(optic)[0] <= np.shape(optic)[1]:
        side_diam = np.shape(optic)[0]
    else:
        side_diam = np.shape(optic)[1]
    '''
    # verify the data to be even, square matrix
    if (np.shape(optic)[0] != np.shape(optic)[1]) or (np.shape(optic)[0] % 2 != 0):
        optic_data = adjustData(optic)
    else:
        optic_data = optic

    # Create the clear aperture mask based on the clear aperture percent
    data_side = np.shape(optic_data)[0]
    npix_diam = np.int(data_side*ca_percent/100) # integer required for hannWin
    # removed because assume data come in is square and not necessarily even
    # npix_diam = np.int(side_diam*ca_percent/100)
    # npix_diam must be even to get the padding to work correctly
    # removed because maybe not needed?
    if npix_diam %2 != 0: # if odd
        npix_diam -= 1 # decrease by 1 to force to even
    ap_clear = np.zeros((data_side, data_side), dtype=np.uint8)
    ca_coords = draw.circle(data_side/2, data_side/2, radius=(npix_diam/2))
    ap_clear[ca_coords] = True

    # calculate the mean and variance of the active region of data
    ap_active = optic_data * ap_clear
    ap_active_coords = optic_data[ap_clear==True]
    ap_avg = np.mean(ap_active_coords)
    ap_var = np.var(ap_active_coords)*surf_units*surf_units
    #ap_var = np.var(ap_active_coords)

    # Subtract the mean from the data
    ap_submean = (ap_active - ap_avg) * ap_clear

    # build the Hann 2D window
    hannWin = han2d((npix_diam, npix_diam))
    # zero-pad the window to fit with the full aperture dimensions
    pad_side = np.int((np.shape(optic_data)[0] - npix_diam)/2)
    pad_hann = np.pad(hannWin, pad_side, mf.pad_with) * ap_clear

    # oversample the windowed data and take FFT for the power
    optic_ovs = zeroPadOversample(ap_submean * pad_hann, np.int(ovs/getSampSide(ap_clear)))*surf_units
    power_optic = calcPower2(optic_ovs)/(surf_units*surf_units)
    
    return power_optic, npix_diam, ap_var
    
def calc_spatialFreq(opt_parms):
    #def calc_spatialFreq(ovs, npix_diam, ca_percent, full_diam):
    # calculate the spatial frequency content
    # declare the variables from the parameters
    ca_diam = opt_parms['diam_ca'] #ca_diam = full_diam * (ca_percent/100)
    npix_diam = opt_parms['npix_diam_ca']
    ovs = opt_parms['ovs']
    
    sample_space = ca_diam / npix_diam # should be in units of mm/pixels
    delta_k = 1/(ovs*sample_space)
    FT_freq = np.fft.fftfreq(n=ovs,d=sample_space)
    return delta_k, FT_freq

def calc_normPSD(power_optic, delta_k):
    sigB = np.sum(power_optic)*(delta_k**2)
    norm_psd = power_optic / sigB
    
    # verify normalized psd:
    var_verify = np.sum(norm_psd) * (delta_k**2) # unitless and 1
    psd_verify = np.allclose(1, var_verify) # both unitless and 1
    if psd_verify == True:
        print('PSD normalized: var={:.3f}'.format(var_verify))
    else:
        print('PSD not normalized: var={:.3f}'.format(var_verify))
    
    return norm_psd

def psd_mask(center, radius, opt_parms):
    n_cen = np.shape(center)[0]
    
    # build the general mask
    mask = np.ones((opt_parms['ovs'], opt_parms['ovs']), dtype=np.float64)
    
    # fill in the mask
    for n_circ in range(0, n_cen):
        mc = np.zeros_like(mask)
        mc_coords = draw.circle(center[n_circ][0], center[n_circ][1], radius=radius[n_circ])
        mc[mc_coords] = True
        mask[mc==True] = np.nan
    
    return mask

def calc_psd_rms(pwr_opt, FT_freq, ind_limits):    
    # make a grid for the average radial power value
    #shift = np.int(FT_freq.size/2)
    shift = np.int(FT_freq.size/2)
    maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
    
    # get full range of radial frequency
    radialFreq = FT_freq[0:shift]
    dr = radialFreq[1] - radialFreq[0]
    
    # calculate the size of the donut:
    ring_width = ind_limits[1] - ind_limits[0]
    
    # make the mask
    radial_mask = makeRingMask(maskY, maskX, ind_limits[0], ring_width)
    #npix_mask = np.sum(np.sum(radial_mask)) # counts number active pixels
    pix_area = (dr**2)
    radial_bin = makeRingMaskBin(pwr_opt.value,radial_mask) * pwr_opt.unit
    #return (npix_mask, mask_area, radial_bin)
    
    rms_val = np.sqrt(np.sum(radial_bin * pix_area))
    
    return (rms_val, radial_mask)
    

# Function: makeRingMask
# Description: Makes the radial median mask, which looks like a ring.
# Input Parameters:
#   y           - meshgrid vertical values (pixel count units)
#   x           - meshgrid horizontal values (pixel count units)
#   inner_r     - inner radial value (pixel count units)
#   dr          - ring thickness (pixel count units)
# Output:
#   ringMask    - ring mask (boolean type)
def makeRingMask(y,x,inner_r,r_width):
    inside_mask = x**2+y**2 <= inner_r**2
    outside_mask = x**2+y**2 <= (inner_r+r_width)**2
    ringMask = outside_mask ^ inside_mask # use xor, such that so long as one is true then it will make ring.
    return ringMask
    
# Function: makeRingMaskBin
# Description: Returns the bin of values from the ring mask
# Input Parameters:
#   power_data  - wavefront power data, must be square matrix
#   ringMask    - ring mask, must be same size as power_data and boolean type
# Output:
#   ringMaskBin - vector of values that survived through the mask
def makeRingMaskBin(power_data, ringMask):
    ringMaskBin = np.extract(ringMask, power_data)
    ringbin = ringMaskBin[~np.isnan(ringMaskBin)]
    return ringbin

def k_locate(freqrange, k_tgt):
    # given a target spatial frequency, find the index bin and value closest to target.
    kscale = np.abs(freqrange.value - k_tgt.value)
    bb = np.where(kscale==np.amin(kscale))
    return (bb[0][0], freqrange[bb][0])

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
    
def calc_radialPSD2(norm_optic, cal_optic, FT_freq, ring_width, k_lowlim):    
    # make a grid for the average radial power value
    shift = np.int(FT_freq.size/2)
    maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
    
    # get full range of radial frequency
    radialFreq = FT_freq[0:shift]
    
    # initialize content
    mean_psd = [] # initialize empty list of mean PSD values
    mean_psd2 = []
    k_val = [] # initialize empty list of spatial frequencies
    
    # set up ring parameters
    # check if ring_width is odd
    if ring_width % 2 == 0:
        ring_width += 1 # increase by 1 to make it into odd 
    r_halfside = np.int((ring_width-1)/2)
    r = 1 # initialize at start
    
    #k_lowlim = 1/(full_diam*(ca_percent/100)).value
    
    # clug along through all the radial frequency values
    while((r+r_halfside)<shift): # while inside the region of interest
        ri = r - r_halfside # inner radius of ring
        if radialFreq[r].value < k_lowlim: # verify that position r is at the low limit
            #print('test k-value too small, iterate to next')
            r+=1
        else:
            if ri > 0:
                radial_mask = makeRingMask(maskY, maskX, ri, ring_width)
                radial_bin = makeRingMaskBin(norm_optic.value,radial_mask)
                mean_psd.append(np.mean(radial_bin))
                radial_bin2 = makeRingMaskBin(cal_optic.value, radial_mask)
                mean_psd2.append(np.mean(radial_bin2))
                k_val.append(radialFreq[r].value)
            r+=ring_width # iterate to the next r value in the loop
    
    k_val = k_val * radialFreq.unit
    mean_psd = mean_psd * norm_optic.unit
    mean_psd2 = mean_psd2 * cal_optic.unit
    
    return (k_val, mean_psd, mean_psd2)

def calc_beta(alpha, var, D_clear, npix_diam):
    kmin = 1 / D_clear
    kmax = 1/ (2 * D_clear / npix_diam)
    
    if alpha == 2:
        beta = var / ( 2* np.pi * np.log(kmax/kmin) )
    else: # when beta is not 2
        beta = var * (alpha - 2) / (2 * np.pi * ( (kmin**(2 - alpha)) - (kmax**(2 - alpha)) ) )
    
    return beta

# Function: calcPower
# Description: Calculate the power of a wavefront, without putting it in exponential (needs better description)
# Input Parameters:
#   wf_data     - wavefront data
# Output:
#   power_data  - REAL PART modulous square of the fourier transform with the wavefront data
def calcPower2(wf_data):
    FT_wf = np.fft.fftshift(np.fft.fft2(wf_data))
    power_data = np.real(FT_wf*np.conjugate(FT_wf))
    return power_data

# Function: getSampSide
# Description: Calculates the sample side based on the largest side of the image
# Input Parameters:
#   optic_data  - 2D array of the data
# Output:
#   samp        - Sample side value
def getSampSide(optic_data):
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

# Function: zeroPadSquare
# Description: Zero pad a non-symmetric array into a square array
# Input Parameters:
#   optic_data  - 2D array of data, non-symmetric in size
# Output:
#   sym_data    - zero-padded symmetric array (size is of the larger side)
def zeroPadSquare(optic_data):
    squareMat = copy.copy(optic_data)
    n_row = np.shape(optic_data)[0]
    n_col = np.shape(optic_data)[1]
    side_diff = np.abs(n_row - n_col)
    #print('side difference: %d' % side_diff)
    
    # check if the difference is odd or even
    if side_diff % 2 == 0:
        odd_diff = False
        #print('The difference is even.')
    else:
        odd_diff = True
        #print('The difference is odd.')
    
    # count how many times to add row/col on both sides (hence divide by 2)
    count = np.int(np.floor(side_diff/2))
    #print('count = %d' % count)
    
    # fill in the matrix
    if n_row > n_col: # if too many rows over columns, fill in more columns both sides
        #print('There are more columns than rows')
        filler_row = np.zeros(n_row)[np.newaxis]
        for c in range(0,count):
            #print('entered the filler loop')
            squareMat = np.hstack((filler_row.T,np.hstack((squareMat,filler_row.T))))
        if odd_diff == True: # add one more column on left if odd difference
            squareMat = np.hstack((filler_row.T,squareMat))
        #print('This is the new matrix dimensions: %d, %d' % (np.shape(squareMat)[0], np.shape(squareMat)[1]))
    
    elif n_col > n_row: # too many columns than rows
        #print('There are more rows than columns')
        filler_col = np.zeros(n_col)
        for c in range(0,count):
            #print('entered the filler loop')
            squareMat = np.vstack((filler_col,np.vstack((squareMat,filler_col))))
        if odd_diff == True:
            squareMat = np.vstack((filler_col,squareMat))
        #print('This is the new matrix dimensions: %d, %d' % (np.shape(squareMat)[0], np.shape(squareMat)[1]))
    
    return squareMat

# Function: zeroPadOversample
# Description: makes a zero pad based on some oversampling requirements
# Input Parameters:
#   optic_data  - 2D array of the data
#   oversamp    - oversampling multiplier
# Output:
#   zp_wf       - zero padded wavefront
def zeroPadOversample(optic_data,oversamp):
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

# Function: doCenterCrop
# Description: crops image into square from center of image
# Input Parameters:
#   optic_data  - original 2D optic data, assumes SQUARE data
#   shift       - how many pixels to shift on each side
# Output:
#   crop_data   - cropped data
def doCenterCrop(optic_data,shift):
    side = np.shape(optic_data)[0]
    center = np.int(side/2)
    crop_data = optic_data[center-shift:center+shift,center-shift:center+shift]
    return crop_data

# Code generated by Kyle Van Gorkom
def get_radial_dist(shape, scaleyx=(1.0, 1.0)):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

# Code generated by Kyle Van Gorkom
def han2d(shape, fraction=1./np.sqrt(2), normalize=False):
    '''
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



###########################
# DEPRECIATED CODE
###########################

# Function: calcPower
# Description: Calculate the power of a wavefront (needs better description)
# Input Parameters:
#   wf_data     - wavefront data
#   wavelength  - wavelength used to measure
# Output:
#   power_data  - REAL PART modulous square of the fourier transform with the wavefront data
def calcPower(wf_data, wavelength):
    wavefront = np.exp((2*np.pi/wavelength)*1j*wf_data)
    FT_wf = np.fft.fftshift(np.fft.fft2(wavefront))
    power_data = np.real(FT_wf*np.conjugate(FT_wf))
    return power_data


def calc_calPSD(power_optic, delta_k, ap_var):
    # redundant code that was replaced for shorter lines
    norm_psd = calc_normPSD(power_optic, delta_k)
    cal_psd = norm_psd * ap_var
    
    # verify calibrated PSD:
    var_verify = np.sum(cal_psd) * (delta_k**2)
    psd_verify = np.allclose(ap_var.value, var_verify.value)
    if psd_verify == True:
        print('PSD calibrated: var={:.3f}'.format(var_verify))
    else:
        print('PSD not calibrated: var={:.3f}'.format(var_verify))
        
    return cal_psd

def calc_radialPSD(power_optic, FT_freq, ring_width, full_diam, ca_percent):  
    # replaced with calc_radialPSD2
    # make a grid for the average radial power value
    shift = np.int(FT_freq.size/2)
    maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
    
    # get full range of radial frequency
    radialFreq = FT_freq[0:shift]
    
    # initialize content
    mean_psd = [] # initialize empty list of mean PSD values
    k_val = [] # initialize empty list of spatial frequencies
    
    # set up ring parameters
    # check if ring_width is odd
    if ring_width % 2 == 0:
        ring_width += 1 # increase by 1 to make it into odd 
    r_halfside = np.int((ring_width-1)/2)
    r = 1 # initialize at start
    
    k_lowlim = 1/(full_diam*(ca_percent/100)).value
    
    # clug along through all the radial frequency values
    while((r+r_halfside)<shift):
        ri = r - r_halfside # inner radius of ring
        if radialFreq[r].value < k_lowlim: # verify that position r is at the low limit
            print('test k-value too small, iterate to next')
            r+=1
        else:
            if ri > 0:
                radial_mask = makeRingMask(maskY, maskX, ri, ring_width)
                radial_bin = makeRingMaskBin(power_optic.value,radial_mask)
                mean_psd.append(np.mean(radial_bin))
                k_val.append(radialFreq[r].value)
            r+=ring_width # iterate to the next r value in the loop
    
    k_val = k_val * radialFreq.unit
    mean_psd = mean_psd * power_optic.unit
    
    return (k_val, mean_psd)



# Function: makeHannWindow
# Description: Builds a 2D Hann window
# Input Paramenters:
#   sideLen     - length of side for Hann window
# Output:
#   hannWin     - 2D (sideLen, sideLen) Hann window
def makeHannWindow(sideLen):
    hannSide = signal.hann(sideLen)
    hannArray = np.tile(hannSide,(sideLen,1)) # tile the rows
    hannWin = hannArray * hannSide[:, np.newaxis] # multiply across each column in tiled row array
    return hannWin


# Function: getRadialSpatFreq
# Description: Determines the spatial frequency value at a radial distance
# Input Parameters:
#   radialFreqVector    - radial frequency in vector format (can do vector since it's radially symmetric
#   r                   - index value for inner radial distance
#   dr                  - radial thickness value
# Output:
#   radialFreq          - radial frequency value
def getRadialSpatFreq(radialFreqVector, r, dr):
    radialFreq = ((radialFreqVector[r+dr] - radialFreqVector[r])/2)+radialFreqVector[r]
    return radialFreq

# Function: SpatFreqMap
# Description: Builds spatial frequency map for PSD generation (pulled from magaox_poppy code)
# Input Parameters:
#   optic_radius    - radius of primary mirror  
#   num_pix         - side length of test region (512 is passed in for MagAO-X)
# Output:
#    magK           - spatial frequency map with proper circular pattern
def SpatFreqMap(optic_radius, num_pix):
    sample_rate = (optic_radius*2)/num_pix
    
    FT_freq = np.fft.fftfreq(num_pix,d=sample_rate)
    kx = np.resize(FT_freq,(FT_freq.size, FT_freq.size))
    
    # Build ky the slow way
    y_val=np.reshape(FT_freq,(FT_freq.size,1))
    ky=y_val
    for m in range (0,y_val.size-1):
        ky=np.hstack((ky,y_val))
    magK = np.fft.fftshift(np.sqrt(kx*kx + ky*ky))
    return magK



def filter_Zygo(surf_data, even_square = True):
    '''
    Filters the garbage from Zygo data
    DEPRECIATED, USE ANOTHER CODE
    '''
    '''
    if even_square == True:
        surf_value = adjustData(surf_data)
    else:
        surf_value = copy.copy(surf_data) # do this to make sure everything isn't the same
    '''
    surf_value = adjustData(surf_data)
    row_side = np.shape(surf_value)[0]
    col_side = np.shape(surf_value)[1]
    mask = np.ones((row_side, col_side)) # the mask is all true initially
    for row in range(0,row_side):
        for col in range(0,col_side):
            if np.absolute(surf_value[row][col]) > 10e3: #Zygo makes out-of-region values at e+308... whyyy
                mask[row][col] = 0; # logical false for the mask
    mask_data = mask*surf_value
    
    return mask_data, mask
