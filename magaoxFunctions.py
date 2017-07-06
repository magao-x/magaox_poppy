#########################################
# Author: Jennifer Lumbres (contact: jlumbres@optics.arizona.edu)
# Last edit: 2017/06/15
# This file is meant to be a reference for the extra functions written for MagAO-X POPPY.

#########################################
# PACKAGE IMPORT
#########################################
#load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.io import fits
from  matplotlib.colors import LogNorm
import scipy.ndimage
#POPPY
import poppy
from poppy.poppy_core import PlaneType

#########################################
# FUNCTION DEFINITIONS
#########################################

# Function: surfFITS
# Description: Initiates a FITS file to add to optical system.
# Input Parameters:
#   file_loc    - string    - path location of FITS file
#   optic_type  - string    - Declare if the file is OPD or Transmission type ('opd' or 'trans')
#   opdunit     - string    - OPD units of FITS file. For some reason, BUNIT header card gives errors.
#   name        - string    - descriptive name for optic. Useful for phase description.
# Output Parameters:
#   optic_surf  - FITSOpticalElement    - Returns FITSOpticalElement to use as surface mapping file.
# Sequence of Initializing:
#   - Call in FITS file
#   - Typecast FITS data to float type (workaround to get POPPY to accept FITS data)
#   - Determine optic type to choose how to build FITSOpticalElement
#   - Return FITSOpticalElement object
def surfFITS(file_loc, optic_type, opdunit, name):
    optic_fits = fits.open(file_loc)
    optic_fits[0].data = np.float_(optic_fits[0].data) # typecasting for POPPY workaround
    if optic_type == 'opd':
        optic_surf = poppy.FITSOpticalElement(name = name, opd=optic_fits, opdunits = opdunit)
    else:
        optic_surf = poppy.FITSOpticalElement(name = name, transmission=optic_fits)
    return optic_surf

# Function: makeRxCSV
# Desription: Get the system prescription from CSV file
# FYI: This has some hardcoded numbers in it, but just follow the specs on the CSV file.
# Input parameters:
#    csv_file   - CSV file location
# Output parameters:
#    sys_rx     - system prescription into a workable array format
def makeRxCSV(csv_file):
    sys_rx=np.genfromtxt(csv_file, delimiter=',', dtype="i2,U19,U10,f8,f8,f8,U90,U90,U10,U10,f8,U10,", skip_header=15,names=True)
    print('CSV file name: %s' % csv_file)
    print('The names of the headers are:')
    print(sys_rx.dtype.names)
    return sys_rx

# Function: csvFresnel
# Description: Builds FresnelOpticalSystem from a prescription CSV file passed in
# Input parameters:
#    rx_csv      - system prescription
#    res         - resolution
#    oversamp    - oversampling convention used in PROPER
#    break_plane - plane to break building the MagAO-X prescription
# Output:
#    sys_build   - FresnelOpticalSystem object with all optics built into it
def csvFresnel(rx_csv, samp, oversamp, break_plane):
    M1_radius=rx_csv['Radius_m'][1]*u.m # Element [1] is M1 because Element [0] is the pupil mask
    
    sys_build = poppy.FresnelOpticalSystem(pupil_diameter=2*M1_radius, npix=samp, beam_ratio=oversamp)

    # Entrance Aperture
    sys_build.add_optic(poppy.CircularAperture(radius=M1_radius))

    # Build MagAO-X optical system from CSV file to the Lyot plane
    for n_optic,optic in enumerate(rx_csv): # n_optic: count, optic: value

        dz = optic['Distance_m'] * u.m # Propagation distance from the previous optic (n_optic-1)
        fl = optic['Focal_Length_m'] * u.m # Focal length of the current optic (n_optic)

        #print('Check PSD file for %s: %s' % (optic['Name'], optic['surf_PSD']))
        # if PSD file present
        if optic['surf_PSD_filename'] != 'none':
            # make a string insertion for the file location
            surf_file_loc = optic['surf_PSD_folder'] + optic['surf_PSD_filename'] + '.fits'
            # call surfFITS to send out surface map
            optic_surface = surfFITS(file_loc = surf_file_loc, optic_type = optic['optic_type'], opdunit = optic['OPD_unit'], name = optic['Name']+' surface')
            # Add generated surface map to optical system
            sys_build.add_optic(optic_surface,distance=dz)

            if fl != 0: # powered optic with PSD file present
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name'])) 
                # no distance; surface comes first
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            elif optic['Type'] != 'pupil': # non-powered optic but has PSD present that is NOT the pupil
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

        # if no PSD file present (DM, focal plane, testing optical surface)
        else:
            # if powered optic is being tested
            if fl !=0: 
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name']), distance=dz)
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # for DM, flat mirrors
            elif optic['Type'] == 'mirror' or optic['Type'] == 'DM':
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            else: # for focal plane, science plane, lyot plane
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)

        # if the most recent optic studied was the break plane, break out of loop.
        if optic['Type'] == break_plane: 
            #print('Finish building FresnelOpticalSystem at %s' % break_plane)
            break
        
    return sys_build

# Function: BuildLyotDMSurf
# Description: Creates Tweeter DM surface map generated by Lyot plane
# Input Parameters:
#    lyot_phase_data    - Lyot phase data in matrix format
#    samp_size          - The sampling size for the system
#    tweeter_diam       - Diameter of Tweeter DM; should be 18.8mm according to Laird Close's system design
#    pupil_mask_data    - pupil mask data in matrix format, must be 512x512 image
#    magK               - Spatial Frequency magnitude map
#    DM_ctrl_BW         - Tweeter DM control bandwidth for LPF
#    wavelength         - wavelength tested
#    lyot_DM_loc        - location to write for lyot DM surface map
# Outputs:
#    none; just does the task.
def BuildLyotDMSurf(lyot_phase_data, samp_size, tweeter_diam, pupil_mask_data, magK, DM_ctrl_BW, wavelength, lyot_DM_loc):
    # Crop Lyot 
    center_pix = 0.5*(lyot_phase_data.shape[0]-1)
    #shift=255.5 # hardcoded, leave me alone
    shift = (samp_size/2) - 0.5 # get to center of pixel
    h_lim = np.int(center_pix+shift) # change to integer to get rid of warning
    l_lim = np.int(center_pix-shift) 
    lyot_phase = lyot_phase_data[l_lim:h_lim+1,l_lim:h_lim+1]
    
    # Multiply Lyot with pupil mask
    lyot_mask = pupil_mask_data*lyot_phase
    
    # Take FT of Lyot to get to focal plane for LPF
    FT_lyot = np.fft.fft2(lyot_mask)
    
    # LPF on FT_Lyot
    filter_lyot = np.zeros((samp_size,samp_size),dtype=np.complex128)
    for a in range (0,samp_size):
        for b in range (0,samp_size):
            #if (np.abs(kx[a][b]) < DM_ctrl_BW) and (np.abs(ky[a][b]) < DM_ctrl_BW): #square corner version
            if magK[a][b] < DM_ctrl_BW: # Curved corner version
                filter_lyot[a][b] = FT_lyot[a][b] # Keep FT value if less than DM BW
                
    # Post-LPF IFT
    lpf_lyot = np.fft.ifft2(filter_lyot)
    
    # Convert Phase to DM surface
    lpf_lyot_surf =(-1.0*wavelength.value/(2*np.pi))*np.real(lpf_lyot) # FINAL!
    
    # Multiply by pupil mask to clean up ringing
    lpf_lyot_mask = pupil_mask_data*np.real(lpf_lyot_surf)
    
    # Write DM surface to file
    lyot_fits = fits.PrimaryHDU(data=lpf_lyot_mask)
    lyot_fits.header.set('PUPLSCAL', tweeter_diam.value/samp_size)
    lyot_fits.header.set('BUNIT', 'meters')
    lyot_fits.writeto(lyot_DM_loc+'.fits', clobber=True)

# Function: SpatFreqMap
# Description: Builds spatial frequency map to be used with 
# Input Parameters:
#    M1_radius  - radius of primary mirror  
#    num_pix    - side length of test region (512 is passed in for MagAO-X)
# Output:
#    magK       - spatial frequency map
def SpatFreqMap(M1_radius, num_pix):
    sample_rate = (M1_radius.value*2)/num_pix
    
    FT_freq = np.fft.fftfreq(num_pix,d=sample_rate)
    kx = np.resize(FT_freq,(FT_freq.size, FT_freq.size))
    
    # Build ky the slow way
    y_val=np.reshape(FT_freq,(FT_freq.size,1))
    ky=y_val
    for m in range (0,y_val.size-1):
        ky=np.hstack((ky,y_val))
    magK = np.sqrt(kx*kx + ky*ky)
    return magK

# Function: vappOPD(opd_surf_data, header_info, fileloc, test_optic)
# Description: Writes vAPP OPD mask to FITS file, WILL OVERRIDE OLD FILE IF fileloc IS REUSED
# Input Parameters:
#    opd_surf_data   - OPD surface data
#    vapp_pixelscl   - pixel scale on m/pix
#    fileloc         - file string location for vAPP OPD mask FITS file
# Output:
#    none (just does the thing)
def vappOPDfile(opd_surf_data, vapp_pixelscl, fileloc):
    vapp_OPD = fits.PrimaryHDU(data=opd_surf_data)
    vapp_OPD.header.set('PUPLSCAL', vapp_pixelscl)
    vapp_OPD.header.comments['PUPLSCAL'] = 'pixel scale [m/pix]'
    vapp_OPD.header.set('BUNIT', 'meters')
    vapp_OPD.header.comments['BUNIT'] = 'opd units'
    vapp_OPD.writeto(fileloc, clobber=True)

# Function: vappTRANS
# Description: Writes vAPP transmission mask to FITS file, WILL OVERRIDE OLD FILE IF fileloc IS REUSED
# Input Parameters:
#    trans_data      - transmission data
#    vapp_pixelscl   - pixel scale on m/pix
#    fileloc         - file string location for vAPP transmission mask FITS file
# Output:
#    none (just does the thing)
def vappTRANSfile(trans_data, vapp_pixelscl, fileloc):
    vapp_trans = fits.PrimaryHDU(data=trans_data)
    vapp_trans.header.set('PUPLSCAL', vapp_pixelscl)
    vapp_trans.header.comments['PUPLSCAL'] = 'pixel scale [m/pix]'
    vapp_trans.writeto(fileloc, clobber=True)
    
# Function: calcDHflux
# Description: Calculates the flux inside a region of the dark hole
# Input Parameters:
#    psf_data        - PSF data information to dig out dark hole
#    DH_center       - center pixel of dark hole
#    DH_side         - half side length of dark hole (a shift of sorts)
#    calcType        - choose between calculating flux in median or mean
# Output:
#    DH_flux         - flux inside the darh hole region, whether median value or mean
def calcFluxDH_List(file_loc, DH_center, DH_side, optics_list, calcType):
    DH_flux_array = []
    DH_flux_ref = 'none'
    
    for test_optic in optics_list:
        file_Data = fits.open(file_loc + test_optic + '.fits')[0].data
        DH_flux = calcDHflux(file_Data, DH_center, DH_side, calcType)
        DH_flux_array.append(DH_flux)
        
        if test_optic == 'noneRemoved':
            DH_flux_ref = DH_flux
    
    return DH_flux_array, DH_flux_ref


# Function: calcDHflux
# Description: Calculates the flux inside a region of the dark hole
# Input Parameters:
#    psf_data        - PSF data information to dig out dark hole
#    DH_center       - center pixel of dark hole
#    DH_side         - half side length of dark hole (a shift of sorts)
#    calcType        - choose between calculating flux in median or mean
# Output:
#    DH_flux         - flux inside the darh hole region, whether median value or mean
def calcDHflux(psf_data, DH_center, DH_side, calcType):
    peak_value = np.amax(np.amax(psf_data))
    contrast_img = psf_data/peak_value
    
    DHside_high = [x+DH_side for x in DH_center]
    DHside_low = [x-DH_side for x in DH_center]
    DH_region = contrast_img[DHside_low[0]:DHside_high[0],DHside_low[1]:DHside_high[1]]
    
    if calcType == 'mean':
        DH_flux = np.mean(DH_region)
    elif calcType == 'median':
        DH_flux = np.median(DH_region)
    
    return DH_flux
