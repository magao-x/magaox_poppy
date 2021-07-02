"""
Analytic optical element classes to introduce a specified wavefront
error in an OpticalSystem

 * ZernikeWFE
 * ParameterizedWFE (for use with hexike or zernike basis functions)
 * SineWaveWFE
 * TODO: MultiSineWaveWFE ?
 * TODO: PowerSpectrumWFE
 * TODO: KolmogorovWFE

"""

import collections
from functools import wraps
import numpy as np
import astropy.units as u

from .optics import AnalyticOpticalElement, CircularAperture
from .poppy_core import Wavefront, PlaneType, BaseWavefront
from poppy.fresnel import FresnelWavefront

from . import zernike
from . import utils
from . import accel_math

__all__ = ['WavefrontError', 'ParameterizedWFE', 'ZernikeWFE', 'SineWaveWFE',
        'StatisticalPSDWFE', 'PowerSpectrumWFE']


def _check_wavefront_arg(f):
    """Decorator that ensures the first positional method argument
    is a poppy.Wavefront or FresnelWavefront
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], BaseWavefront):
            raise ValueError("The first argument must be a Wavefront or FresnelWavefront object.")
        else:
            return f(*args, **kwargs)
    return wrapper


class WavefrontError(AnalyticOpticalElement):
    """A base class for different sources of wavefront error

    Analytic optical elements that represent wavefront error should
    derive from this class and override methods appropriately.
    Defined to be a pupil-plane optic.
    """

    def __init__(self, **kwargs):
        if 'planetype' not in kwargs:
            kwargs['planetype'] = PlaneType.pupil
        super(WavefrontError, self).__init__(**kwargs)
        # in general we will want to see phase rather than intensity at this plane
        self.wavefront_display_hint = 'phase'

    @_check_wavefront_arg
    def get_opd(self, wave):
        """Construct the optical path difference array for a wavefront error source
        as evaluated across the pupil for an input wavefront `wave`

        Parameters
        ----------
        wave : Wavefront
            Wavefront object with a `coordinates` method that returns (y, x)
            coordinate arrays in meters in the pupil plane
        """
        raise NotImplementedError('Not implemented yet')

    def rms(self):
        """RMS wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')

    def peaktovalley(self):
        """Peak-to-valley wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')


def _wave_y_x_to_rho_theta(y, x, pupil_radius):
    """
    Return wave coordinates in (rho, theta) for a Wavefront object
    normalized such that rho == 1.0 at the pupil radius

    Parameters
    ----------
    wave : Wavefront
        Wavefront object with a `coordinates` method that returns (y, x)
        coordinate arrays in meters in the pupil plane
    pupil_radius : float
        Radius (in meters) of a circle circumscribing the pupil.
    """

    if accel_math._USE_NUMEXPR:
        rho = accel_math.ne.evaluate("sqrt(x**2+y**2)/pupil_radius")
        theta = accel_math.ne.evaluate("arctan2(y / pupil_radius, x / pupil_radius)")
    else:
        rho = np.sqrt(x ** 2 + y ** 2) / pupil_radius
        theta = np.arctan2(y / pupil_radius, x / pupil_radius)
    return rho, theta

class PowerSpectrumWFE_old(WavefrontError):
    """
    Power spectrum PSD WFE class from characterizing and modeling 
    optical surface power spectrum and applying optical noise.
    
    References:
    Males, Jared. MagAO-X Preliminary-Design Review, 
        Section 5.1: Optics Specifications, Eqn 1
        https://magao-x.org/docs/handbook/appendices/pdr/
    Lumbres, et al. In Prep.

    Parameters
    ----------
    name : string
        name of the optic
    psd_parameters: list of array with various astropy quantities
        Specifies the various PSD parameters with appropriate units, 
        ordered according to which PSD model set. Follows the order:
        [alpha, beta, outer_scale, inner_scale, surf_roughness]
        Comments on the PSD parameter units:
            alpha: unitless
            beta: variance (surface or OPD) units * spatial units ** (2-alpha)
                Note: spatial units preferably meters to match POPPY standards
            outer_scale: spatial units (meters preferance)
            inner_scale: unitless
            surf_roughness: variance (surface or OPD) units * spatial units**2.
    psd_weight: iterable list of floats
        Specifies the weight muliplier to set onto each model PSD
    seed : integer
        Seed for the random phase screen generator
    apply_reflection: boolean
        Applies 2x scale for the OPD as needed for reflection.
        Default to False. Set to True if the PSD model only accounts for surface.
    screen_size: integer
        Sets how large the PSD matrix will be calculated.
        If None passed in, then code will default size to 4x wavefront's side.
    wfe: astropy quantity
        Optional. Use this to force the wfe RMS.
        If a value is passed in, be aware if it is surface or opd.
        If None passed, then the wfe RMS produced is what shows up in PSD calculation.
    incident_angle: astropy quantity
        Adjusts the wavefront error based on incident angle value.
        Can be passed in degrees or radians, but must have a unit.
        Default is 0 degrees (paraxial).
    """

    @utils.quantity_input(wfe=u.nm, radius=u.meter)
    def __init__(self, name='Model PSD WFE', psd_parameters=None, psd_weight=None, seed=None, 
                 apply_reflection=False, screen_size=None, wfe=None, incident_angle=0*u.deg, 
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.psd_parameters = psd_parameters
        self.seed = seed
        self.apply_reflection = apply_reflection
        self.screen_size = screen_size
        
        # check incident angle units
        if hasattr(incident_angle, 'unit'):
            if incident_angle.unit == u.deg:
                assert incident_angle.value < 90, "Incident angle must be less than 90 degrees."
            elif incident_angle.unit == u.rad:
                assert incident_angle.value < (np.pi/2), "Incident angle must be less than pi/2 radians."
            else:
                raise Exception('Incident angle units must be either degrees or radians.')
            self.incident_angle = incident_angle
        else:
            raise Exception('Incident angle missing units, must be either degrees or radians.')
        
        # check wfe
        if wfe is not None:
            if hasattr(wfe, 'unit'):
                self.wfe = wfe.to(u.m)
            else:
                self.wfe = wfe * u.m # default assume meters
        else:
            self.wfe = wfe
        
        if psd_weight is None:
            self.psd_weight = np.ones((len(psd_parameters))) # default to equal weights
        else:
            self.psd_weight = psd_weight
        

    @_check_wavefront_arg
    def get_opd(self, wave):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        """
        
        # check that screen size is at least larger than wavefront size
        wave_size = wave.shape[0]
        if wave.ispadded is True: # get true wave size if padded to oversample.
            wave_size = int(wave_size/wave.oversample)
        
        # check that screen size exists
        if self.screen_size is None:
            self.screen_size = wave.shape[0]
            
            if wave.ispadded is False: # sometimes the wave is not padded.
                self.screen_size = self.screen_size * 4 # default 4x, open for discussion
        
        elif self.screen_size < wave_size:
            raise Exception('PSD screen size smaller than wavefront size, recommend at least 2x larger')
        
        # get pixelscale to calculate spatial frequency spacing
        pixelscale = wave.pixelscale * u.pixel # default setting is m/pix, force to meter only
        dk = 1/(self.screen_size * pixelscale) # units: 1/m
        
        # build spatial frequency map
        cen = int(self.screen_size/2)
        maskY, maskX = np.mgrid[-cen:cen, -cen:cen]
        ky = maskY*dk.value
        kx = maskX*dk.value
        k_map = np.sqrt(kx**2 + ky**2) # unitless for the math, but actually 1/m
        
        # calculate the PSD
        psd = np.zeros_like(k_map) # initialize the total PSD matrix
        for n in range(0, len(self.psd_weight)):
            # loop-internal localized PSD variables
            alpha = self.psd_parameters[n][0]
            beta = self.psd_parameters[n][1]
            outer_scale = self.psd_parameters[n][2]
            inner_scale = self.psd_parameters[n][3]
            surf_roughness = self.psd_parameters[n][4]
            
            # unit check
            psd_units = beta.unit / ((dk.unit**2)**(alpha/2))
            assert surf_roughness.unit == psd_units, "PSD parameter units are not consistent, please re-evaluate parameters."
            surf_unit = (psd_units*(dk.unit**2))**(0.5)
            
            # initialize loop-internal PSD matrix
            psd_local = np.zeros_like(psd)
            
            # Calculate the PSD equation denominator based on outer_scale presence
            if outer_scale.value == 0: # skip out or else PSD explodes
                # temporary overwrite of k_map at k=0 to stop div/0 problem
                k_map[cen][cen] = 1*dk.value
                # calculate PSD as normal
                psd_denom = (k_map**2)**(alpha/2)
                # calculate the immediate PSD value
                psd_interm = (beta.value*np.exp(-((k_map*inner_scale)**2))/psd_denom)
                # overwrite PSD at k=0 to be 0 instead of the original infinity
                psd_interm[cen][cen] = 0
                # return k_map to original state
                k_map[cen][cen] = 0
            else:
                psd_denom = ((outer_scale.value**(-2)) + (k_map**2))**(alpha/2) # unitless currently
                psd_interm = (beta.value*np.exp(-((k_map*inner_scale)**2))/psd_denom)
            
            # apply surface roughness
            psd_interm = psd_interm + surf_roughness.value
            
            # apply as the sum with the weight of the PSD model
            psd = psd + (self.psd_weight[n] * psd_interm) # this should all be m2 [surf_unit]2, but stay unitless for all calculations
        
        # set the random noise
        psd_random = np.random.RandomState()
        psd_random.seed(self.seed)
        rndm_noise = np.fft.fftshift(np.fft.fft2(psd_random.normal(size=(self.screen_size, self.screen_size))))
        
        psd_scaled = (np.sqrt(psd/(pixelscale.value**2)) * rndm_noise)
        opd = ((np.fft.ifft2(np.fft.ifftshift(psd_scaled)).real*surf_unit).to(u.m)).value 
        
        # Set wfe value based on the active region of beam
        if self.wfe is not None:
            beam_diam = pixelscale * wave_size # has units, needed for circ
            circ = CircularAperture(name='beam diameter', radius=beam_diam/2)
            ap = circ.get_transmission(wave)
            active_ap = opd[ap==True]
            rms = np.sqrt(np.mean(np.square(active_ap)))
            opd = opd * (self.wfe.to(u.m).value/rms) # appropriately scales entire OPD
            
        # apply the angle adjustment for rms
        if self.incident_angle.value != 0:
            opd = opd / np.cos(self.incident_angle).value
        
        # Set reflection OPD
        if self.apply_reflection == True:
            opd = opd*2
            
        # Resize PSD screen to shape of wavefront
        if self.screen_size > wave.shape[0]: # crop to wave shape if needed
            opd = utils.pad_or_crop_to_shape(array=opd, target_shape=wave.shape)
        
        #opd_rms = np.sqrt(np.mean(np.square(opd)))
        #rms_nm = (opd_rms*u.m).to(u.nm)
        #print('{0} rms = {1:.3f}'.format(self.name, rms_nm))
        self.opd = opd
        return self.opd
    
    
    
    
