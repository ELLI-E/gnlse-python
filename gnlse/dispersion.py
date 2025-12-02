"""Dispersion operator in optical fibres.

Based on delivered frequency vector and damping indexes
script calculates linear dispersion operator in
frequency domain.

"""
import numpy as np
import math
from scipy import interpolate

from gnlse.common import c


class Dispersion(object):
    """
    Attributes
    -----------
    loss : float
        Loss factor [dB/m]
    """

    def __init__(self, loss):
        self.loss = loss

    def D(V):
        """Calculate linear dispersion operator
        for given frequency grid created during simulation

        Parameters
        ----------
        V : ndarray, (N)
            Frequency vector

        Returns
        -------
        ndarray, (N)
            Linear dispersion operator in frequency domain
        """

        raise NotImplementedError('Dispersion not implemented')

    def calc_loss(self):
        """Calculate damping
        for given frequency grid created during simulation
        """
        self.alpha = np.log(10**(self.loss / 10))


class DispersionFiberFromTaylor(Dispersion):
    """Calculates the dispersion in frequency domain

    Attributes
     ----------
    loss : float
        Loss factor [dB/m]
    betas : ndarray (N)
        Derivatives of constant propagations at pump wavelength
        [ps^2/m, ..., ps^n/m]
    """

    def __init__(self, loss, betas):
        self.loss = loss
        self.betas = betas

    def D(self, V):
        # Damping
        self.calc_loss()
        # Taylor series for subsequent derivatives
        # of constant propagation
        B = sum(beta / math.factorial(i + 2) * V**(i + 2)
                for i, beta in enumerate(self.betas))
        L = 1j * B - self.alpha / 2
        return L

class DispersionFiberFromTaylorWithGain(Dispersion):
    #under development - not yet functional
    """Calculates the dispersion in frequency domain with the gain operator

    Attributes
     ----------
    loss : float
        Loss factor [dB/m]
    betas : ndarray (N)
        Derivatives of constant propagations at pump wavelength
        [ps^2/m, ..., ps^n/m]
    """

    def __init__(self, loss, betas,fiber_area,dopant_concentration,emission_cross_section,absorption_cross_section,pump_power=None,overlap_pump=1,overlap_signal=1):
        self.loss = loss
        self.betas = betas

        #active fiber parameters
        self.fiber_area = fiber_area
        self.dopant_concentration = dopant_concentration # if None, fiber is passive, otherwise model gain
        self.pump_power = pump_power #assumed to be constant for now
        self.emission_cross_section = emission_cross_section #need to insert the array for the wavelength dependent cross section here
        self.absorption_cross_section = absorption_cross_section 
        self.overlap_pump = overlap_pump
        self.overlap_signal = overlap_signal

    def N2(self,V):
        Ip = self.pump_power/self.fiber_area
        Is = np.mean(V**2) #basic model - intensity of signal is the average across the whole pulse


        
    def D(self, V):
        # Damping
        self.calc_loss()
        # Taylor series for subsequent derivatives
        # of constant propagation
        B = sum(beta / math.factorial(i + 2) * V**(i + 2)
                for i, beta in enumerate(self.betas))
        #calculate N2
        #N2 = getN2(amplifierParams,) use average power of the pulse?
        #compute value of gain function
        L = 1j * B - self.alpha / 2
        return L

class DispersionFiberFromInterpolation(Dispersion):
    """Calculates the propagation function in frequency domain, using
    the extrapolation method based on delivered refractive indexes
    and corresponding wavelengths. The returned value is a vector
    of dispersion operator.

    Attributes
    -----------
    loss : float
        Loss factor [dB/m]
    neff : ndarray (N)
        Effective refractive index
    lambdas : ndarray (N)
        Wavelength corresponding to refractive index
    central_wavelength : float
        Wavelength corresponding to pump wavelength in nm
    """

    def __init__(self, loss, neff, lambdas, central_wavelength):
        # Loss factor in dB/m
        self.loss = loss
        # refractive indices
        self.neff = neff
        # wavelengths for neffs in [nm]
        self.lambdas = lambdas
        # Central frequency in [1/ps = THz]
        self.w0 = (2.0 * np.pi * c) / central_wavelength

    def D(self, V):
        # Central frequency [1/ps = THz]
        omega = 2 * np.pi * c / self.lambdas
        dOmega = V[1] - V[0]
        Bet = self.neff * omega / c * 1e9  # [1/m]

        # Extrapolate betas for a frequency vector
        fun_interpolation = interpolate.interp1d(omega,
                                                 Bet,
                                                 kind='cubic',
                                                 fill_value="extrapolate")

        B = fun_interpolation(V + self.w0)
        # Propagation constant at central frequency [1/m]
        B0 = fun_interpolation(self.w0)
        # Value of propagation at a lower end of interval [1/m]
        B0plus = fun_interpolation(self.w0 + dOmega)
        # Value of propagation at a higher end of interval [1/m]
        B0minus = fun_interpolation(self.w0 - dOmega)

        # Difference quotient, approximation of
        # derivative of a function at a point [ps/m]
        B1 = (B0plus - B0minus) / (2 * dOmega)

        # Damping
        self.calc_loss()

        # Linear dispersion operator
        L = 1j * (B - (B0 + B1 * V)) - self.alpha / 2
        return L
