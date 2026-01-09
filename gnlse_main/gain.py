"""
new file containing gain class and functions
part of rewrite to fix gain implementation
plan to implement:

seperate gain from dispersion operator
modify rhs function to linearly add gain component to rv
modify main solving method to no longer use solve_ivp and instead perform rk45 step by step such that
    step size can be passed to rhs for correct calculation of gain"""
import numpy as np
from gnlse_main.common import hplanck,c

class Gain:
    AW = None #amplitude spectrum
    n2 = None #population inversion fraction
    NT = None #total density of dopant, m^-3
    emission_cross_section = None
    absorption_cross_section = None
    lifetime = None
    fiber_area = None
    pump_power = None
    pump_wavelength = None #in nm
    repetition_rate = None
    overlap_signal = None
    overlap_pump = None
    frequencies = None
    wavelengths = None
    def __init__(self,emission_cs,absorption_cs,lifetime,fiber_area,rep_rate,overlap_s,overlap_p,pump_power,NT, pump_wavelength = 975):
        #defines time-independent gain parameters
        self.emission_cross_section = emission_cs
        self.absorption_cross_section = absorption_cs
        self.lifetime = lifetime
        self.fiber_area = fiber_area
        self.repetition_rate = rep_rate
        self.overlap_signal = overlap_s
        self.overlap_pump = overlap_p
        self.pump_power = pump_power
        self.pump_wavelength = pump_wavelength
        self.NT = NT
    
    def N2(self):
        if type(self.AW) == bool:
            raise TypeError("Amplitude spectrum was not defined, cannot compute population inversion. D.AW must not be None.")
        Ip = self.pump_power/self.fiber_area
        Isw = np.square(np.abs(self.AW))
        #first we find the pulse energy - integrate Isw over v
        At = np.fft.ifft(self.AW)
        pulse_energy = np.trapezoid(np.square(np.abs(At)),dx=(self.dt*1e-12))
        Ps = pulse_energy * self.repetition_rate #average signal power
        Is = Ps/self.fiber_area
        #set central frequency as average weighted by intensity
        central_frequency = np.average(self.frequencies,weights=Isw)
        #get as wavelength in nm
        central_wavelength = (c/central_frequency)*1e12
        #now we find the cross section at central_frequency
        pump_frequency = c/(self.pump_wavelength *1e-12)
        em_cross_section_signal = self.emission_cross_section[r"cross section(m^2)"][(np.abs(self.emission_cross_section["wavelength(nm)"]-central_wavelength)).argmin()] #cross section @ closest wavelength
        abs_cross_section_signal = self.absorption_cross_section[r"cross section(m^2)"][(np.abs(self.absorption_cross_section["wavelength(nm)"]-central_wavelength)).argmin()]
        em_cross_section_pump = self.emission_cross_section[r"cross section(m^2)"][(np.abs(self.emission_cross_section["wavelength(nm)"]-self.pump_wavelength)).argmin()] 
        abs_cross_section_pump = self.absorption_cross_section[r"cross section(m^2)"][(np.abs(self.absorption_cross_section["wavelength(nm)"]-self.pump_wavelength)).argmin()]

        #Now compute R and W values
        R12 = (abs_cross_section_pump*Ip)/(hplanck*pump_frequency)
        R21 = (em_cross_section_pump*Ip)/(hplanck*pump_frequency)
        W12 = (abs_cross_section_signal*Is)/(hplanck*central_frequency)
        W21 = (em_cross_section_signal*Is)/(hplanck*central_frequency)

        #finally compute N2
        numerator = R12 + W12
        denominator = R12 + W12 + R21 + W21 + (1/self.lifetime)
        self.n2 = numerator/denominator

    def SetFrequency(self,v):
        self.v = v
        self.wavelengths = (c/v)*1e12
        self.sigma_a = np.array([self.emission_cross_section[r"cross section(m^2)"][(np.abs(self.emission_cross_section["wavelength(nm)"] - wavelength)).argmin()] for wavelength in self.wavelengths])
        self.sigma_e = np.array([self.emission_cross_section[r"cross section(m^2)"][(np.abs(self.emission_cross_section["wavelength(nm)"] - wavelength)).argmin()] for wavelength in self.wavelengths])
    
    def CalculateGain(self):
        lhs = np.multiply(self.sigma_e, self.n2)
        rhs = np.multiply(self.sigma_a, np.subtract(1, self.n2))
        self.gain = self.NT * np.subtract(lhs,rhs)