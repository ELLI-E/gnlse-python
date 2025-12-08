import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gnlse_main

if __name__ == '__main__':
    setup = gnlse_main.gnlse.GNLSESetup()

    # Numerical parameters
    # number of grid time points
    setup.resolution = 2**13
    # time window [ps]
    setup.time_window = 12.5
    # number of distance points to save
    setup.z_saves = 200
    # relative tolerance for ode solver
    setup.rtol = 1e-6
    # absoulte tolerance for ode solver
    setup.atol = 1e-6

    # Physical parameters
    # Central wavelength [nm]
    setup.wavelength = 1030
    # Nonlinear coefficient [1/W/m]
    setup.nonlinearity = 0.0
    # Dispersion: derivatives of propagation constant at central wavelength
    # n derivatives of betas are in [ps^n/m]
    betas = np.array([-11.830e-3])
    # Input pulse: pulse duration [ps]
    tFWHM = 0.050
    # for dispersive length calculation
    t0 = tFWHM / 2 / np.log(1 + np.sqrt(2))

    # 3rd order soliton conditions
    ###########################################################################
    # Fiber length [m]
    setup.fiber_length = .5
    # Type of pulse:  hyperbolic secant
    setup.pulse_model = gnlse_main.GaussianEnvelope(331400, 0.050)
    # Loss coefficient [dB/m]
    loss = 0
    # Type of dyspersion operator: build from Taylor expansion
    #Set parameters necessary for gain modelling
    gain_medium_radius = 3e-6
    fiber_area = np.pi * (gain_medium_radius**2)
    dopant_concentration = (5e25)
    emission = pd.read_csv(r"data\emissionCS.csv") #get absorption and emission cross sections from csv
    absorption = pd.read_csv(r"data\absorptionCS.csv")
    lifetime = 1e-3
    repetition_rate = 1e8 #100MHz
    pump_power = 9 #pump power in watts

    setup.self_steepening = True
    setup.active_fiber=True
    setup.dispersion_model = gnlse_main.DispersionFiberFromTaylorWithGain(loss,betas,fiber_area,dopant_concentration,emission,absorption,lifetime,pump_power,repetition_rate=repetition_rate)
    solver = gnlse_main.gnlse.GNLSE(setup)
    N2 = solver.dispersion_model.N2()
    print(N2)
    gain = solver.dispersion_model.CalculateGain()
    print(gain)
    plt.plot(solver.dispersion_model.wavelengths,gain)
    #test in dispersion operator
    solver.dispersion_model.D(solver.V)
    plt.xlim(800,1300)
    plt.show()