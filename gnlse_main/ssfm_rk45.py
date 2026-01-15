#functions used in split step method with rk45
import numpy as np
from copy import deepcopy
import gnlse_main.gnlse
import tqdm
def ssfm_step(solver,Aw0,HRw,center_frequency):
    if not hasattr(solver.dispersion_model,"dopant_concentration"):
        raise RuntimeError("Dispersion model incompatible with solver - cannot model gain")
    #executes a single ssfm_rk45 step, fixed step size
    #get basic parameters
    Aw = deepcopy(Aw0)
    
    dz = solver.fiber_length / solver.z_saves
    #requires dispersion with gain - dispersion must have been updated with the most recent amplitude
    #apply dispersion (with gain) in frequency domain
    Aw = Aw * np.exp(solver.D * (dz/2))
    At = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Aw)))
    A0 = deepcopy(At)
    #evaluate Tr
    dt = solver.t[1] - solver.t[0]
    dF = 1/(solver.N * dt)
    Tr = solver.fr * np.gradient(np.imag(np.fft.fftshift(HRw)),dF)
    Tr = Tr[0]

    #evaluate nonlinear part
    k1 = dz * NonlinearRK(At,center_frequency,solver.gamma,Tr,dt)
    k2 = dz * NonlinearRK(At+(k1/2),center_frequency,solver.gamma,Tr,dt)
    k3 = dz * NonlinearRK(At+(k2/2),center_frequency,solver.gamma,Tr,dt)
    k4 = dz * NonlinearRK(At+(k3),center_frequency,solver.gamma,Tr,dt)

    Aw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(At + (1/6)*(k1+(2*k2)+(2*k3)+k4))*dt))

    #Energy normalisation
    E0 = sum(abs(At)**2) * dt
    E1 = sum(abs(Aw)**2) * dF
    Aw = Aw * np.sqrt(E0/E1)

    #next dispersion part
    Aw = Aw * np.exp(solver.D * (dz/2))
    return Aw

def NonlinearRK(At,center_frequency,gamma,Tr,dt):
    N1 = At * np.conjugate(At) * At
    N2 = (1j/center_frequency)*np.gradient(N1,dt)
    N3 = Tr * At * np.gradient(At * np.conjugate(At),dt)
    return 1j * gamma * (N1 + N2 - N3)

def solve(solver,Aw0):
    dz = solver.fiber_length / solver.z_saves
    dt = solver.t[1] - solver.t[0]
    dF = 1/(solver.N * dt)
    Aw = deepcopy(Aw0)
    HRw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(solver.RT))) * dt

    aw_grid = [Aw]
    at_grid = [np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Aw))/(dt))]
    power_curve_s = [sum((abs(Aw)**2)*dF*solver.dispersion_model.repetition_rate*1e-12)]
    n2 = []
    progress_bar = tqdm.tqdm(total=solver.fiber_length, unit='m')
    for i in range(solver.z_saves):
        progress_bar.n = round((i+1)*dz,3)
        progress_bar.update(0)

        solver.dispersion_model.AW = Aw
        solver.D = np.fft.fftshift(solver.dispersion_model.D(solver.V))
        
        Aw = ssfm_step(solver,Aw,HRw,solver.dispersion_model.w0*np.pi*2)

        aw_grid.append(deepcopy(Aw)) #tracks spectral shape evolution
        at_grid.append(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Aw))/(dt)))
        #log change in average power
        power_curve_s.append(sum((abs(Aw)**2)*dF*solver.dispersion_model.repetition_rate*1e-12))
        n2.append(solver.dispersion_model.N2Total)
    
    progress_bar.close()
    return gnlse_main.gnlse.Solution(solver.t,solver.Omega,solver.w_0,np.linspace(0,solver.fiber_length,solver.z_saves),at_grid,aw_grid)

