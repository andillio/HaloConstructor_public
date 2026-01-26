"""
This file finds the eigenvectors for the specified density
"""
# pylint: disable=C,W 
import numpy as np
import gridUtils as gu
import mathUtils as mu
import plotUtils as pu 
import astroUtils as au


# configuration parameters
# simulation parameters
L = 100. # box length in kpc
N = 1024 # resolution
# derived paramters
dx = L / N
r = np.linspace(dx / 2., L/2.*np.sqrt(3), N)
dr = L/2.*np.sqrt(3) / N 

# NFW parameters
meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
Rs = 2. # scale radius in kpc
Rhalf = 0.3 # half light radius
# dervived parameters
### virial radius: using the relationship between the virial and half-light radius from https://arxiv.org/abs/1212.2980
Rvir = Rhalf/0.015 
Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
con = Rvir / Rs # concentration parameter
rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3


m22 = 5. # mass of uldm particle
rc = float(1.6 * (m22)**(-1) * (Mvir/1.0e9)**(-1/3.))# core radius in kpc
hbar_ = au.h_tilde(m22)


def HandleProfiles():
	rho_NFW = au.NFW(r, Rs, rho0)
	rho_core = au.Core(r, m22, rc)
	rho_Total = au.CoredNFW(r, Rs, rho0, m22, rc)

	np.save('r_NFW.npy',r)
	np.save('rho_NFW.npy',rho_NFW)
	fo.AddPlot(r, rho_NFW, label = r"NFW", alpha = .5, ls = '--')
	fo.AddLine(r, rho_core, label = r"core", alpha = .5, ls = '--')
	fo.AddLine(r, rho_Total, label = r"cored NFW")
	fo.SetLogLog(r, rho_NFW)

	fo.SetLabels(r'$r \,[\mathrm{kpc}]$', r'$\rho(r) \, [\mathrm{M_\odot / kpc^3}]$')
	fo.legend()

	return rho_Total


def HandleCumMass():
	initial = rho_Total[0]*4*np.pi*r[0]**3 /3.
	enclMass = mu.cumInt(rho_Total*4*np.pi*r**2, r, initial=initial) # mass enclosed using traprule

	Mtot = np.sum(4*np.pi*r**2 * rho_Total)*dr # mass enclosed using riemann sum

	fo.AddPlot(r, enclMass, label = r'$M_{enc}(<r)$')
	fo.AddHorLine(Mtot, label = r"total mass")
	fo.IncreaseYLim()
	fo.SetLabels(r'$r \,[\mathrm{kpc}]$', r'$M \, [\mathrm{M_\odot}]$')
	fo.legend()

	return Mtot


def HandlePotential():
	Phi_r = au.radialPotential(r, rho_Total)


	Phi_th = -4*au.G*rho0*Rs**3 / r * np.log(1 + r/Rs)
	fo.AddPlot(r, Phi_th, label = "NFW")
	fo.AddLine(r, Phi_r - au.G*Mtot/r[-1], label = r"estimate")


	fo.legend()

	fo.SetLabels(r'$r \,[\mathrm{kpc}]$', r'$\Phi \, [\mathrm{kpc^2 / Myr^2}]$')

	return Phi_r


def HandleEigenValues(n,l):
	H_ = au.Hamiltonian_radial(r, Phi_r, l = l, hbar_ = hbar_, dx = dx)
	E, u = mu.Eig(H_)

	psi = u[:,n]/r
	y = np.abs(psi)**2
	fo.AddPlot(r, y)
	fo.SetLogLog(r, y)


if __name__ == "__main__":
	fo = pu.FigObj(2,2)

	# handle profiles
	rho_Total = HandleProfiles()
	# handle enclosed mass
	Mtot = HandleCumMass()
	# handle potential
	Phi_r = HandlePotential()
	# handle 
	HandleEigenValues(0,0)

	fo = pu.FigObj()
	# handle profiles
	rho_Total = HandleProfiles()
	fo.SetXLim(xHigh=10.)
	fo.SetYLim(yLow=1e4)
	fo.SetTitle(r"density profiles")
	fo.Save("densityProfiles")

	fo.show()