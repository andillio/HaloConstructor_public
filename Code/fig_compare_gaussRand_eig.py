# pylint: disable=C,W
# produces a fig comparing different halo surface density constructions
import HaloUtils as hu
import mathUtils as mu
import gridUtils as gu
import plotUtils as pu
import cupy as np
import numpy as np_
import time 
import astroUtils as au
from scipy import signal
import sysUtils as su

np_.random.seed(1)
simName = "test_Res256"

### density profile parameters
# NFW parameters
meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
Rs = 2. # scale radius in kpc
Rhalf = 0.3 # half light radius
Rvir = Rhalf / .015
Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
con = Rvir / Rs # concentration parameter
rho0 = Mvir/4/np.pi/Rs**3 /(np_.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3

m22 = 5. # mass of uldm particle
hbar_ = au.h_tilde(m22) # hbar / m
rc = float(1.6 * (m22)**(-1) * (Mvir/1.0e9)**(-1/3.))# core radius in kpc

rmin = 0.05 
rmax = Rvir*1.16
dr = rmin

### halo parameters
N = 256 # resolution
L = Rvir*2 / np_.sqrt(3)
x = gu.grid(N,L)

## lensing stuff
dx = L/N
sigma_d = np_.sqrt(Mvir*au.G / Rvir)
lambda_chi = hbar_ / sigma_d
Sigma_c = 1. 


def constructSmoothHaloDensity(r_tr, rho_Total_tr):
	R, Theta, Phi = gu.sphrGrid(N,L)

	rho_smooth = np_.interp(R,r_tr, rho_Total_tr) # in the box
	
	return R, rho_smooth


def constructHalo(r, beta, E, zi):
	X, Y, Z = gu.grid((N, N, N), L = L, gpu= True)
	R, Theta, Phi = gu.cart2sphr(X,Y,Z, gpu= True)

	rmin = np.min(r)
	rmax = np.max(r)

	psi = np.zeros(R.shape) + 0j

	time0 = time.time()
	total = int(np.sum(E[:,2]*2 + 1))
	done = 0
	for i in range(len(E)-1,-1,-1):
		l = int(E[i,2])
		n = E[i,1]
		anlm = np.sqrt(beta[i])

		zi_ = np.interp(R, r, zi[i])
		zi_[R < rmin] = np.max(zi[i])
		zi_[R > rmax] = np.min(zi[i])

		for m in range(-l,l+1):
			Y = mu.sph_harm(m,l,Theta,Phi, gpu=True)
			phi = np.random.uniform(0,2*np.pi)
			norm = np.sum(np.abs(zi_*Y)**2)*dx**3
			psi += anlm*zi_*Y*np.exp(1j*phi) / np.sqrt(norm)

			done += 1
			su.PrintTimeUpdate(done,total,time0)
	return R, psi


if __name__ == "__main__":
	### get potential
	time0 = time.time()
	Phi_r_tr, rho_Total_tr, r_tr = hu.get_potential_and_density(rmin, rmax, Rs, rho0, m22, rc,
	 dr = dr, maxFactor = 10, fullReturns=True)

	R, rho_smooth = constructSmoothHaloDensity(r_tr, rho_Total_tr)
	Mtot = rho_smooth.sum()*dx**3

	rho_simple = Mvir / Rvir**3
	m_pbh = rho_simple*lambda_chi**3
	Sigma_c = 1. 

	R_sigma, Phi_sigma = gu.cirGrid(N,L)
	Sigma = rho_smooth.sum(axis = 2)*dx
	Var = np_.sqrt(np_.pi)*lambda_chi*np_.sum(rho_smooth**2, axis = 2)*dx

	white_noise = np_.random.normal(0, np_.sqrt(Var))
	# white_noise = np_.random.exponential(np_.sqrt(Var))
	kernel = np_.exp(-.5*(x/lambda_chi)**2)
	kernel = np_.einsum("i,j->ij",kernel, kernel)
	kernel /= np_.sqrt(np_.sum(kernel**2))
	dSigma = signal.fftconvolve(white_noise, kernel, mode='same')
	# dSigma = mu.fftConvolve(white_noise, kernel).real
	Sigma_gr = Sigma + dSigma
	# Sigma_gr = mu.fftConvolve(white_noise, kernel).real

	Sigma_pbh = m_pbh*np_.random.poisson(Sigma / m_pbh * dx**2) / dx**2

	r_np, rho_new, rho_target, beta_new, E_new, Psi_new = \
		hu.constructProfile_axionyx(meandens, Rs, Rhalf, m22, rmin,
		fullReturns= True)
	r, beta, E, Psi = gu.cpu2gpu(r_np, beta_new, E_new, Psi_new)
	R, psi_new = constructHalo(r, beta, E, Psi)
	psi_new = gu.cpuThis(psi_new)
	Sigma_e = np_.sum(np_.abs(psi_new)**2, axis = 2)*dx
	# Sigma_e = np_.abs(psi_new[N//2,:,:])**2
	print(Sigma.sum()*dx*dx / Mtot)
	print(Sigma_e.sum()*dx*dx / Mtot)
	print(Sigma_gr.sum()*dx*dx / Mtot)
	print(Sigma_pbh.sum()*dx*dx / Mtot)

	np_.save(su.getDataDir(simName) + "Psi.npy", Psi)
	np_.save(su.getDataDir(simName) + "beta.npy", beta)
	np_.save(su.getDataDir(simName) + "E.npy", E)
	np_.save(su.getDataDir(simName) + "psi_new.npy", psi_new)
	np_.save(su.getDataDir(simName) + "rho_target.npy", rho_target)
	np_.save(su.getDataDir(simName) + "rho_smooth.npy", rho_smooth)
	np_.save(su.getDataDir(simName) + "r.npy", r_np)
	np_.save(su.getDataDir(simName) + "R.npy", R)
	np_.save(su.getDataDir(simName) + "Sigma_gr.npy", Sigma_gr)
	np_.save(su.getDataDir(simName) + "Sigma_pbh.npy", Sigma_pbh)

	su.PrintCompletedTime(time0)

	fo = pu.FigObj(3)
	fo.AddDens2d([-L/2.,L/2],Sigma_e, log = True)
	fo.SetTitle(r"Eigenvalues")
	fo.AddDens2d([-L/2.,L/2],Sigma_gr, log = True)
	fo.RemoveYLabels()
	fo.SetTitle(r"Gauss random")
	fo.AddDens2d([-L/2.,L/2], Sigma_pbh, log = True)
	fo.RemoveYLabels()
	fo.SetTitle(r"MACHOs")
	fo.RemoveWhiteSpace()
	fo.save("compareHalos")
	fo.show()

