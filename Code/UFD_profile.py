# pylint: disable=C,W
import statUtils as st
import HaloUtils as hu
import mathUtils as mu
import gridUtils as gu
import plotUtils as pu
import scipy as sp
import cupy as cp
import astroUtils as au
import numpy as np
import time 
import sysUtils as su

lazy_method = False
fileName = 'Segue1_256_1e2m22_axionyx'
Rhalf = 24e-3 # half light radius, [kpc]
Rvir = Rhalf / .015 # virial radius, 1.6 kpc
Rs = Rhalf / .3 * 2 # scale radius in kpc
sigma_v = 4 * au.kms2kpcMyr # half light velocity in [kpc/Myr]
m22 = 1e2  # mass of uldm particle
hbar_ = au.h_tilde(m22)


# simulation parameters
L = 10. # box length in kpc
N = 256 # resolution
Nhalo = 256
# derived paramters
dx = L / N
r = np.linspace(dx / 2., L/2.*np.sqrt(3), N)

gpu = False
betaCutOff = 1e-12


def HandleProfiles():
	rho_NFW = au.NFW(r, Rs)
	M_encl = au.M_NFW(Rhalf, Rs)
	rho0 = sigma_v**2 * Rhalf / M_encl / au.G
	Mvir = au.M_NFW(Rvir, Rs, rho0)

	rc = float(1.6 * (m22)**(-1) * (Mvir/1.0e9)**(-1/3.))# core radius in kpc
	print(rc, Rhalf, au.CoreRadius(m22, Mvir, zeta = 350))

	rho_NFW = au.NFW(r, Rs, rho0)
	np.save('../Data/' + fileName + '/rho_NFW.npy', rho_NFW)
	np.save('../Data/' + fileName + '/r.npy', r)

	rho_core = au.Core(r, m22, au.CoreRadius(m22, Mvir, zeta = 1))
	rho_Total = au.CoredNFW(r, Rs, rho0, m22, Mvir = Mvir)
	np.save('../Data/' + fileName + '/rho_Total.npy', rho_Total)


	return rho_Total, rho0, Mvir

def HandlePotential():
	Phi_r = au.radialPotential(r, rho_Total)

	return Phi_r

def dens_out_func(beta, X):
	"""
	given weight estimators beta, and degeneracy weighted eigenfunc matrix X,
	return the corresponding density

	:beta: array-like, 
	"""
	return np.einsum("ji,j->i", X, beta)


def cost_func(beta_geuss, X, rho_target):
	rho_out = dens_out_func(beta_geuss, X)
	return np.sum( ( (rho_out - rho_target)/rho_target)**2 ) 


def DensityOutputs():
	fo = pu.FigObj()
	fo.AddPlot(r, rho_Total, label = 'target')
	fo.AddLine(r, rho_app, color = 'r', label = 'approximation')
	fo.AddVertLine(Rvir, label = 'virial radius')
	fo.SetLogLog(r, rho_Total)
	fo.SetXLabel(r'$r \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$\rho(r) \, [\mathrm{M_\odot kpc^{-3}}]$')
	fo.legend()
	fo.SaveInDataDir(fileName, 'densityProfiles')



def FitEigenModes():
	M_max = Mvir
	l = E[:,2]
	X = np.einsum("i,ij->ij", 2*l + 1, np.abs(Psi)**2)

	beta = st.linearFit_beta_pos(X,rho_Total / M_max)
	rho_app = np.einsum("ji,j->i", X, beta*M_max)
	rho_target = rho_Total / M_max

	beta_geuss = np.ones(len(beta)) / len(beta)
	high = 1e3*3.
	low = betaCutOff / 100.

	if lazy_method:

		beta_usefull = beta[beta>np.max(beta)*betaCutOff]
		X_usefull = X[beta>np.max(beta)*betaCutOff,:]
		E_usefull = E[beta>np.max(beta)*betaCutOff]
		Psi_usefull = Psi[beta>np.max(beta)*betaCutOff,:]
		rho_app_lazy = np.einsum("ji,j->i", X_usefull, beta_usefull*M_max)
		return rho_app_lazy, beta_usefull, E_usefull, Psi_usefull

	beta_bounds = sp.optimize.Bounds(beta_geuss*low, beta_geuss*high, keep_feasible=False)
	print("starting to optimize")
	aln_raw= sp.optimize.minimize(cost_func, beta,
	 args=(X, rho_target), bounds = beta_bounds)
	aln_new=aln_raw.x
	aln_raw= sp.optimize.minimize(cost_func, aln_new,
	 args=(X, rho_target), bounds = beta_bounds)
	beta=aln_raw.x
	print("done optimizing")

	beta_usefull = beta[beta>np.max(beta)*betaCutOff]
	X_usefull = X[beta>np.max(beta)*betaCutOff,:]
	E_usefull = E[beta>np.max(beta)*betaCutOff]
	Psi_usefull = Psi[beta>np.max(beta)*betaCutOff,:]
	rho_app = np.einsum("ji,j->i", X_usefull, beta_usefull*M_max)

	return rho_app, beta_usefull, E_usefull, Psi_usefull



def PlotPsi(psi):
	rho = np.abs(psi)**2
	Sigma = np.sum(rho, axis = 1)

	x = [-L/2., L/2.]

	fo = pu.FigObj(2)
	fo.AddDens2d(x,rho[N//2], log = True)
	fo.SetTitle(r"Slice")
	fo.AddDens2d(x,Sigma, log = True)
	fo.RemoveYLabels()
	fo.SetTitle(r"Projection")
	fo.RemoveWhiteSpace()
	fo.SaveInDataDir(fileName, 'densitySliceProj')
	fo.show()


if __name__ == "__main__":
	# get profile
	su.makeDataDir(fileName)

	rho_Total, rho0, Mvir = HandleProfiles()
	dataDict = {}
	dataDict['rho0'] = rho0
	dataDict['Rs'] = Rs
	dataDict['Rhalf'] = Rhalf
	dataDict['m22'] = m22
	dataDict['sigma'] = sigma_v
	dataDict['L'] = L
	dataDict['N'] = N
	su.Dict2Toml(dataDict, '../Data/' + fileName + '/params.toml')

	# Phi_r = HandlePotential()
	# for Segue1 we can just use nfw theorhetical potnetials
	phi = au.V_NFW(r, Rs, rho0)

	Evir = -.5*Mvir*au.G/Rvir/hbar_ 
	factor = 0
	Ecutoff = Evir * factor
	E, Psi = hu.Eigenfuncs(r, Ecutoff, phi = phi, hbar_= hbar_)

	rho_app, beta_usefull, E_usefull, Psi_usefull = FitEigenModes()
	DensityOutputs()

	if gpu:
		r, beta_usefull, E_usefull, Psi_usefull = su.cpu2gpu(r, beta_usefull, E_usefull, Psi_usefull)
	psi = hu.constructHalo(Nhalo, Rvir*2., r, beta_usefull, E_usefull, Psi_usefull, gpu = gpu)
	
	if gpu:
		cp.save("../Data/" + fileName + "/psi.npy", psi)
		psi = su.cpuThis(psi)
	np.save("../Data/" + fileName + "/psi.npy", psi)


	PlotPsi(psi)