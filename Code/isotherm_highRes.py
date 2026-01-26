# pylint: disable=C,W
import astroUtils as au
import statUtils as st
import time as time
import plotUtils as pu
import HaloUtils as hu
import sysUtils as su
import numpy as np 
import cupy as cp
import scipy as sp

fileName = 'isotherm_10m22'

R_ein = 2.062
c_L = 5.
c_lam = 2.

m22 = 10.
hbar_ = au.h_tilde(m22)
sigma = 100 * au.kms2kpcMyr
lam = hbar_ / sigma

dx_min = lam / c_lam
L = R_ein * 2 * c_L
Rmax = L
N_min = L / dx_min

print("min L:", L)
print("N_min:", N_min)

N = 1024
Nhalo = 256
dr = Rmax / N 
r = np.arange(.5, N, 1)*dr

betaCutOff = 1e-15

lazy_method = True
gpu = False
construct = False

def HandleProfiles():
	rho_iso = sigma**2 / 2. / np.pi / au.G / r
	Mvir = 2 * sigma**2 * Rmax / au.G

	np.save('../Data/' + fileName + '/r.npy', r)
	np.save('../Data/' + fileName + '/rho_iso.npy', rho_iso)

	return rho_iso, Mvir

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
def FitEigenModes(rho_Total):
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

def DensityOutputs(rho_Total, rho_app):
	fo = pu.FigObj()
	fo.AddPlot(r, rho_Total, label = 'target')
	fo.AddLine(r, rho_app, color = 'r', label = 'approximation')
	fo.AddVertLine(R_ein, label = 'Einstein radius')
	fo.SetLogLog(r, rho_Total)
	fo.SetXLabel(r'$r \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$\rho(r) \, [\mathrm{M_\odot kpc^{-3}}]$')
	fo.legend()
	fo.SaveInDataDir(fileName, 'densityProfiles')


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
	su.makeDataDir(fileName)

	dataDict = {}
	dataDict['N'] = N
	dataDict['Nhalo'] = Nhalo
	dataDict['L'] = L
	dataDict['Rmax'] = Rmax 
	dataDict['Rein'] = R_ein
	dataDict['m22'] = m22
	dataDict['sigma'] = sigma
	su.Dict2Toml(dataDict, '../Data/' + fileName+ '/params.toml')

	rho_iso, Mvir = HandleProfiles()

	### get isothermal profile
	phi = 2*sigma**2*np.log(r/Rmax) 

	Ecutoff = 0.
	E, Psi = hu.Eigenfuncs(r, Ecutoff, phi = phi, hbar_= hbar_, l_max=300)

	rho_app, beta_usefull, E_usefull, Psi_usefull = FitEigenModes(rho_iso)
	DensityOutputs(rho_iso, rho_app)

	if construct:
		if gpu:
			r, beta_usefull, E_usefull, Psi_usefull = su.cpu2gpu(r, beta_usefull, E_usefull, Psi_usefull)
		psi = hu.constructHalo(Nhalo, Rmax*2., r, beta_usefull, E_usefull, Psi_usefull, gpu = gpu)
		
		if gpu:
			cp.save("../Data/" + fileName + "/psi.npy", psi)
			psi = su.cpuThis(psi)
		np.save("../Data/" + fileName + "/psi.npy", psi)

		PlotPsi(psi)