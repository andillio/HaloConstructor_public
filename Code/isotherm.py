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

simName = "low_m_low_res"
N = 256
Rmax = 6
R_ein = 2.062
L = 3*R_ein
dr = Rmax / N
r = np.arange(.5, N, 1)*dr
m22 = 1.
hbar_ = au.h_tilde(m22)

# velocity dispersion
sigma = 100 * au.kms2kpcMyr

### profile details
# 3e10 M_solar within 2 kpc
# half light radius is 20.4 kpc

betaCutOff = 1e-15


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


def PlotPsi(psi, tag = ''):
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
	fo.SaveInDataDir(simName, 'densitySliceProj' + tag)


if __name__ == "__main__":
	time0 = time.time()
	su.makeDataDir(simName)
	dataDict = {}
	dataDict['N'] = N
	dataDict['L'] = L
	dataDict['Rmax'] = Rmax 
	dataDict['Rein'] = R_ein
	dataDict['m22'] = m22
	dataDict['sigma'] = sigma
	su.Dict2Toml(dataDict, '../Data/' + simName + '/params.toml')

	### get isothermal profile
	phi = 2*sigma**2*np.log(r/Rmax) 
	a = -2*sigma**2 / r
	M_encl = 2 * sigma**2 * r / au.G
	M_max = 2 * sigma**2 * Rmax / au.G
	rho_iso = sigma**2 / 2. / np.pi / au.G / r
	np.save(f'../Data/{simName}/rho_target.npy', rho_iso)
	
	# rc = au.CoreRadius(m22, M_max)
	# rho_core = au.Core(r, m22, rc)
	# r_a = np.max(r[rho_core > rho_iso])

	# rho_total = rho_core
	# rho_total[r > r_a] = rho_iso[r > r_a]

	Evir = -.5*M_max*au.G/Rmax/hbar_ 
	factor = 0
	Ecutoff = Evir * factor

	### get eigenfunctions
	E, Psi = hu.Eigenfuncs(r, Ecutoff, phi = phi, hbar_= hbar_)

	### sum them up
	l = E[:,2]
	X = np.einsum("i,ij->ij", 2*l + 1, np.abs(Psi)**2)
	beta = st.linearFit_beta_pos(X,rho_iso / M_max)
	rho_app = np.einsum("ji,j->i", X, beta*M_max)
	rho_target = rho_iso / M_max

	beta_geuss = np.ones(len(beta)) / len(beta)
	high = 10. 
	low = betaCutOff / 100.

	beta_usefull = beta[beta>np.max(beta)*betaCutOff]
	X_usefull = X[beta>np.max(beta)*betaCutOff,:]
	E_usefull = E[beta>np.max(beta)*betaCutOff]
	Psi_usefull = Psi[beta>np.max(beta)*betaCutOff,:]
	rho_app_lazy = np.einsum("ji,j->i", X_usefull, beta_usefull*M_max)
	print(len(beta_usefull))
	r, beta_usefull, E_usefull, Psi_usefull = su.cpu2gpu(r, beta_usefull, E_usefull, Psi_usefull)
	psi1 = hu.constructHalo(N*2, L, r, beta_usefull, E_usefull, Psi_usefull, gpu = True)
	PlotPsi(psi1, 'lazy')
	cp.save("../Data/" + simName + "/psi_lazy.npy", psi1)
	su.PrintCompletedTime(time0, "lazy scheme")
	time0 = time.time()

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
	r, beta_usefull, E_usefull, Psi_usefull = su.cpu2gpu(r, beta_usefull, E_usefull, Psi_usefull)
	psi1 = hu.constructHalo(N//2, L, r, beta_usefull, E_usefull, Psi_usefull, gpu = True)
	PlotPsi(psi1, 'axionyx')
	cp.save("../Data/" + simName + "/psi_axionyx.npy", psi1)
	print(len(beta_usefull))
	su.PrintCompletedTime(time0, "updated scheme")

	r, rho_iso, rho_app, rho_app_lazy = su.gpu2cpu(r, rho_iso, rho_app, rho_app_lazy)
	fo = pu.FigObj()
	fo.AddPlot(r, rho_iso, label = 'target')
	fo.AddLine(r, rho_app, label = 'axionyx')
	fo.AddLine(r, rho_app_lazy, label = 'lazy')
	fo.AddVertLine(R_ein, label = 'einstein radius')
	fo.SetLogLog(r, rho_iso)
	fo.SetXLabel(r'$r \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$\rho \, [M_\odot \mathrm{kpc}^{-3}]$')
	fo.legend()
	fo.SaveInDataDir(simName,"target_profiles")
	# fo.show()