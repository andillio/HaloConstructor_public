"""
This file finds the eigenvectors for the specified density
"""
# pylint: disable=C,W 
import numpy as np
import scipy as sp
import gridUtils as gu
import mathUtils as mu
import plotUtils as pu 
import statUtils as stu
import astroUtils as au
import HaloUtils as hu


# configuration parameters
# simulation parameters
# input parameters
# L = 200. # box length in kpc
# N = 1024 # resolution
# # derived paramters
# dx = L / N
# r = np.linspace(dx, L/2.*np.sqrt(3) - dx/2., N)
# dr = L/2.*np.sqrt(3) / N 

# NFW parameters
# input parameters
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
# derived parameters
rc = float(1.6 * (m22)**(-1) * (Mvir/1.0e9)**(-1/3.))# core radius in kpc
print(rc)
hbar_ = au.h_tilde(m22) # hbar / m

rmax = Rvir*1.16
rmin = 0.05 
dr = rmin 


if __name__ == "__main__":

	### obtain density profile
	r_int = np.arange(rmin, rmax*20, dr) # for integrating Poissons equations
	r = np.arange(rmin,rmax, dr)
	Nr = len(r)
	print(dr, (np.max(r) - np.min(r))/(Nr-1))


	rho_Total = au.CoredNFW(r_int, Rs, rho0, m22, rc)
	Mtot = np.sum(4*np.pi*r_int**2 * rho_Total)*dr # mass enclosed using riemann sum

	### solve Poisson equation
	Phi_r = au.radialPotential(r_int, rho_Total) 
	Phi_r_tr = Phi_r[:Nr]

	### energy cutoff
	Em = -.5*Mvir*au.G / Rvir / hbar_ 
	factor = 2. # input parameter
	Ecutoff = Em*factor
	rho_Total_tr = rho_Total[:Nr]
	Mtot_tr = np.sum(4*np.pi*r**2 * rho_Total_tr)*dr
	E, Psi = hu.Eigenfuncs(r, Ecutoff, phi = Phi_r_tr, hbar_ = hbar_, normalize = True)
	l = E[:,2]

	### perform fit
	X = np.einsum("i,ij->ij", 2*l + 1, np.abs(Psi)**2)	
	X0 = X[0,r < rc]
	X0 = np.reshape(X0, (1,len(X0)))
	X_else = X[1:,:]
	rho_Total_tr = rho_Total_tr / Mtot_tr
	rho_inner = rho_Total_tr[r < rc]

	from scipy.optimize import nnls
	beta0, res = nnls(X0.T, rho_inner)
	beta_, res = nnls(X_else.T,rho_Total_tr - beta0 * X[0,:])
	beta = np.zeros(len(beta_) + 1)
	beta[0] = beta0
	beta[1:] = beta_

	rho_app = np.einsum("ji,j->i", X, beta)*Mtot_tr
	ground = X[0,:]*beta[0]*Mtot_tr
	print(len(beta[beta>0]))
	print(np.sum(beta))

	beta_geuss = np.ones(len(beta)) / len(beta)
	rho_geuss = np.einsum("ji,j->i", X, beta_geuss)*Mtot_tr
	high = 10. 
	low = .005

	fo = pu.FigObj()
	fo.AddPlot(r_int, rho_Total, label = r"target")
	# fo.AddLine(r, rho_geuss, mk = 'x', label = r"geuss")
	# fo.AddLine(r, rho_geuss*high, mk = 'x', label = r"high")
	# fo.AddLine(r, rho_geuss*low, mk = 'x', label = r"low")
	fo.SetLogLog(r, rho_Total_tr*Mtot_tr)
	fo.IncreaseYLim(upper_factor=10)
	# fo.AddLine(r, rho_app, mk = 'x', label = r"my constructed")


	beta_bounds = sp.optimize.Bounds(beta_geuss*low, beta_geuss*high, keep_feasible=False)
	print("starting to optimize")
	aln_raw= sp.optimize.minimize(hu.cost_func, beta,
	 args=(X, rho_Total_tr,beta0), bounds = beta_bounds)
	aln_new=aln_raw.x
	aln_new[0] = beta0
	aln_raw= sp.optimize.minimize(hu.cost_func, aln_new,
	 args=(X, rho_Total_tr, beta0), bounds = beta_bounds)
	aln_new=aln_raw.x
	aln_new[0] = beta0
	# aln_new[aln_new < np.max(aln_new*1e-4)] *= 0
	print(len(aln_new[aln_new > np.max(aln_new*1e-4)]))
	rho_new = hu.dens_out_func(aln_new, X)
	print("done optimizing")
	fo.AddLine(r, rho_new*Mtot_tr, mk = 'x', label = r"new constructed")

	print(hu.cost_func(beta_geuss, X, rho_Total_tr))
	print(hu.cost_func(beta, X, rho_Total_tr))
	print(hu.cost_func(beta_geuss*low, X, rho_Total_tr))
	print(hu.cost_func(beta_geuss*high, X, rho_Total_tr))
	print(hu.cost_func(aln_new, X, rho_Total_tr))

	fo.legend()
	fo.save("new_Density")
	fo.show()
