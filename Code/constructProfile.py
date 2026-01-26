"""
This file finds the eigenvectors for the specified density
"""
# pylint: disable=C,W 
import numpy as np
import gridUtils as gu
import mathUtils as mu
import plotUtils as pu 
import statUtils as stu
import astroUtils as au
from HaloUtils import Eigenfuncs


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
hbar_ = au.h_tilde(m22) # hbar / m

rmax = Rvir*1.16
rmin = 0.05 
dr = rmin 


if __name__ == "__main__":

	### obtain density profile
	r_int = np.arange(rmin, rmax*10, dr) # for integrating Poissons equations
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
	factor = 2 # input parameter
	Ecutoff = Em*factor
	rho_Total_tr = rho_Total[:Nr]
	Mtot_tr = np.sum(4*np.pi*r**2 * rho_Total_tr)*dr
	E, Psi = Eigenfuncs(r, Ecutoff, phi = Phi_r_tr, hbar_ = hbar_, normalize = True)
	l = E[:,2]

	### perform fit
	X = np.einsum("i,ij->ij", 2*l + 1, np.abs(Psi)**2)	
	X0 = X[0,r < rc]
	X0 = np.reshape(X0, (1,len(X0)))
	cuttoff = 1.
	r_cuttoff = r[:int(cuttoff*len(r))]
	X_else = X[1:,:int(cuttoff*len(r))]
	rho_Total_tr = rho_Total_tr 
	rho_inner = rho_Total_tr[r < rc]
	rho_mid = rho_Total_tr[:int(cuttoff*len(r))]

	from scipy.optimize import nnls
	beta0, res = nnls(X0.T, rho_inner)
	beta_, res = nnls(X_else.T,rho_mid - beta0 * X[0,:int(cuttoff*len(r))])
	beta = np.zeros(len(beta_) + 1)
	beta[0] =beta0
	beta[1:] = beta_

	# beta = stu.linearFit(X, rho_Total_tr)
	rho_app = np.einsum("ji,j->i", X, beta)
	ground = X[0,:]*beta[0]
	print(len(beta[beta>0]))
	print(np.sum(beta))
	# print(beta)
	# fo = pu.FigObj()
	# y = np.abs(Psi[1,:])**2
	# print(np.sum(y*4*np.pi*r**2)*dr)
	# fo.AddPlot(r,y)
	# fo.SetLogLog(r,y)
	# fo.show()

	fo = pu.FigObj()
	fo.AddPlot(r_int, rho_Total, label = r"target")
	fo.SetLogLog(r*1.1, rho_Total)
	fo.IncreaseYLim(upper_factor=2)
	fo.AddVertLine(np.max(r_cuttoff), label = r"$r_{max}$")
	fo.AddLine(r, rho_app, mk = 'x', label = r"constructed")
	fo.legend()
	fo.SetXLabel(r"$r \, [\mathrm{kpc}]$")
	fo.SetYLabel(r"$\rho \, [M_\odot \mathrm{kpc}^{-3}]$")
	fo.Save("constructedDensities")

	fo = pu.FigObj()
	fo.AddPlot(r_int, rho_Total, label = r"total density")
	fo.AddLine(r, ground, label = r"ground state")
	fo.SetLogLog(r_int, rho_Total)
	# fo.SetLogLog(r_int, ground)
	fo.IncreaseYLim(upper_factor=8)
	fo.legend()

	fo.show()
