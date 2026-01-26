# pylint: disable=C,W
import HaloUtils as hu
import plotUtils as pu 
import gridUtils as gu
import mathUtils as mu
import astroUtils as au
import matplotlib.pyplot as plt
import numpy as np

### density profile parameters
# NFW parameters
meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
Rs = 2. # scale radius in kpc
Rhalf = 0.3 # half light radius

m22 = 5. # mass of uldm particle

rmin = 0.05 

# NFW stuff
Rvir = Rhalf/0.015 
Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
con = Rvir / Rs # concentration parameter
rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3

# soliton stuff
rc = float(1.6 * (m22)**(-1) * (Mvir/1.0e9)**(-1/3.))# core radius in kpc
hbar_ = au.h_tilde(m22) # hbar / m

rmax = Rvir*1.16
dr = rmin
r = np.arange(rmin,rmax, dr)

### halo parameters
N = 128 # resolution
L = Rvir*2 / np.sqrt(3)
dx = L/N
sigma_d = np.sqrt(Mvir*au.G / Rvir)


def constructSmoothHaloDensity(r_tr, rho_Total_tr):
	R, Theta, Phi = gu.sphrGrid(N,L)

	rho_smooth = np.interp(R,r_tr, rho_Total_tr)

	return R, rho_smooth



if __name__ == "__main__":
	### get potential
	Phi_r_tr, rho_Total_tr, r_tr = hu.get_potential_and_density(rmin, rmax, Rs, rho0, m22, rc,
	 dr = dr, maxFactor = 10, fullReturns=True)

	R, rho_smooth = constructSmoothHaloDensity(r_tr, rho_Total_tr)

	lambda_chi = hbar_ / sigma_d
	Sigma_c = 1. 

	R_sigma, Phi_sigma = gu.cirGrid(N,L)
	Sigma = rho_smooth.sum(axis = 2)*dx
	Var = np.sqrt(np.pi)*lambda_chi*np.sum(rho_smooth**2, axis = 2)*dx

	fo = pu.FigObj(3)

	white_noise = np.random.normal(0, np.sqrt(Var))
	# white_noise = np.random.exponential(np.sqrt(Var))
	kernel = np.exp(-(R_sigma/lambda_chi)**2)
	Sigma_app = Sigma + mu.fftConvolve(white_noise, kernel).real
	fo.AddDens2d([-L/2.,L/2], Sigma_app, log = True)

	fo.setTitle("density, standard")

	white_noise = np.random.normal(0, np.sqrt(Var/4))
	# white_noise = np.random.exponential(np.sqrt(Var))
	kernel = np.exp(-(R_sigma/lambda_chi)**2)
	Sigma_app = Sigma + mu.fftConvolve(white_noise, kernel).real
	fo.AddDens2d([-L/2.,L/2], Sigma_app, log = True)
	fo.SetTitle("density, four fields")
	fo.RemoveYLabels()

	white_noise = np.random.normal(0, np.sqrt(Var))
	# white_noise = np.random.exponential(np.sqrt(Var))
	kernel = np.exp(-(R_sigma/lambda_chi)**2)
	Sigma_app = Sigma + .25*mu.fftConvolve(white_noise, kernel).real
	fo.AddDens2d([-L/2.,L/2], Sigma_app, log = True)
	fo.RemoveYLabels()
	fo.SetTitle("density, fraction = 0.25")

	# fo.AddImshow(white_noise)
	plt.subplots_adjust(wspace=0,hspace = 0)
	fo.save()
	fo.show()