# pylint: disable=C,W
import HaloUtils as hu
import mathUtils as mu
import astroUtils as au
import gridUtils as gu
import plotUtils as pu 
import numpy as np
import time 
import sysUtils as su

### density profile parameters
# NFW parameters
meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
Rs = 2. # scale radius in kpc
Rhalf = 0.3 # half light radius

m22 = 5. # mass of uldm particle

rmin = 0.05 


### halo parameters
N = 256 # resolution
Rvir = Rhalf / .015
L = Rvir*2 / np.sqrt(3)

maxEmode = 300

def compareDensityMethods(r, rho_target, rho_app, rho_new):
	fo = pu.FigObj()
	fo.AddPlot(r, rho_target, label = r"target density")
	fo.SetLogLog(r, rho_target)
	fo.IncreaseYLim(upper_factor=1.5)
	fo.AddLine(r, rho_app, mk = 'x', ls = '', label = r"constructed density")
	fo.AddLine(r, rho_new, mk = 'x', ls = '', label = r"constructed density new")
	fo.SetXLabel(r"$r \, [\mathrm{kpc}]$")
	fo.SetYLabel(r"$\rho \, [M_\odot \mathrm{kpc}^{-3}]$")
	fo.legend()

	fo.show()



def constructHalo(r, beta, E, zi):
	X, Y, Z = gu.grid((N, N, N), L = L)
	R, Theta, Phi = gu.cart2sphr(X,Y,Z)

	rmin = np.min(r)
	rmax = np.max(r)

	psi = np.zeros(R.shape) + 0j

	time0 = time.time()
	total = int(np.sum(E[:,2]*2 + 1))
	done = 0

	maxE = len(E)
	if maxEmode >= 0 and maxEmode < len(E):
		maxE = maxEmode

	for i in range(maxE-1,-1,-1):
		l = int(E[i,2])
		n = E[i,1]
		anlm = np.sqrt(beta[i])


		zi_ = np.interp(R, r, zi[i])
		zi_[R < rmin] = np.max(zi[i])
		zi_[R > rmax] = np.min(zi[i])

		for m in range(-l,l+1):
			Y = mu.sph_harm(m,l,Theta,Phi)
			phi = np.random.uniform(0,2*np.pi)
			psi += anlm*zi_*Y*np.exp(1j*phi)

			done += 1
			su.PrintTimeUpdate(done,total,time0)

	return psi


if __name__ == "__main__":
	time0 = time.time()

	### construct profile using axionyx method
	r, rho_new, rho_target, beta_new, E_new, Psi_new = \
		hu.constructProfile_axionyx(meandens, Rs, Rhalf, m22, rmin, fullReturns= True)
	np.save("qNums.npy",E_new)
	psi_new = constructHalo(r, beta_new, E_new, Psi_new)
	np.save(f"eri_{maxEmode}Emax.npy", psi_new)
	rho_slice_new = np.log(np.abs(psi_new[N//2,:,:])**2)

	### construct profile using my method
	r, rho_app, rho_target, beta, E, Psi = \
		hu.constructProfile(meandens, Rs, Rhalf, m22, rmin, fullReturns= True)
	psi = constructHalo(r, beta, E, Psi)
	rho_slice = np.log(np.abs(psi[N//2,:,:])**2)




	# compareDensityMethods(r, rho_target, rho_app, rho_new)

	# TODO:
	# - try this but with gpu
	### construct halo

	su.PrintCompletedTime(time0)


	fo = pu.FigObj(2)
	fo.AddImshow(rho_slice)
	fo.AddImshow(rho_slice_new)

	fo = pu.FigObj()
	rho = np.abs(psi_new)**2
	R_grid,_,_ = gu.sphrGrid(N, L)
	rvals, rho_profile = au.radialProfile(R_grid, rho) # radial fdm profile
	fo.AddPlot(rvals, rho_profile)
	fo.SetLogLog(rvals, rho_profile)

	fo.show()




