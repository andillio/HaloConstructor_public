# pylint: disable=C,W
import HaloUtils as hu
import mathUtils as mu
import gridUtils as gu
import plotUtils as pu
import cupy as np
import numpy as np_
import time 
import sysUtils as su

### density profile parameters
# NFW parameters
meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
Rs = 2. # scale radius in kpc
Rhalf = 0.3 / 2.# half light radius
m22 = 5. * 3. # mass of uldm particle
rmin = 0.05 


### halo parameters
N = 256 # resolution
Rvir = Rhalf / .015 # virial radius 20 kpc
L = Rvir*2 / np.sqrt(3) # box size
print(L)
dx = L/N
maxE = 10.0#-0.06

def constructHalo(r, beta, E, zi):
	"""
	constructs 3D halo

	:r: array-like, radius at which zi are defined
	:beta: array-like, weights of each eigenvector
	:E: array-like, quantum numbers and eigenvalues
	:zi: array-like, eigenvectors

	:return: array-like, psi the 3D field
	"""

	### get coorindates
	R, Theta, Phi = gu.sphrGrid(N,L,gpu=True)

	rmin = np.min(r)
	rmax = np.max(r)

	psi = np.zeros(R.shape) + 0j

	time0 = time.time()
	total = int(np.sum(E[:,2]*2 + 1)) # count total eigenvectors 
	done = 0
	print(f"using {len(E)} radial eigenvectors in construction")

	### loop over radial eigenvectors 
	# (start from back so the time estimate is convervative)
	for i in range(len(E)-1,-1,-1):
		l = int(E[i,2]) # angular momentum quntum number
		n = E[i,1] # energy quantum number
		E_ = E[i,0] # eigen energy
		if E_ < maxE:
			anlm = np.sqrt(beta[i]) # complex amplitude

			zi_ = np.interp(R, r, zi[i]) # interp the radial eigenvector on our grid
			zi_[R < rmin] = np.max(zi[i]) # handle the boundaries
			zi_[R > rmax] = np.min(zi[i])

			# loop over angular momenta degeneracies
			for m in range(-l,l+1):
				Y = mu.sph_harm(m,l,Theta,Phi, gpu=True) # relevant spherical harmonic
				phi = np.random.uniform(0,2*np.pi) # random phase
				norm = np.sum(np.abs(zi_*Y)**2)*dx**3 # L2 norm
				psi += anlm*zi_*Y*np.exp(1j*phi) / np.sqrt(norm) # add to field

		done += 1
		su.PrintTimeUpdate(done,total,time0) # timing info

	return psi

if __name__ == "__main__":
	time0 = time.time()
	### construct profile using axionyx method
	r, rho_new, rho_target, beta_new, E_new, Psi_new = \
		hu.constructProfile_axionyx(meandens, Rs, Rhalf, m22, rmin, 
			maxEfactor= 1., fullReturns= True)
	r, beta, E, Psi = gu.cpu2gpu(r, beta_new, E_new, Psi_new)
	psi_new = constructHalo(r, beta, E, Psi)
	psi_new = gu.cpuThis(psi_new)
	np.save("halo3.npy", psi_new)
	rho_slice_new = np_.abs(psi_new[N//2,:,:])**2
	M_ref = np_.sum(np_.abs(psi_new)**2)*dx**3
	print(M_ref)
	Sigma_e = np_.sum(np_.abs(psi_new)**2, axis = 2)


	su.PrintCompletedTime(time0, "axionyx method")

