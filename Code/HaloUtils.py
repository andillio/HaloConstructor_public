# pylint: disable=C,W
import astroUtils as au
import numpy as np 
import mathUtils as mu
import time
import sysUtils as su
import gridUtils as gu
import statUtils as st
import scipy as sp
CUPY_IMPORTED = True 
try:
	import cupy as cp 
	import cupyx as cpx
except ImportError:
	CUPY_IMPORTED = False

class State(object):

	def __init__(self):
		self.gpu_all = False
STATE = State()


# TODO: add warning about scipy ceiling
# gives eigenfunctions and energies for a halo profile
# funct(x,f,x,x,f,b,b)
def Eigenfuncs(r, Ecutoff, phi = [], rho = [], hbar_ = 1.
	, verbose = True, normalize = True, l_max = 85):
	"""
	returns the eigenfunctions and values for the Hamiltonian for the given (radial) density

	:r: array-like, the radial values 
	:Ecutoff: float, eigenvalues with greater energy will not be used, Ecuttoff = E/hbar
	:phi: array-like, the radial potential of the halo, default: empty, will try to use rho 
		to generate potential
	:rho: array-like, the radial density of the halo, default: empty, will try to use phi
		to generate potential
	:hbar_: float, plancks reduced constant over the particle mass, default: 1
	:verbose: bool, print out timing information, default: True
	:normalize: bool, normalize the eigenfunctions, defualt: True
	:l_max: int, maximum angular momentum to use, default 85 (scipy's max)

	:return: (array-like, array-like), (E, Psi)
		E[eigen number, 3] - array of eigenvalues in asending order by real part, 
			E[i] = [energy val / hbar_, n, l]
		Psi[eigen number, r] - array of corresponding eigenvectors
	"""
	# get params
	if len(phi) == 0 and len(rho) == 0:
		raise Exception("phi and rho have not been specified," +\
			" there is no way to determine the potential.\n" +\
			"recommendation: specify rho or phi arguments")

	N = len(r)
	Phi_r = phi
	if len(phi) == 0 and len(rho) != 0:
		Phi_r = au.radialPotential(r, rho)
	dr = (np.max(r) - np.min(r))/(N-1)

	vals = [] # eigenvalues
	vecs = [] # eigenvectors
	nFound = 1
	l = 0
	totFound = 0

	time0 = time.time()
	if  verbose:
		print("beginning loop over angular momentum...\n")
	# loop over angular momenta
	while l <= l_max and nFound > 0:
		# naive implementation
		# H_ = au.Hamiltonian_radial(r, Phi_r, l = l,
		# 	 hbar_ = hbar_, dx = dr) # Hamiltonian for this angular momentum 
		# E, u = mu.Eig(H_) # list of eigenvals and vecs in ascending order
		# axionyx implementation
		H_diag, H_offDiag = au.Hamiltonian_radial_tridiag(
			r, Phi_r, l = l, hbar_ = hbar_, dx = dr)
		E, u = mu.Eig_tridiag(H_diag, H_offDiag)	
		vals_ = E[E<Ecutoff]

		nFound = len(vals_)
		l += 1
		totFound += nFound

		if nFound > 0:
			vals.append(vals_)

			vecs_ = u[:,E<Ecutoff] / r[:,np.newaxis]
			vecs_ = np.swapaxes(vecs_,0,1)
			vecs.append(vecs_)

		# print statement
		if verbose:
			su.repeat_print(
				"l = {0}. {1} eigenvalues used. {2} total eigenvectors used".format(
					l,nFound,totFound))

	nAngMom = len(vals)

	if nAngMom < 1:
		raise Exception("No eigenvectors found.\n" +\
			"recommendation: change the cutoff energy")


	E = np.zeros((totFound, 3)) # stores energy, energy q num, and angular momentum q num
	Psi = np.zeros((totFound, N)) + 0j
	currentIndex = 0

	for l in range(nAngMom):
		vals_ = np.array(vals[l])
		nFound = len(vals_)
		vecs_ = np.array(vecs[l])

		E[currentIndex:currentIndex + nFound,0] = vals_ # assign energies eigenvalues
		E[currentIndex:currentIndex + nFound,1] = np.arange(nFound) # assign energy q number
		E[currentIndex:currentIndex + nFound,2] = np.ones(nFound)*l # assign ang mom q number

		# normalize these vectors
		if normalize:
			norms = np.sqrt(np.sum( 4*np.pi*r**2 * np.abs(vecs_)**2, axis = 1)*dr)
			Psi[currentIndex:currentIndex + nFound,:] = vecs_ / norms[:,np.newaxis]
		else:
			Psi[currentIndex:currentIndex + nFound,:] = vecs_

		currentIndex += nFound

	if verbose:
		print("\neigenvalue analysis complete...")
	su.PrintCompletedTime(time0)

	return E, Psi


# TODO: 
# - comment these
# - add gpu 
# given weights and degeneracy weight eigenfuncs return density
def dens_out_func(beta, X, beta0 = 0):
	"""
	given weight estimators beta, and degeneracy weighted eigenfunc matrix X,
	return the corresponding density

	:beta: array-like, 
	"""
	beta_ = beta.copy()
	if beta0 > 0:
		beta_[0] =beta0
	return np.einsum("ji,j->i", X, beta_)


def cost_func(beta_geuss, X, rho_target, beta0 = 0):
	rho_out = dens_out_func(beta_geuss, X, beta0)
	return np.sum( ( (rho_out - rho_target)/rho_target)**2 ) 



def get_potential_and_density(rmin, rmax, Rs, rho0, m22, rc,
	 dr = 0, maxFactor = 10, fullReturns = False, gpu = False):
	if dr == 0:
		dr = rmin
	r_int = np.arange(rmin, rmax*maxFactor, dr) # for integrating Poissons equations
	r = np.arange(rmin,rmax, dr)
	Nr = len(r)

	rho_Total = au.CoredNFW(r_int, Rs, rho0, m22, rc)

	### solve Poisson equation
	Phi_r = au.radialPotential(r_int, rho_Total) 
	if fullReturns:
		return Phi_r[:Nr], rho_Total[:Nr], r[:Nr]
	return Phi_r[:Nr], rho_Total[:Nr]


def constructProfile(meandens, Rs, Rhalf, m22, rmin, rmax = 0, dr = 0, maxRFactor = 10,
	 maxEfactor = 1., cutoffRFactor = .9, fullReturns = False, betaCutOff = 1e-10):
	"""
	gives approximated halo profile

	:meandens: float, mean density 
	:Rs: float, scale radius
	:Rhalf: float, half light radius
	:m22: float, dark matter mass [1e-22 eV]
	:rmin: float, minimum halo radius in returned profile
	:rmax: float, maximum halo radius in calculations, default: use 1.16*Rvir
	:dr: float, inter-radius spacing, default: use rmin
	:maxRFactor: float, potential calculation integrates out to rmax*maxRFactor, default: 10
	:maxEfactor: float, factor multiplying the energy cutoff, i.e. Ecutoff = Evir*maxEfactor,
		 default: 1
	:cutoffRFactor: float, max radius in profile is rmax*cutoffRfactor, default: .9
	:fullReturns: bool, return the target density profile, default: False
	:betaCutOff: float, eigenfuncs with weight less than weight_max*betaCutOff will be
		excluded, default: 1e-10

	:return: (array-like, array-like, array-like, array-like, array-like),
		 (r, rho_app, rho_target, beta, X[eigen number, r]),\n
		r - radial coordinate for profile, \n
		rho_app - the constructed density profile, \n
		(fullReturns only) \n
		rho_target - the target profile, \n
		beta - the eigenfunction weights, \n
		X - eigenfunction matrix
	"""
	# NFW stuff
	Rvir = Rhalf/0.015 
	Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
	con = Rvir / Rs # concentration parameter
	rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3
	
	# soliton stuff
	rc = float(1.6 * (m22)**(-1) * (Mvir/1.0e9)**(-1/3.))# core radius in kpc
	hbar_ = au.h_tilde(m22) # hbar / m

	if rmax == 0:
		rmax = Rvir*1.16
	if dr == 0:
		dr = rmin
	r = np.arange(rmin,rmax, dr)

	### get potential
	Phi_r_tr, rho_Total_tr = get_potential_and_density(rmin, rmax, Rs, rho0, m22, rc,
	 dr = dr, maxFactor = 10)

	### get eigenfuncs
	Em = -.5*Mvir*au.G / Rvir / hbar_ 
	Ecutoff = Em*maxEfactor
	E, Psi = Eigenfuncs(r, Ecutoff, phi = Phi_r_tr, hbar_ = hbar_, normalize = True)
	l = E[:,2]

	### perform fit
	X = np.einsum("i,ij->ij", 2*l + 1, np.abs(Psi)**2)	
	X0 = X[0,r < rc]
	X0 = np.reshape(X0, (1,len(X0)))
	X_else = X[1:,:int(cutoffRFactor*len(r))]
	X = X[:,:int(cutoffRFactor*len(r))]
	Psi = Psi[:,:int(cutoffRFactor*len(r))]

	rho_inner = rho_Total_tr[r < rc]
	rho_mid = rho_Total_tr[:int(cutoffRFactor*len(r))]

	beta0 = st.linearFit_beta_pos(X0, rho_inner)
	beta_ = st.linearFit_beta_pos(X_else,rho_mid - beta0 * X[0,:])
	beta = np.zeros(len(beta_) + 1)
	beta[0] =beta0
	beta[1:] = beta_
	beta_usefull = beta[beta>np.max(beta)*betaCutOff]
	X_usefull = X[beta>np.max(beta)*betaCutOff,:]
	E_usefull = E[beta>np.max(beta)*betaCutOff]
	Psi_usefull = Psi[beta>np.max(beta)*betaCutOff,:]

	# rho_app = np.einsum("ji,j->i", X, beta)[:int(cutoffRFactor*len(r))]
	rho_app = np.einsum("ji,j->i", X_usefull, beta_usefull)

	if fullReturns:
		return r[:int(cutoffRFactor*len(r))], rho_app, rho_mid,\
			beta_usefull, E_usefull, Psi_usefull

	return r[:int(cutoffRFactor*len(r))], rho_app


# TODO: refactor the part of the function once you have the eigenvalues and rho_target
def constructProfile_axionyx(meandens, Rs, Rhalf, m22, rmin, rmax = 0, dr = 0, maxRFactor = 10,
	 maxEfactor = 1., cutoffRFactor = .9, fullReturns = False, betaCutOff = 1e-10):
	"""
	gives approximated halo profile

	:meandens: float, mean density 
	:Rs: float, scale radius
	:Rhalf: float, half light radius
	:m22: float, dark matter mass [1e-22 eV]
	:rmin: float, minimum halo radius in returned profile
	:rmax: float, maximum halo radius in calculations, default: use 1.16*Rvir
	:dr: float, inter-radius spacing, default: use rmin
	:maxRFactor: float, potential calculation integrates out to rmax*maxRFactor, default: 10
	:maxEfactor: float, factor multiplying the energy cutoff, i.e. Ecutoff = Evir*maxEfactor,
		 default: 1
	:cutoffRFactor: float, max radius in profile is rmax*cutoffRfactor, default: .9
	:fullReturns: bool, return the target density profile, default: False
	:betaCutOff: float, eigenfuncs with weight less than weight_max*betaCutOff will be
		excluded, default: 1e-10

	:return: (array-like, array-like, array-like, array-like, array-like),
		 (r, rho_app, rho_target, beta, X[eigen number, r]),\n
		r - radial coordinate for profile, \n
		rho_app - the constructed density profile, \n
		(fullReturns only) \n
		rho_target - the target profile, \n
		beta - the eigenfunction weights, \n
		X - eigenfunction matrix
	"""
	# NFW stuff
	Rvir = Rhalf/0.015 
	Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
	con = Rvir / Rs # concentration parameter
	rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3
	
	# soliton stuff
	rc = float(1.6 * (m22)**(-1) * (Mvir/1.0e9)**(-1/3.))# core radius in kpc
	hbar_ = au.h_tilde(m22) # hbar / m

	if rmax == 0:
		rmax = Rvir*1.16
	if dr == 0:
		dr = rmin
	r = np.arange(rmin,rmax, dr)

	### get potential
	Phi_r_tr, rho_Total_tr = get_potential_and_density(rmin, rmax, Rs, rho0, m22, rc,
	 dr = dr, maxFactor = 10)
	Mtot_tr = np.sum(4*np.pi*r**2 * rho_Total_tr)*dr

	### get eigenfuncs
	Em = -.5*Mvir*au.G / Rvir / hbar_ 
	Ecutoff = Em*maxEfactor
	E, Psi = Eigenfuncs(r, Ecutoff, phi = Phi_r_tr, hbar_ = hbar_, normalize = True)
	l = E[:,2]

	### perform fit
	X = np.einsum("i,ij->ij", 2*l + 1, np.abs(Psi)**2)	
	X0 = X[0,r < rc]
	X0 = np.reshape(X0, (1,len(X0)))
	X_else = X[1:,:int(cutoffRFactor*len(r))]
	X = X[:,:int(cutoffRFactor*len(r))]
	rho_Total_tr = rho_Total_tr / Mtot_tr
	rho_inner = rho_Total_tr[r < rc]
	rho_mid = rho_Total_tr[:int(cutoffRFactor*len(r))]

	beta0 = st.linearFit_beta_pos(X0, rho_inner)
	beta_ = st.linearFit_beta_pos(X_else,rho_mid - beta0 * X[0,:])
	beta = np.zeros(len(beta_) + 1)
	beta[0] =beta0
	beta[1:] = beta_

	beta_geuss = np.ones(len(beta)) / len(beta)
	high = 10. 
	low = .005

	beta_bounds = sp.optimize.Bounds(beta_geuss*low, beta_geuss*high, keep_feasible=False)
	print("starting to optimize")
	aln_raw= sp.optimize.minimize(cost_func, beta,
	 args=(X, rho_mid, beta0), bounds = beta_bounds)
	aln_new=aln_raw.x
	aln_new[0] = beta0
	aln_raw= sp.optimize.minimize(cost_func, aln_new,
	 args=(X, rho_mid, beta0), bounds = beta_bounds)
	beta=aln_raw.x
	beta[0] = beta0
	beta_usefull = beta[beta>np.max(beta)*betaCutOff]
	X_usefull = X[beta>np.max(beta)*betaCutOff,:]
	E_usefull = E[beta>np.max(beta)*betaCutOff]
	Psi_usefull = Psi[beta>np.max(beta)*betaCutOff,:int(cutoffRFactor*len(r))]
	rho_new = np.einsum("ji,j->i", X_usefull, beta_usefull)	
	print("done optimizing")

	print("Mtot:",Mtot_tr, Mvir)
	if fullReturns:
		return r[:int(cutoffRFactor*len(r))], rho_new*Mtot_tr, rho_mid*Mtot_tr,\
			beta_usefull*Mtot_tr, E_usefull, Psi_usefull

	return r[:int(cutoffRFactor*len(r))], rho_new*Mtot_tr


def constructHalo(N, L, r, beta, E, zi,gpu = False):
	"""
	constructs 3D halo

	:r: array-like, radius at which zi are defined
	:beta: array-like, weights of each eigenvector
	:E: array-like, quantum numbers and eigenvalues
	:zi: array-like, eigenvectors
	:N: int, grid resolution
	:L: float, box size

	:return: array-like, psi the 3D field
	"""

	### get coorindates
	np_ = np
	if CUPY_IMPORTED and gpu:
		np_ = cp
	dx = L/N
	R, Theta, Phi = gu.sphrGrid(N,L,gpu=gpu)

	rmin = np_.min(r)
	rmax = np_.max(r)

	psi = np_.zeros(R.shape) + 0j

	time0 = time.time()
	total = int(np_.sum(E[:,2]*2 + 1)) # count total eigenvectors 
	done = 0
	print(f"using {len(E)} radial eigenvectors in construction")

	### loop over radial eigenvectors 
	# (start from back so the time estimate is convervative)
	for i in range(len(E)-1,-1,-1):
		l = int(E[i,2]) # angular momentum quntum number
		n = E[i,1] # energy quantum number
		anlm = np_.sqrt(beta[i]) # complex amplitude

		zi_ = np_.interp(R, r, zi[i]) # interp the radial eigenvector on our grid
		zi_[R < rmin] = np_.max(zi[i]) # handle the boundaries
		zi_[R > rmax] = np_.min(zi[i])

		# loop over angular momenta degeneracies
		for m in range(-l,l+1):
			Y = mu.sph_harm(m,l,Theta,Phi, gpu=gpu) # relevant spherical harmonic
			phi = np_.random.uniform(0,2*np_.pi) # random phase
			norm = np_.sum(np_.abs(zi_*Y)**2)*dx**3 # L2 norm
			psi += anlm*zi_*Y*np_.exp(1j*phi) / np_.sqrt(norm) # add to field

			done += 1
			su.PrintTimeUpdate(done,total,time0) # timing info

	return psi


# TODO: comment these
def haloRealization(meandens, Rs, Rhalf, m22, rmin, N, L, gpu = False):
	r, rho_new, rho_target, beta, E, Psi = \
		constructProfile_axionyx(meandens, Rs, Rhalf, m22, rmin, fullReturns= True)

	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	if gpu_:
		r, beta, E, Psi = gu.cpu2gpu(r, beta, E, Psi)
	
	return constructHalo(N, L, r, beta, E, Psi, gpu = gpu_)


# TODO: gpu implementation
def constructSmoothHaloDensity(N, L, r_tr, rho_Total_tr, gpu = False):
	R, Theta, Phi = gu.sphrGrid(N,L)

	rho_smooth = np.interp(R,r_tr, rho_Total_tr)

	return R, rho_smooth

# TODO: gpu implementation
def lensModelSurfaceDensity(meandens, Rs, Rhalf, m22, rmin, N, L, frac = 1., gpu = False):
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np = cp
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
	dx = N/L
	sigma_d = np.sqrt(Mvir*au.G / Rvir)

	### get potential
	Phi_r_tr, rho_Total_tr, r_tr = get_potential_and_density(rmin, rmax, Rs, rho0, m22, rc,
	 dr = dr, maxFactor = 10, fullReturns=True, gpu = gpu)

	R, rho_smooth = constructSmoothHaloDensity(N, L, r_tr, rho_Total_tr, gpu = gpu)

	lambda_chi = hbar_ / sigma_d
	Sigma_c = 1. 

	R_sigma, Phi_sigma = gu.cirGrid(N,L)
	Sigma = rho_smooth.sum(axis = 2)*dx
	Var = np.sqrt(np.pi)*lambda_chi*np.sum(rho_smooth**2, axis = 2)*dx

	white_noise = np.random.normal(0, np.sqrt(Var))
	kernel = np.exp(-(R_sigma/lambda_chi)**2)
	return Sigma + frac*mu.fftConvolve(white_noise, kernel).real