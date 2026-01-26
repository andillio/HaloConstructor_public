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

# maxEmode = 300

if __name__ == "__main__":
	
	r, rho_new, rho_target, beta_new, E_new, Psi_new = \
		hu.constructProfile_axionyx(meandens, Rs, Rhalf, m22, rmin, fullReturns= True)
	print(E_new)

	n = E_new[:,1]
	l = E_new[:,2]
	E = E_new[:,0]
	print(n)
	print(l)
	print(E)

	fo = pu.FigObj()

	# fo.AddPlot(n,E, ls = '', mk = 'o', color = 'k')
	# fo.AddLine(n[0:200],E[0:200], ls = '', mk = 'o', color = 'b')
	# fo.AddLine(n[0:10],E[0:10], ls = '', mk = 'o', color = 'r')
	
	# fo.AddPlot(l,E, ls = '', mk = 'o', color = 'k')
	# fo.AddLine(l[0:200],E[0:200], ls = '', mk = 'o', color = 'b')
	# fo.AddLine(l[0:10],E[0:10], ls = '', mk = 'o', color = 'r')

	eigennumber = np.arange(len(E))
	energy = np.sort(E)
	np.save("eigennumber.npy", eigennumber)
	np.save("energy.npy", energy)

	fo.AddLine(eigennumber, energy, ls = '', mk = 'o', color = 'k')
	# fo.AddLine(np.arange(25),np.sort(E[0:25]), ls = '', mk = 'o', color = 'b')

	fo.show()
