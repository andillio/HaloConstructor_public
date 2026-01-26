# pylint: disable=C,W 
import numpy as np
import gridUtils as gu
import mathUtils as mu
import plotUtils as pu 
import astroUtils as au

# configuration parameters
L = 23. # box length in kpc
N = 128 # resolution
rho0 = 1.1e6 # scale density in solar masses / kpc^3
Mvir = 1e10 # virial mass in solar masses
dx = L / N
rc = .9
Rs = 10.
m22 = 1.
hbar_ = au.h_tilde(m22)

x = gu.grid(N, L = L) 
X, Y, Z = gu.grid((N, N, N), L = L)
R, Theta, Phi = gu.cart2sphr(X,Y,Z)
r = np.linspace(dx / 2., L/2., N)
dr = L/2. / N
# del X,Y,Z

Y11 = mu.sph_harm(5,6,Theta,Phi) 
amps = np.abs(Y11)**2
print(amps)
fo = pu.FigObj(2)
fo.AddPlot(amps[:,:,0])
fo.show()


rho_NFW = au.NFW(r, Rs, rho0)
rho_core = au.Core(r, m22, rc)
rho_Total = au.CoredNFW(r, Rs, rho0, m22, rc)
# print(au.CoreRadius(1,1e10, zeta = 1))

# print(rho_core)
# print(rho_NFW)

ax, im = fo.AddPlot(r, rho_NFW)
ax.plot(r, rho_core)
fo.AddLine(r, rho_Total)
fo.SetLogLog(r, rho_NFW)
# fo.AddVertLine(rc*2)

enclMass = mu.cumInt(rho_Total*4*np.pi*r**2, r)
# fo.AddPlot(r, enclMass)
Mtot = np.sum(4*np.pi*r**2 * rho_Total)*dr
# fo.AddHorLine(Mtot)
Phi_r = au.radialPotential(r, rho_Total)

# fo.AddPlot(r, au.radialPotential(r, rho_Total))

# fo.show()

l = 1
T_ = au.kineticOperator(N, hbar_, pts = 3, dx = dx)
V_ = Phi_r/hbar_ - .5*hbar_*l*(l+1)/r**2
V_ = np.diag(V_)
H_ = T_ + V_ + 0j
eigVals, eigVecs = np.linalg.eig(H_)
eig_g = eigVecs[0,:]
print(eigVals)
f = eig_g / r
fo.AddPlot(r, np.abs(f))
# fo.SetLogLog(r, np.abs(eig_g))
fo.show()

# TODO:
# 	- combine them to get halo
