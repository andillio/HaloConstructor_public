# pylint: disable=C,W
from cupy import dtype
import numpy as np 
import plotUtils as pu 
import gridUtils as gu 
import mathUtils as mu
import scipy.special as spp
from scipy.special import lpmv

simName = "low_m_low_res"
R_ein = 2.062
L = 3*R_ein
m = -86
l = 86

# R, Theta, Phi = gu.sphrGrid(N,L,dtype = np.double)

# Y = spp.sph_harm(m,l,Theta,Phi)

# # Y = mu.sph_harm(m,l,Theta,Phi) # relevant spherical harmonic
# print(lpmv(m,l,np.cos(Theta)))
# print(Y)

psi = np.load(simName + "psi1.npy")
N = len(psi)
# print(psi)
rho = np.abs(psi)**2
Sigma = np.sum(rho, axis = 0)
x = np.linspace(-1*L/2, L/2, N)

fo = pu.FigObj()
# fo.AddPlot(Sigma)
fo.AddDens2d(x, np.log(Sigma) )
# fo.AddVertLine(R_ein, color = 'r')
# fo.AddVertLine(-R_ein, color = 'r')
# fo.AddHorLine(-R_ein, color = 'r')
# fo.AddHorLine(R_ein, color = 'r')
fo.AddCircle(R_ein, color = 'w')
fo.SetXLim(-L/2,L/2.)
fo.SetYLim(-L/2.,L/2.)
fo.SetTitle(r"density projection")
fo.save("isothermProjection")
# fo.AddLine(x, rho[N//2,N//2,:])
fo.show()
