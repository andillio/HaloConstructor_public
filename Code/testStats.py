# pylint: disable=C,W
import numpy as np 
import astroUtils as au
import sysUtils as su
import plotUtils as pu

simName = "test_highRes"

if __name__ == "__main__":

	Psi = np.load(su.getDataDir(simName) + "Psi.npy")
	E = np.load(su.getDataDir(simName) + "E.npy")
	beta = np.load(su.getDataDir(simName) + "beta.npy")
	l = E[:,2]
	X = np.einsum("i,ij->ij", 2*l + 1, np.abs(Psi)**2)
	rho_psi = np.einsum("ji,j->i", X, beta)

	R = np.load(su.getDataDir(simName) + "R.npy")
	N = R.shape[0]
	D = 3
	dx = np.max(R) / (N-1) / np.sqrt(D)
	psi_new = np.load(su.getDataDir(simName) + "psi_new.npy")
	rho_e = np.abs(psi_new)**2
	print(np.max(rho_e), np.min(rho_e))
	print(R.shape, rho_e.shape)

	r_e, rho_e_pro = au.radialProfile(R, rho_e)

	rho_smooth = np.load(su.getDataDir(simName) + "rho_smooth.npy")
	r_sm, rho_smooth_pro = au.radialProfile(R, rho_smooth)
	
	rho_target = np.load(su.getDataDir(simName) + "rho_target.npy")
	r = np.load(su.getDataDir(simName) + "r.npy")
	
	Sigma_gr = np.load(su.getDataDir(simName) + "Sigma_gr.npy")
	Sigma_pbh = np.load(su.getDataDir(simName) + "Sigma_pbh.npy")
	Sigma_e = np.sum(np.abs(psi_new)**2, axis = 1)*dx

	print(np.max(rho_e_pro), np.min(rho_e_pro))

	### look at radial profiles
	fo = pu.FigObj()
	fo.AddLine(r, rho_target, label = r"target density")
	fo.AddLine(r_e, rho_e_pro, label = r"actual realization")
	# fo.AddLine(r_sm, rho_smooth_pro, label = r"smooth density")
	fo.AddLine(r, rho_psi, label = r"only radial sum")
	fo.SetLogLog(r, rho_psi)
	fo.legend()
	fo.show()

	### look at power spectrum
	# k_e, PS_e = au.powerspectrum(Sigma_e, dx)
	# k_pbh, PS_pbh = au.powerspectrum(Sigma_pbh, dx)
	# k_gr, PS_gr = au.powerspectrum(Sigma_gr, dx)
	# fo.AddLine(k_e, PS_e, label = r'eigenvalues')
	# fo.AddLine(k_gr, PS_gr, label = r'lens method')
	# fo.AddLine(k_pbh, PS_pbh, label = r'PBH')
	# fo.SetLogLog(k_e, PS_gr)
	# fo.legend()
	# fo.show()