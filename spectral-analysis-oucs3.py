from oucs3 import init_oucs3
import numpy as np
import numpy.matlib as mlib
import matplotlib.pyplot as plt
import numba # numba makes code faster comment if not installed

##Note: CD2 is used for Uxx term



def Spectral(n, beta1, beta2, Cfl, Pe, a, b, Node):
	"""
	Returns Numerical G and Physical G for given gridsize(n), 
	beta values, Cfl, Pe, a, b and node at which to calculate G's.
	CD2 for Uxx term.
	
	"""
	C = init_oucs3(n, beta1, beta2) # initializing C = Ainverse * B

	A = mlib.zeros((100,1))
	Sin_mtrix = mlib.zeros((n,1)) # matrix to store sin(l-j) values where j is node and l is all points
	Cos_mtrix = mlib.zeros((n,1)) # similarly cos matrix
	Gnum = mlib.zeros((100,1)) #Numerical G
	Gphy = mlib.zeros((100,1)) # Physical G
	khi = np.linspace(0,np.pi,100) # Values of kh 
	Node = int(Node) # node to calculate and compare G's
	#make the numpy matrices complex else it will discard imaginary parts
	A = A + 0j
	Gnum = Gnum + 0j
	Gphy = Gphy + 0j

	for j in range(100):
		for l in range(n):
			Sin_mtrix[l] = np.sin((l-Node)*khi[j])
			Cos_mtrix[l] = np.cos((l-Node)*khi[j])
		A[j] = Cfl*C[Node,:]*Cos_mtrix + 4*Pe*((np.sin(khi[j]/2.))**2) + 1j*Cfl*C[Node,:]*Sin_mtrix 
		# using cd2 for double derivative on rightside Uxx
		Gnum[j] = 1 - A[j] + (A[j]**2)/2. - b*(A[j]**3) + a*(A[j]**4)
		Gphy[j] = (np.exp(-1j*Cfl*khi[j]))*(np.exp(-Pe*(khi[j]**2)))

	return Gnum, Gphy, khi

# Spectral is the func
# example 	

Cfl = 0.5 # CFL number
Pe = 0.1 # Peckle number
a = 1/24.
b = 1/6. 

n = 30 # grid size
beta1 = -0.025 # beta values for near boundary stencils
beta2 = 0.09
Node = n/2

Gnum, Gphy, khi = Spectral(n, beta1, beta2, Cfl, Pe, a, b, n/2)
plt.plot(khi , np.real(Gnum))
plt.plot(khi , np.real(Gphy))
plt.legend(["G-numerical", "G-physical"])
plt.show()

