import numpy as np
import numpy.matlib as mlib
from numpy.linalg import inv as matrix_inv
import numba # numba makes code faster comment if not installed

@numba.jit # comment if numba is not installed 
def init_oucs3(n,beta_1,beta_2):
	"""
	Intializes a C = Ainverse*B matrix with OUCS3 for internal 
	points and Special stencils for boundary and near boundary points

	"""
	# constants to calculate coefficients in stencils
	D = 0.379384912
	F = 1.57557379
	E = 0.183205192
	eta = -2
 
	# Coefficients of OUCS3 on left side	
	p_lp1 = D + (eta/60.)
	p_lm1 = D - (eta/60.)

	# Coefficients of OUCS3 on left side
	q_0 = -11.*eta/150.
	q_m1 = (-E/2.) + (eta/30.)
	q_p1 = (+E/2.) + (eta/30.)
	q_m2 = (-F/4.) + (eta/300.)
	q_p2 = (+F/4.) + (eta/300.)

		
	B = mlib.zeros((n,n))
	# Boundary and near boundary stencils
	B[0,0] = -3./2.
	B[0,1] = 4./2.
	B[0,2] = -1./2.

	B[1,0] = ((2*beta_1/3.) - (1/3.))
	B[1,1] = -((8*beta_1/3.) + (1/2.))
	B[1,2] = (4*beta_1 + 1.) 
	B[1,3] = -((8*beta_1/3.)+(1/6.))
	B[1,4] = 2*beta_1/3.
	B[n-1,n-1] = +3./2.
	B[n-1,n-2] = -4./2.
	B[n-1,n-3] = +1./2.
	beta = 0.09
	B[n-2,n-1] = -((2*beta_2/3.) - (1/3.))
	B[n-2,n-2] = +((8*beta_2/3.) + (1/2.))
	B[n-2,n-3] = -(4*beta_2 + 1.) 
	B[n-2,n-4] = +((8*beta_2/3.)+(1/6.))
	B[n-2,n-5] = -2*beta_2/3.

	# Inner stencil 
	for k in range(2, n-2):
		B[k,k-2] = q_m2
		B[k,k-1] = q_m1
		B[k,k] = q_0
		B[k,k+1] = q_p1
		B[k,k+2] = q_p2

	A = mlib.zeros((n,n))
	
	# Inner stencil of A
	for i in range(1,n):
		for j in range(1,n):
			if i==j:
				A[i,j] = 1.
				A[i-1,j] = p_lp1
				A[i, j-1] = p_lm1
	# Boundary and near boundary stencils of A	
	A[0,0] = 1.
	A[0,1] = 0.
	A[1, 0] = 0.
	A[1, 2] = 0.
	A[n-2, n-1] = 0.
	A[n-1, n-2] = 0.
	A[n-2, n-3] = 0.

	A_inv = matrix_inv(A)

	C = A_inv*B
	return C

























