"""
Module: integrate.py

Module used to integrate given functions on arbitrary domains using 
Gaussian quadrature
"""

import numpy as np


# Preload a certain number of quadrature nodes and weights
# to make the quadrature routine more efficient
CONST_NODES = []
CONST_WEIGHTS = []
CONST_N_MAX = 10

for n in range(CONST_N_MAX+1):
	
	if n == 0:
		CONST_NODES.append(None)
		CONST_WEIGHTS.append(None)
		continue

	nodes, weights = np.polynomial.legendre.leggauss(n)	
	CONST_NODES.append(nodes)
	CONST_WEIGHTS.append(weights)



def integrate_2D_quadrature(func, n, x_limit, y_limit):
	
	"""	Integrate the given function (a function of 2 variables) on the 
	domain [x_l, x_u] x [y_l, y_u] using gaussian quadrature.
	
	Args:
	    func (function): The multivariate function to integrate. Defined on the
	    	x and y domain.
	    n (int): The number of quadrature points to use along each direction
	    	on the square domain
	    x_limit (list): The x limits for the integration
	    y_limit (list): The y limits for the integration
	
	Returns:
	    float: The value of the integral over the 2D domain
	"""

	# Mapping functions

	x1, x2 = x_limit[0], x_limit[1]
	y1, y2 = y_limit[0], y_limit[1]

	x_xi = lambda xi: 0.5*(x1+x2) + 0.5*(x2-x1)*xi
	y_eta = lambda eta: 0.5*(y1+y2) + 0.5*(y2-y1)*eta

	func_xi_eta = lambda xi, eta: func(x_xi(xi), y_eta(eta))*(x2-x1)*(y2-y1)*0.5*0.5


	# Perform the gaussian quadrature
	nodes, weights = None, None

	if n == 0:
		raise ValueError("Unsupported Quadrature n value")

	elif n <= CONST_N_MAX:
		nodes, weights = CONST_NODES[n], CONST_WEIGHTS[n]

	else:
		nodes, weights = np.polynomial.legendre.leggauss(n)


	integral_val = 0.
	for i in range(n):
		for j in range(n):
			integral_val += weights[i]*weights[j] * func_xi_eta(nodes[i], nodes[j])

	return integral_val


def integrate_2D_ref_triangle(func, p):
	
	"""Integrate a given function over the standard reference
	triangle domain.
	
	Args:
	    func (function): The function to be integrated over the 2D domain.
	    p (int): Order of exact integration. This will set how many
	    	points are needed for quadrature to get an exact integral for
	    	the specified p. Maximum value of p supported is p = 2.
	"""

	A_ref = 0.5  # Area of the reference triangle

	if p == 1:

		xi_c, eta_c = 1./3., 1./3.
		return A_ref*func(xi_c, eta_c)

	elif p == 2:

		# Get the midpoints of the faces of the triangle

		xi_eta_vals = [
			[0.5, 0.],
			[0., 0.5],
			[0.5, 0.5]
		]

		sum_val = 0.
		for i in range(3):
			sum_val += func(xi_eta_vals[i][0], xi_eta_vals[i][1])

		return (1./3.)*A_ref*sum_val

	else:
		raise ValueError("Unsupported P")




