"""
Module: reference_element.py

Holds the data structure that will have the finite element reference element
information
"""


import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy


class QuadReferenceElement(object):

	"""Holds the reference element information for quads. Each finite
	element on the physical space will point to a given reference
	element. Note that the basis functions are defined on this space.
	"""
	
	def __init__(self, p):
		
		"""Constructor for the reference element
		
		Args:
		    p (int): Order of the basis functions to be used (are nodal)
		"""

		self.p = p
		self.num_psi = (self.p+1)**2  # Number of psi basis functions

		self.psi_hat_basis = self.__get_psi_hat_basis_functions()

		self.grad_psi_hat_basis = self.__get_grad_psi_hat_basis_functions()


	def plot_basis_function(self, basis_i, basis_j, basis_type):
		
		"""Plot a specified basis function to visualize it. Plot either the 
		basis function or its gradient.
		
		Args:
		    basis_i (int): i Index for which basis function to plot
		    basis_j (int): j Index for which basis function to plot
		    basis_type (str): Options are "basis" if the basis function is to be plotted
		    	and "grad_basis" if the gradient of the basis is to be plotted.
		"""

		n_plot_pts = 100

		xi_vals = numpy.linspace(-1., 1., n_plot_pts)
		eta_vals = numpy.linspace(-1., 1., n_plot_pts)

		xi_vals, eta_vals = numpy.meshgrid(xi_vals, eta_vals)

		if basis_type == "basis":

			phi_vals = numpy.zeros((n_plot_pts, n_plot_pts))
			
			for i in range(n_plot_pts):
				for j in range(n_plot_pts):
					phi_vals[i,j] = self.psi_hat_basis[basis_i][basis_j](xi_vals[i,j], eta_vals[i,j])

			plt.figure()
			ax = plt.gcf().gca(projection='3d')
			surf = ax.plot_surface(xi_vals, eta_vals, phi_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			plt.xlabel(r"$\xi$")
			plt.ylabel(r"$\eta$")
			

		elif basis_type == "grad_basis":
			
			phi_vals_xi = numpy.zeros((n_plot_pts, n_plot_pts))
			phi_vals_eta = numpy.zeros((n_plot_pts, n_plot_pts))
			
			for i in range(n_plot_pts):
				for j in range(n_plot_pts):
					phi_vals_xi[i,j] = self.grad_psi_hat_basis[basis_i][basis_j][0](xi_vals[i,j], eta_vals[i,j])
					phi_vals_eta[i,j] = self.grad_psi_hat_basis[basis_i][basis_j][1](xi_vals[i,j], eta_vals[i,j])

			plt.figure()
			ax = plt.gcf().gca(projection='3d')
			surf = ax.plot_surface(xi_vals, eta_vals, phi_vals_xi, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			plt.xlabel(r"$\xi$")
			plt.ylabel(r"$\eta$")
			plt.title("xi Derivative")

			plt.figure()
			ax = plt.gcf().gca(projection='3d')
			surf = ax.plot_surface(xi_vals, eta_vals, phi_vals_eta, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			plt.xlabel(r"$\xi$")
			plt.ylabel(r"$\eta$")
			plt.title("eta Derivative")

		else:
			raise ValueError("Unsupported")



	# TODO: Modify the ordering of the basis functions to be consistent
	# for higher order p values

	def __get_psi_hat_basis_functions(self):
		
		"""Get the psi basis functions. These are the nodal basis functions, where
		the number of nodes is given by the order. These basis functions are 
		defined on the reference domain.

		The ordering the basis functions in the matrix returned is consistent
		with the ordering of the nodes. i increases from left to right and j from
		bottom to top. Then, all j = 0 are first placed, then all j = 1, ...
		just like the ordering of the nodes.

		NOTE: For now, we analytically compute the basis functions for the P = 1 case.
		NOTE: We only allow the P = 1 case for now.

		Returns:
		    List: List of lists with the lambda expressions for the basis functions on the 
		    	reference domain. Ordering is consistent with what is described above.
		"""

		if self.p == 1:

			# First order basis functions

			psi_00 = lambda xi, eta : 0.25 * (1. - xi) * (1. - eta)
			psi_10 = lambda xi, eta : 0.25 * (1. + xi) * (1. - eta)
			psi_01 = lambda xi, eta : 0.25 * (1. - xi) * (1. + eta)
			psi_11 = lambda xi, eta : 0.25 * (1. + xi) * (1. + eta)

			#return [psi_00, psi_10, psi_01, psi_11]

			return [
				[psi_00, psi_01],
				[psi_10, psi_11]
			]

		else:

			# P > 1
			# Ordering of the basis functions is all j = 1, then all j = 2, ...
			# Lagrange basis functions are used, with equidistant nodes

			raise ValueError("Unsupported P")


	def __get_grad_psi_hat_basis_functions(self):
		
		"""Obtain the gradient of the nodal basis functions. Ordering is consisted
		with the ordering of the basis functions psi_hat.
		
		Returns:
		    List: List of lists with the lambda expressions for the gradient basis functions 
		    	on the reference domain. Ordering is consistent with what is described in psi_hat_basis.
		    	Each entry of the list is a list of the form [dpsi_hat/dxi, dpsi_hat/deta]. 
		"""
		
		if self.p == 1:

			# First order basis functions

			grad_psi_00 = [lambda xi, eta : - 0.25 * (1. - eta),
						  lambda xi, eta : - 0.25 * (1. - xi)]
			grad_psi_10 = [lambda xi, eta :   0.25 * (1. - eta),
						  lambda xi, eta : - 0.25 * (1. + xi)]
			grad_psi_01 = [lambda xi, eta : - 0.25 * (1. + eta),
						  lambda xi, eta :   0.25 * (1. - xi)]
			grad_psi_11 = [lambda xi, eta :   0.25 * (1. + eta),
						  lambda xi, eta :   0.25 * (1. + xi)]

			#return [grad_psi_1, grad_psi_2, grad_psi_3, grad_psi_4]

			return [
				[grad_psi_00, grad_psi_01],
				[grad_psi_10, grad_psi_11]
			]

		else: 
			
			# P > 1
			# Compute the derivative of the lagrange basis functions numerically

			raise ValueError("Unsupported P")



class TriReferenceElement(object):

	"""Holds the reference element information for tris. Each finite
	element on the physical space will point to a given reference
	element. Note that the basis functions are defined on this space.
	"""
	
	def __init__(self, p):
		
		"""Constructor for the reference element.

		NOTE: We require p=1 for now. Will possibly generalize this
			in the future.
		
		Args:
		    p (int): Order of the basis functions to be used (are nodal)
		"""

		if p != 1:
			raise ValueError("Unsupported P Value for Triangles")

		self.p = 1
		self.num_psi = 3  # Number of psi basis functions

		self.psi_hat_basis = self.__get_psi_hat_basis_functions()

		self.grad_psi_hat_basis = self.__get_grad_psi_hat_basis_functions()


	def point_in_reference_triangle(self, xi, eta):
		
		"""Check whether a given point is in the reference triangle.
		
		Args:
		    xi (float): Description
		    eta (float): Description
		
		Returns:
		    boolean: True if the point is in the triangle and false otherwise
		"""

		if xi >= 0 and eta >= 0 and (xi + eta) <= 1:
			return True

		return False



	def plot_basis_function(self, basis_i, basis_type):
		
		"""Plot a specified basis function to visualize it. Plot either the 
		basis function or its gradient.

		NOTE: Although a square domain is given, plot the basis on the domain
			of the reference triangle only (outside of the triangle, set
			the value of the basis and its gradients to be 0)
		
		Args:
		    basis_i (int): i Index for which basis function to plot
		    basis_type (str): Options are "basis" if the basis function is to be plotted
		    	and "grad_basis" if the gradient of the basis is to be plotted.
		"""

		n_plot_pts = 100

		xi_vals = numpy.linspace(-1., 1., n_plot_pts)
		eta_vals = numpy.linspace(-1., 1., n_plot_pts)

		xi_vals, eta_vals = numpy.meshgrid(xi_vals, eta_vals)

		if basis_type == "basis":

			phi_vals = numpy.zeros((n_plot_pts, n_plot_pts))
			
			for i in range(n_plot_pts):
				for j in range(n_plot_pts):
					xi_ij, eta_ij = xi_vals[i,j], eta_vals[i,j]
					
					if self.point_in_reference_triangle(xi_ij, eta_ij):
						phi_vals[i,j] = self.psi_hat_basis[basis_i](xi_ij, eta_ij)
					else:
						phi_vals[i,j] = 0

			plt.figure()
			#ax = plt.gcf().gca(projection='3d')
			#surf = ax.plot_surface(xi_vals, eta_vals, phi_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			plt.contourf(xi_vals, eta_vals, phi_vals, 100, cmap=plt.get_cmap("jet"))
			plt.colorbar()
			plt.xlabel(r"$\xi$")
			plt.ylabel(r"$\eta$")

			plt.xlim([-0.1,1.1])
			plt.ylim([-0.1,1.1])

		elif basis_type == "grad_basis":
			
			phi_vals_xi = numpy.zeros((n_plot_pts, n_plot_pts))
			phi_vals_eta = numpy.zeros((n_plot_pts, n_plot_pts))
			
			for i in range(n_plot_pts):
				for j in range(n_plot_pts):

					xi_ij, eta_ij = xi_vals[i,j], eta_vals[i,j]
					
					if self.point_in_reference_triangle(xi_ij, eta_ij):
						phi_vals_xi[i,j] = self.grad_psi_hat_basis[basis_i][0](xi_ij, eta_ij)
						phi_vals_eta[i,j] = self.grad_psi_hat_basis[basis_i][1](xi_ij, eta_ij)
					else:
						phi_vals_xi[i,j] = 0
						phi_vals_eta[i,j] = 0


			plt.figure()
			#ax = plt.gcf().gca(projection='3d')
			#surf = ax.plot_surface(xi_vals, eta_vals, phi_vals_xi, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			plt.contourf(xi_vals, eta_vals, phi_vals_xi, 100, cmap=plt.get_cmap("jet"))
			plt.colorbar()
			plt.xlabel(r"$\xi$")
			plt.ylabel(r"$\eta$")
			plt.title("xi Derivative")

			plt.xlim([-0.1,1.1])
			plt.ylim([-0.1,1.1])

			plt.figure()
			#ax = plt.gcf().gca(projection='3d')
			#surf = ax.plot_surface(xi_vals, eta_vals, phi_vals_eta, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			plt.contourf(xi_vals, eta_vals, phi_vals_eta, 100, cmap=plt.get_cmap("jet"))
			plt.colorbar()
			plt.xlabel(r"$\xi$")
			plt.ylabel(r"$\eta$")
			plt.title("eta Derivative")

			plt.xlim([-0.1,1.1])
			plt.ylim([-0.1,1.1])

		else:
			raise ValueError("Unsupported")



	# TODO: Modify the ordering of the basis functions to be consistent
	# for higher order p values

	def __get_psi_hat_basis_functions(self):
		
		"""Get the psi basis functions. These are the nodal basis functions defined on the
		triangle. The ordering of the basis functions is given by

			3
			|\
			| \
			|  \
			|   \
		   1 ---- 2

		where the triangle shown is on the reference domain (the standard P=1 reference triangle).
		This ordering is consistent throughout the code.

		NOTE: For now, we analytically compute the basis functions for the P = 1 case.
		NOTE: We only allow the P = 1 case for now.

		Returns:
		    List: List with the lambda expressions for the basis functions on the 
		    	reference domain. Ordering is consistent with what is described above.
		"""

		if self.p == 1:

			# First order basis functions

			psi_1 = lambda xi, eta : 1. - xi - eta
			psi_2 = lambda xi, eta : xi
			psi_3 = lambda xi, eta : eta

			return [psi_1, psi_2, psi_3]

		else:

			# P > 1
			# Ordering of the basis functions is all j = 1, then all j = 2, ...
			# Lagrange basis functions are used, with equidistant nodes

			raise ValueError("Unsupported P")


	def __get_grad_psi_hat_basis_functions(self):
		
		"""Obtain the gradient of the nodal basis functions. Ordering is consisted
		with the ordering of the basis functions psi_hat.
		
		Returns:
		    List: List with the lambda expressions for the gradient basis functions 
		    	on the reference domain. Ordering is consistent with what is described in psi_hat_basis.
		    	Each entry of the list is a list of the form [dpsi_hat/dxi, dpsi_hat/deta]. 
		"""
		
		if self.p == 1:

			# First order basis functions

			grad_psi_1 = [lambda xi, eta : -1.,
						  lambda xi, eta : -1.]
			grad_psi_2 = [lambda xi, eta :  1.,
						  lambda xi, eta :  0.]
			grad_psi_3 = [lambda xi, eta :  0.,
						  lambda xi, eta :  1.]

			return [grad_psi_1, grad_psi_2, grad_psi_3]

		else: 
			
			# P > 1
			# Compute the derivative of the lagrange basis functions numerically

			raise ValueError("Unsupported P")




