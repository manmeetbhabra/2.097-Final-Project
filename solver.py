"""
Module: solver.py

Generates the solver for the given partial differential equation. The
equation is solved using the finite element method (the variational 
form of the equation will be solved). 

In this module, the several solver classes exist for the different
equations to be solved. Each solver takes in just the mesh, dofs and 
basis functions and then will set up the system of equations.
"""

import matplotlib.pyplot as plt
import numpy as np
import integrate
import math
import mesh
import node
import tri_finite_element



class PoissonSolver(object):

	"""FEM solver for the Poisson Equation
	"""
	
	def __init__(self, mesh, function_rhs):
		
		"""Constructor for the Poisson equation solver
		
		Args:
		    mesh (TYPE): The mesh object on which we will solve
		    	the given equation.
		    function_rhs (TYPE): Function for the rhs of the 
		    	poisson equation. That is, 
		    		D^2 u = function_rhs
		    	where D^2 is the laplacian operator.
		"""

		self.mesh = mesh
		self.f_rhs = function_rhs

		self.LHS_matrix = None
		self.RHS_vector = None


	def assemble_system(self):
	
		"""Assmble the LHS and RHS for the system of equations
		"""
		
		# Assemble the LHS and RHS for the system of equations
		self.assemble_LHS()
		self.assemble_RHS()


	def assemble_LHS(self):
		
		"""Assemble the bilinear operator's matrix. This is
		the left hand side of the system of equations.

		NOTE: This method currently will work exclusively for triangles.
			Will possibly generalize later to quads.
		"""
		
		# Get the list of degrees of freedom
		dof_list = self.mesh.dof_list

		self.LHS_matrix = np.zeros((len(dof_list), len(dof_list)))

		for i in range(len(dof_list)):
			for j in range(i, len(dof_list)):
				
				dof_i = dof_list[i]
				dof_j = dof_list[j]

				a_psi_i_psi_j = 0.0

				# Get all elements that hold both these degrees of freedom and 
				# compute their contribution to the LHS matrix
				common_elements = self.get_common_elements(dof_i, dof_j)
				for element in common_elements:


					# Get the basis functions and their gradients (defined on the reference
					# domain) for the element.
					psi_hat_i, grad_psi_hat_i = element.get_basis_funcs_given_dof(dof_i)
					psi_hat_j, grad_psi_hat_j = element.get_basis_funcs_given_dof(dof_j)

					psi_hat_i_by_xi = grad_psi_hat_i[0]
					psi_hat_i_by_eta = grad_psi_hat_i[1]

					psi_hat_j_by_xi = grad_psi_hat_j[0]
					psi_hat_j_by_eta = grad_psi_hat_j[1]


					# Get the metric terms on this element. These are lambda expressions
					# defined over the reference domain
					x_xi = element.x_xi
					x_eta = element.x_eta
					y_xi = element.y_xi
					y_eta = element.y_eta
					J = element.J


					# The integrand for the mapped volume integral onto the reference element
					integrand_mapped = lambda xi, eta: \
						((y_eta(xi,eta)*psi_hat_i_by_xi(xi,eta) - y_xi(xi,eta)*psi_hat_i_by_eta(xi,eta)) *\
						(y_eta(xi,eta)*psi_hat_j_by_xi(xi,eta) - y_xi(xi,eta)*psi_hat_j_by_eta(xi,eta)) + \
						(x_xi(xi,eta)*psi_hat_i_by_eta(xi,eta) - x_eta(xi,eta)*psi_hat_i_by_xi(xi,eta)) *\
						(x_xi(xi,eta)*psi_hat_j_by_eta(xi,eta) - x_eta(xi,eta)*psi_hat_j_by_xi(xi,eta)))*\
						(1./J(xi,eta))

					integral_val = integrate.integrate_2D_ref_triangle(integrand_mapped, 2)
					a_psi_i_psi_j += integral_val


				self.LHS_matrix[i,j] = a_psi_i_psi_j
				self.LHS_matrix[j,i] = a_psi_i_psi_j



	def assemble_RHS(self):
		
		"""Assmble the linear operator's vector. This is the RHS
		for the system of equations
		"""

		# Get the list of degrees of freedom
		dof_list = self.mesh.dof_list

		self.RHS_vector = np.zeros((len(dof_list), 1))

		for i in range(len(dof_list)):

			b_psi_i = 0.0

			dof_i = dof_list[i]

			# Get all elements that hold this degree of freedom and 
			# compute their contribution to the RHS vector
			dof_i_element_list = dof_i.element_list
			for element in dof_i_element_list:

				psi_hat_i, grad_psi_hat_i = element.get_basis_funcs_given_dof(dof_i)
				
				# Get the Jacobian and mapping functions
				J = element.J
				x_mapping_func, y_mapping_func = element.x_mapping_func, element.y_mapping_func

				# Map the rhs function onto the reference domain
				f_mapped = lambda xi, eta, x_mapping_func=x_mapping_func, y_mapping_func=y_mapping_func: \
					-1.*self.f_rhs(x_mapping_func(xi,eta), y_mapping_func(xi,eta))

				# Compute the integral contribution
				integrand_mapped = lambda xi, eta, f_mapped=f_mapped: psi_hat_i(xi, eta) * f_mapped(xi, eta) * J(xi,eta)
				integral_val = integrate.integrate_2D_ref_triangle(integrand_mapped, 2)

				b_psi_i += integral_val

			self.RHS_vector[i][0] = b_psi_i


		# Get the contribution to the RHS from the dirichlet BCs
		dirichlet_bc_list = self.mesh.dirichlet_bc_list

		for i in range(len(dof_list)):

			c_psi_i = 0.

			dof_i = dof_list[i]

			for j in range(len(dirichlet_bc_list)):
				# Loop over the dirichlet BC nodes

				dof_j = dirichlet_bc_list[j]

				# Get the elements that hold both dof_i and dof_j (non-zero
				# contribution to the integral will only be from these elements)
				common_elements = self.get_common_elements(dof_i, dof_j)

				c_psi_i_psi_j = 0.0

				for element in common_elements:

					# Get the basis functions and their gradients (defined on the reference
					# domain) for the element.
					psi_hat_i, grad_psi_hat_i = element.get_basis_funcs_given_dof(dof_i)
					psi_hat_j, grad_psi_hat_j = element.get_basis_funcs_given_dof(dof_j)

					psi_hat_i_by_xi = grad_psi_hat_i[0]
					psi_hat_i_by_eta = grad_psi_hat_i[1]

					psi_hat_j_by_xi = grad_psi_hat_j[0]
					psi_hat_j_by_eta = grad_psi_hat_j[1]


					# Get the metric terms on this element. These are lambda expressions
					# defined over the reference domain
					x_xi = element.x_xi
					x_eta = element.x_eta
					y_xi = element.y_xi
					y_eta = element.y_eta
					J = element.J


					# The integrand for the mapped volume integral onto the reference element
					integrand_mapped = lambda xi, eta: \
						((y_eta(xi,eta)*psi_hat_i_by_xi(xi,eta) - y_xi(xi,eta)*psi_hat_i_by_eta(xi,eta)) *\
						(y_eta(xi,eta)*psi_hat_j_by_xi(xi,eta) - y_xi(xi,eta)*psi_hat_j_by_eta(xi,eta)) + \
						(x_xi(xi,eta)*psi_hat_i_by_eta(xi,eta) - x_eta(xi,eta)*psi_hat_i_by_xi(xi,eta)) *\
						(x_xi(xi,eta)*psi_hat_j_by_eta(xi,eta) - x_eta(xi,eta)*psi_hat_j_by_xi(xi,eta)))*\
						(1./J(xi,eta))

					integral_val = integrate.integrate_2D_ref_triangle(integrand_mapped, 2)
					c_psi_i_psi_j += integral_val

				c_psi_i += c_psi_i_psi_j*dof_j.value

			self.RHS_vector[i][0] -= c_psi_i


	def get_common_elements(self, dof_i, dof_j):
		
		"""Get the list of elements that hold both
		dof_i and dof_j as these are the elements that will have
		a non-zero contribution to the rhs
		
		
		Args:
		    dof_i (object): Degree of freedom i
		    dof_j (object): Degree of freedom j
		
		Returns:
		    List: The list of elements that hold both dof_i and 
		    	dof_j. These are the only elements that will have
		    	a non-zero contribution to the LHS
		"""
		
		elem_list_i = dof_i.element_list
		elem_list_j = dof_j.element_list

		common_elements = []
		for elem in elem_list_i:
			if elem in elem_list_j:
				common_elements.append(elem)

		return common_elements


	def solve_system(self):
	
		"""Solve the given system of equations and 
		store the result into the dof values in the mesh.

		NOTE: This function must be called after assemble system.
		"""
	
		# Get the list of degrees of freedom
		dof_list = self.mesh.dof_list

		dof_vals = np.linalg.solve(self.LHS_matrix, self.RHS_vector)

		for i in range(len(dof_list)):
			dof_list[i].set_value(dof_vals[i][0])


	def compute_L2_error(self, u_analytical):
		
		"""Compute the L2 error of the resulting numerical 
		solution. To do this, for now, we will need an analytical solution
		in closed form.
		
		Args:
		    u_analytical (function): Closed form analytical solution
		"""
		
		global_L2_error = 0.0

		# Go through all the elements and compute the L2 error
		for element in self.mesh.element_list:

			psi_hat_basis = element.reference_element.psi_hat_basis
			u_sol_nodal_vals = [x.value for x in element.node_list]

			u_h = lambda xi, eta: element.basis_expansion(psi_hat_basis, u_sol_nodal_vals, xi, eta)

			# Map the analytical solution to the reference domain
			x_mapping_func, y_mapping_func = element.x_mapping_func, element.y_mapping_func

			# Map the rhs function onto the reference domain
			u_exact_mapped = lambda xi, eta, x_mapping_func=x_mapping_func, y_mapping_func=y_mapping_func: \
				u_analytical(x_mapping_func(xi,eta), y_mapping_func(xi,eta))

			# Get the jacobian of the transformation
			J = element.J

			e_integrand_mapped = lambda xi, eta, u_exact_mapped=u_exact_mapped, u_h=u_h: \
				((u_h(xi, eta) - u_analytical(xi, eta))**2.) * J(xi,eta)
			
			L2_error_integral_val = integrate.integrate_2D_ref_triangle(e_integrand_mapped, 2)

			# Set the error value in the element
			element.local_L2_error = L2_error_integral_val

			global_L2_error += L2_error_integral_val


		return math.sqrt(global_L2_error/float(len(self.mesh.element_list)))


	def compute_Linf_error(self, u_analytical):
		
		"""Compute the Linf error of the resulting numerical 
		solution. To do this, for now, we will need an analytical solution
		in closed form.
		
		Args:
		    u_analytical (function): Closed form analytical solution
		"""
		
		global_Linf_error = 0.0

		# Go through all the elements and compute the L2 error
		for element in self.mesh.element_list:

			psi_hat_basis = element.reference_element.psi_hat_basis
			u_sol_nodal_vals = [x.value for x in element.node_list]

			u_h = lambda xi, eta: element.basis_expansion(psi_hat_basis, u_sol_nodal_vals, xi, eta)

			# Map the analytical solution to the reference domain
			x_mapping_func, y_mapping_func = element.x_mapping_func, element.y_mapping_func

			# Map the rhs function onto the reference domain
			u_exact_mapped = lambda xi, eta, x_mapping_func=x_mapping_func, y_mapping_func=y_mapping_func: \
				u_analytical(x_mapping_func(xi,eta), y_mapping_func(xi,eta))

			# Get the jacobian of the transformation
			J = element.J

			e_integrand_mapped = lambda xi, eta, u_exact_mapped=u_exact_mapped, u_h=u_h: \
				(abs(u_h(xi, eta) - u_analytical(xi, eta))) * J(xi,eta)
			
			Linf_error_integral_val = integrate.integrate_2D_ref_triangle(e_integrand_mapped, 2)

			# Set the error value in the element
			element.local_Linf_error = Linf_error_integral_val

			global_Linf_error = max(Linf_error_integral_val, global_Linf_error)

		return global_Linf_error


	def compute_L2_error_indicator(self):
		
		"""Summary
		"""
		
		max_local_L2_error = None

		for element in self.mesh.element_list:

			if max_local_L2_error is None:
				max_local_L2_error = element.local_L2_error
				continue

			max_local_L2_error = max(max_local_L2_error, element.local_L2_error)


		#print "max_local_L2_error : %e" % (max_local_L2_error)

		# Compute the error indicator for each element
		for element in self.mesh.element_list:
			element.error_indicator = float(element.local_L2_error)/max_local_L2_error

			# print "element : %s" % element
			# print "\t local_L2_error : %e" % (element.local_L2_error) 
			# print "\t error_indicator : %e" % (element.error_indicator)


	def compute_implicit_error_indicator(self):
		
		"""Compute the error indicator from Babuska's work
		where we solve an auxilliary problem and then compute the
		error indicator using it.
		"""

		# Loop only through the interior degrees of freedom

		for dof in self.mesh.dof_list:
			self.compute_implicit_error_indicator_psi_i_support(dof)



	def compute_implicit_error_indicator_psi_i_support(self, dof_i):
		
		"""Compute the integral over the support of a given
		basis function. We do this by creating a domain around this
		degree of freedom. Then, once a mesh object has been created,
		refine the elements and solve the auxiliary problem. With this,
		finally compute the integral to obtain the error indicator value
		
		Args:
		    dof_i (obj): The degree of freedom to identify the basis function
		    	whose support we are interested in.
		"""

		# Get the elements from this degree of freedom
		support_elements = dof_i.element_list

		# ================================
		#           Node Copies
		# ================================

		# Make copies of the elements and their nodes. We use copies as we
		# do not want to overwrite memory corresponding to the original mesh.
		# The list here is a list of lists, as it will hold the copy of the node
		# and the original (so we can easily identify nodes).

		element_node_copies = []

		for element in support_elements:

			for node_original in element.node_list:

				# Check if the node is in the node copies list (look at the originals)
				# and, if not, create a copy
				element_node_original_list = [x[1] for x in element_node_copies]
				if node_original not in element_node_original_list:
					node_copy = node.Node(node_original.node_type, node_original.x, node_original.y)
					node_copy.value = node_original.value
					element_node_copies.append([node_copy, node_original])


		# # Print the copied nodes for testing
		# for n_pair in element_node_copies:
		# 	print n_pair[0]

		# ================================
		#          Element Copies
		# ================================

		# Create the copies of the elements using the copied nodes

		element_copies = []

		element_node_original_list = [x[1] for x in element_node_copies]
		element_node_copies_list = [x[0] for x in element_node_copies]

		for element in support_elements:

			# List of copy nodes that make up this copy element
			element_copy_node_list = []
			
			for n in element.node_list:
				i = element_node_original_list.index(n)
				element_copy_node_list.append(element_node_copies_list[i])

			# Create the copied element
			element_copy = tri_finite_element.TriFiniteElement(1, element_copy_node_list)

			element_copies.append(element_copy)


		# Set the element connectivity
		for elem_i in element_copies:
			for elem_j in element_copies:
				if elem_i != elem_j:
					tri_finite_element.TriFiniteElement.set_element_connectivity(elem_i, elem_j)


		# Set all outer nodes to be dirichlet bc
		for elem_copy in element_copies:

			for f in range(3):
				if elem_copy.neighbors[f] is None:
					for n in elem_copy.get_nodes_on_face(f):
						n.node_type = "dirichlet_bc"


		# ================================
		#          Suport Mesh
		# ================================

		# Create a mesh object for the domain of interest which spans the 
		# support of the given node's basis function. Fill the data manually
		# using the copied information
		dof_support_mesh = mesh.TriMeshGMSH(1, {"dirichlet_bc_flag" : None}, None)

		dof_support_mesh.element_list = element_copies
		dof_support_mesh.set_node_to_element_connectivity()
		dof_support_mesh.set_dof_list()
		dof_support_mesh.set_dirichlet_bc_list()

		# # Plot the support mesh
		# plt.figure()
		# dof_support_mesh.plot_mesh(False)
		# plt.title(r"Unrefined $\Omega_i$")
		# plt.savefig("../Results/unrefined_supp_domain.pdf")


		# ================================
		#       Refine Support Mesh
		# ================================

		# Refine all elements on the support mesh

		# Collect all the red elements and refine all those that remain red after each
		# element is refined
		element_red_list = []
		for elem in dof_support_mesh.element_list:
			if elem not in element_red_list:
				if elem.element_refinement_info["color"] == "red":
					element_red_list.append(elem)

		for elem_red in element_red_list:
			if elem_red in dof_support_mesh.element_list:
				dof_support_mesh.h_refine_mesh([elem_red])


		# Collect now all the green elements and refine them sequentially
		element_green_list = []
		for elem in dof_support_mesh.element_list:
			if elem not in element_green_list:
				if elem.element_refinement_info["color"] == "green":
					element_green_list.append(elem)

		for elem_green in element_green_list:
			if elem_green in dof_support_mesh.element_list:
				dof_support_mesh.h_refine_mesh([elem_green])


		# Mark all the nodes on the boundary as dirichlet points
		for elem in dof_support_mesh.element_list:

			for face_i in range(3):
				if elem.neighbors[face_i] is None:
					
					# Have found a boundary face. Set the nodes
					# on it to be dirichlet nodes

					face_node_list = elem.get_nodes_on_face(face_i)

					for n in face_node_list:
						n.node_type = "dirichlet_bc"


		# plt.figure()
		# dof_support_mesh.plot_mesh(False)
		# plt.title(r"Refined $\Omega_i$")
		# plt.savefig("../Results/refined_supp_domain.pdf")

		# We need to now set the values of the dirichlet_bc nodes that have
		# a value of None. Do this by finding what element belongs to each
		# dirichlet bc node, finding the other dirichlet bc nodes on the
		# neighboring elements and taking the average of the values

		dof_support_mesh.set_dirichlet_bc_list()
		for dirichlet_node in dof_support_mesh.dirichlet_bc_list:

			if dirichlet_node.value is None:


				dirichlet_support_element_dirichlet_nodes = []

				for elem in dirichlet_node.element_list:
					for n in elem.node_list:
						if n.node_type == "dirichlet_bc" and n.value is not None:
							if n not in dirichlet_support_element_dirichlet_nodes:
								dirichlet_support_element_dirichlet_nodes.append(n)

				# if self.mesh.dof_list.index(dof_i) == 4:
				# 	print "None dirichlet_node : %s" % (dirichlet_node)
				# 	print "\t neighbor nodes:"
				# 	for n in dirichlet_support_element_dirichlet_nodes:
				# 		print "\t %s -> value : %s" % (n, n.value)

				if len(dirichlet_support_element_dirichlet_nodes) < 2:
					raise ValueError("Did not find enough neighboring dirichlet nodes")

				if len(dirichlet_support_element_dirichlet_nodes) > 2:
					# If we found more than 2 nodes, we need to do some checks here.
					# For now, place a temporary solution where we use the closest
					# nodes for the averaging data

					nodes_and_r_list = []
					for n in dirichlet_support_element_dirichlet_nodes:
						r = math.sqrt((n.x - dirichlet_node.x)**2. + (n.y - dirichlet_node.y)**2.)
						nodes_and_r_list.append([n, r])
					
					# Sort the list according to distance and take the minimum two
					nodes_and_r_list = sorted(nodes_and_r_list, key= lambda x: x[1])

					dirichlet_support_element_dirichlet_nodes = [
						nodes_and_r_list[0][0], nodes_and_r_list[1][0]
					]

				avg_node_val = 0.5 * (dirichlet_support_element_dirichlet_nodes[0].value + \
					dirichlet_support_element_dirichlet_nodes[1].value)

				dirichlet_node.set_value(avg_node_val)




		# ================================
		#     Solve Auxiliary Problem
		# ================================

		dof_support_mesh.set_node_to_element_connectivity()
		dof_support_mesh.set_dof_list()
		dof_support_mesh.set_dirichlet_bc_list()

		# Solve for the numerical solution (auxiliary problem)
		# on this support mesh
		poisson_solver = PoissonSolver(dof_support_mesh, self.f_rhs)
		poisson_solver.assemble_system()
		poisson_solver.solve_system()


		# # Visualize the solution
		# x_vals, y_vals, z_vals = [], [], []

		# for n in dof_support_mesh.dof_list:
		# 	x_vals.append(n.x)
		# 	y_vals.append(n.y)
		# 	z_vals.append(n.value)

		# for n in dof_support_mesh.dirichlet_bc_list:
		# 	x_vals.append(n.x)
		# 	y_vals.append(n.y)
		# 	z_vals.append(n.value)


		# plt.figure()
		# plt.tripcolor(x_vals, y_vals, z_vals, cmap=plt.get_cmap("jet"))
		# plt.colorbar()


		# ================================
		#     	  Set the Error
		# ================================

		# Go through the refined support mesh and set the error values at each node
		# with the unrefined mesh. Then, use this error field, and the energy norm,
		# to compute the error indicator

		for elem_copy in dof_support_mesh.element_list:
			# Loop through all the elements on the support mesh

			for node_copy in elem_copy.node_list:

				if node_copy.node_type == "dof":

					node_copy_x, node_copy_y = node_copy.x, node_copy.y
					node_copy_u_hat_val = None  # Value on the original mesh at this node

					# Loop through all the elements in the original mesh. Check if the given
					# (x,y) point is in the given element and, if so, compute the approximation
					# at that point

					for elem_original in dof_i.element_list:
						if elem_original.mapping_physical_to_reference_domain(node_copy_x, node_copy_y)[1]:
							node_copy_u_hat_val = elem_original.get_solution_at_position(node_copy_x, node_copy_y)

					if node_copy_u_hat_val is None:
						raise ValueError("Did not find approximate solution at node : %s, %s" % (node_copy))

					node_copy.set_error_value(node_copy.value - node_copy_u_hat_val)

				else:
					# For dirichlet nodes, the approximate solution and auxiliary solution are the same
					node_copy.set_error_value(0.0)


		# # Visualize the error
		# x_vals, y_vals, z_vals = [], [], []

		# for n in dof_support_mesh.dof_list:
		# 	x_vals.append(n.x)
		# 	y_vals.append(n.y)
		# 	z_vals.append(n.error_value)

		# for n in dof_support_mesh.dirichlet_bc_list:
		# 	x_vals.append(n.x)
		# 	y_vals.append(n.y)
		# 	z_vals.append(n.error_value)

		# plt.figure()
		# plt.tripcolor(x_vals, y_vals, z_vals, cmap=plt.get_cmap("jet"))
		# plt.colorbar()


		# ================================
		#   Compute the Error Indicator
		# ================================

		# Use the energy norm to get the value of the error indicator
		# over the given support domain

		eta_i_squared = 0.0

		for element in dof_support_mesh.element_list:
			# Loop through all the mesh elements and compute their contribution
			# to the energy norm squared value

			# Get the basis functions and their gradients defined on the reference domain
			psi_hat_basis = element.reference_element.psi_hat_basis
			grad_psi_hat_basis = element.reference_element.grad_psi_hat_basis
			del_psi_hat_del_xi_basis = [x[0] for x in grad_psi_hat_basis]
			del_psi_hat_del_eta_basis = [x[1] for x in grad_psi_hat_basis]

			e_sol_nodal_vals = [x.error_value for x in element.node_list]

			# The error field (and the field's gradient)
			e_h = lambda xi, eta: element.basis_expansion(psi_hat_basis, e_sol_nodal_vals, xi, eta)
			del_e_h_del_xi = lambda xi, eta: element.basis_expansion(del_psi_hat_del_xi_basis, e_sol_nodal_vals, xi, eta)
			del_e_h_del_eta = lambda xi, eta: element.basis_expansion(del_psi_hat_del_eta_basis, e_sol_nodal_vals, xi, eta)

			# Get the metric terms
			x_xi = element.x_xi
			x_eta = element.x_eta
			y_xi = element.y_xi
			y_eta = element.y_eta
			J = element.J

			# Integrand for the energy norm
			e_integrand_mapped = lambda xi, eta: \
				((y_eta(xi,eta)*del_e_h_del_xi(xi,eta) - y_xi(xi,eta)*del_e_h_del_eta(xi,eta)) *\
				(y_eta(xi,eta)*del_e_h_del_xi(xi,eta) - y_xi(xi,eta)*del_e_h_del_eta(xi,eta)) + \
				(x_xi(xi,eta)*del_e_h_del_eta(xi,eta) - x_eta(xi,eta)*del_e_h_del_xi(xi,eta)) *\
				(x_xi(xi,eta)*del_e_h_del_eta(xi,eta) - x_eta(xi,eta)*del_e_h_del_xi(xi,eta)))*\
				(1./J(xi,eta))

			e_integral_val = integrate.integrate_2D_ref_triangle(e_integrand_mapped, 2)
			eta_i_squared += e_integral_val

		eta_i = math.sqrt(eta_i_squared)


		# Set the error indicator for all the elements. We will simply
		# assign the maximum indicator to each element (since each element
		# will in general have multiple basis functions on its support and
		# as such it will have multiple error indicators)

		for element in dof_i.element_list:

			if element.error_indicator is None:
				element.error_indicator = eta_i
			else:
				element.error_indicator = max(element.error_indicator, eta_i)



	def set_hanging_nodes(self):
		
		"""Set the values of the hanging nodes. Use the
		constraint equations to constrain the values of the nodes
		based on the constraint nodes. 

		We are working only with bilinear basis functions (P=1) so
		the value at hanging nodes is just given by the average of 
		the surrounding nodes.
		"""
		
		hanging_nodes = self.mesh.hanging_node_list

		for hanging_node in hanging_nodes:
			hanging_node.set_value(0.5*(hanging_node.hanging_node_constraint_nodes[0].value + \
				hanging_node.hanging_node_constraint_nodes[1].value))





class HelmholtzSolver(object):
	pass


