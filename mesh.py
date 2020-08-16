"""
Module: mesh.py

Generates the mesh on which the equations will be solved
"""


import math
import node
import quad_finite_element
import tri_finite_element
import matplotlib.pyplot as plt
import numpy as np
import vertex


class QuadMeshRectangular(object):

	"""Class to hold the quadrilateral mesh. These meshes will be generated
	on a rectangular domain. Moreover, 
	"""
	
	def __init__(self, num_i_elem, num_j_elem, p, mesh_properties):
		
		"""Constructor for the Quadrilateral Mesh
		
		Will generate a quadrilateral mesh (rectangular) of certain order
		elements. The mesh will initially be "structured" however it will
		be stored in memory as an unstructured mesh. Then, we will work
		with the unstructured mesh to do all solving and adaptation.
		
		Args:
		    num_i_elem (int): Number of elements to initialize along the i direction
		    num_j_elem (int): Number of elements to initialize along the j direction
		    p (int): The order of the basis functions
		    mesh_properties (dict): Dictionary holding the following parameters
		    	"x_range": List with the lower and upper x limits of the mesh (for rectangular case)
		    	"y_range": List with the lower and upper y limits of the mesh (for rectangular case)
		    	"dirichlet_bc_flag": The function for setting the dirichlet bc
		"""


		# ===========================
		#            Setup
		# ===========================

		# x and y limits of the rectangular mesh
		self.x_range = mesh_properties.get("x_range")
		self.y_range = mesh_properties.get("y_range")

		# Order of the mesh
		self.p = p


		# Generate the initial mesh by creating equal dimension elements on a
		# num_i_elem x num_j_elem grid.

		# Spacing of the elements
		dx_elem, dy_elem =  float(self.x_range[1] - self.x_range[0])/num_i_elem, \
							float(self.y_range[1] - self.y_range[0])/num_j_elem



		# ===========================
		#      Vertices Generation
		# ===========================

		# Generate the grid of vertex points
		vertex_points_grid = []

		for i in range(num_i_elem+1):
			col = []
			for j in range(num_j_elem+1):

				x_pt, y_pt = i*dx_elem, j*dy_elem
				vertex_pt = vertex.Vertex(x_pt, y_pt)

				col.append(vertex_pt)

			vertex_points_grid.append(col)



		# ===========================
		#   Element/DOF Generation
		# ===========================

		# As we are initially creating a structured mesh, exploit this fact
		# to be able to quickly determine the connectivity between the elements
		element_matrix = []

		for i in range(num_i_elem):
			col = []
			for j in range(num_j_elem):

				# Get the range of the element on the physical domain
				elem_x_range = [i*dx_elem + self.x_range[0], (i+1)*dx_elem + self.x_range[0]]
				elem_y_range = [j*dy_elem + self.y_range[0], (j+1)*dy_elem + self.y_range[0]]

				# The matrix of points for the vertices for this element
				vertex_pt_matrix = [[None, None],[None, None]]
				for i_vertex in range(2):
					for j_vertex in range(2):
						vertex_pt_matrix[i_vertex][j_vertex] = vertex_points_grid[i+i_vertex][j+j_vertex]

				# Create the element with these vertices
				col.append(quad_finite_element.QuadFiniteElement(self.p, elem_x_range, elem_y_range, vertex_pt_matrix, 0))

			element_matrix.append(col)

		
		# Set the neigbors of the elements
		for i in range(num_i_elem):
			for j in range(num_j_elem):

				elem_ij = element_matrix[i][j]

				if i == 0:
					elem_ij.set_neighbor([element_matrix[i+1][j]], 1, [3])
				elif i == num_i_elem-1:
					elem_ij.set_neighbor([element_matrix[i-1][j]], 3, [1])
				else:
					elem_ij.set_neighbor([element_matrix[i+1][j]], 1, [3])
					elem_ij.set_neighbor([element_matrix[i-1][j]], 3, [1])


				if j == 0:
					elem_ij.set_neighbor([element_matrix[i][j+1]], 2, [0])
				elif j == num_j_elem-1:
					elem_ij.set_neighbor([element_matrix[i][j-1]], 0, [2])
				else:
					elem_ij.set_neighbor([element_matrix[i][j+1]], 2, [0])
					elem_ij.set_neighbor([element_matrix[i][j-1]], 0, [2])


		# Now, with all the neighbors set, we place all the elements in a list
		self.element_list = []
		for i in range(num_i_elem):
			for j in range(num_j_elem):
				self.element_list.append(element_matrix[i][j])


		# Set the dofs for the elements. Exploit the structured data for now to
		# create the shared dofs
		node_points_grid = []
		for i in range(num_i_elem+1):
			col = []
			
			for j in range(num_j_elem+1):
				x_pt, y_pt = i*dx_elem, j*dy_elem
				node_pt = node.Node("dof", x_pt, y_pt)
				col.append(node_pt)
			
			node_points_grid.append(col)


		for i in range(num_i_elem):	
			for j in range(num_j_elem):

				elem_ij = element_matrix[i][j]

				i_dof_min = i
				j_dof_min = j

				for i_node_elem in range(2):
					for j_node_elem in range(2):
						
						elem_ij.node_matrix[i_node_elem][j_node_elem] = \
							node_points_grid[i_dof_min + i_node_elem][j_dof_min + j_node_elem]


		self.set_node_to_element_connectivity()

		# Summary: 
		# - At this point in the mesh generation, we now have a list of
		# 	elements. This list makes it such that we can effectively handle
		# 	an unstructured mesh of elements.
		# - All nodes have been generated


		# ===========================
		#      Dirichlet BC Setup 
		# ===========================

		# Set the Dirichlet BC on the mesh and its nodes
		self.dirichlet_bc_flag_function = mesh_properties["dirichlet_bc_flag"]
		self.set_dirichlet_boundary_condition()


		# ===========================
		#      Global DOF List
		# ===========================

		# Set the list of dofs for this mesh. This will be used for the system solve
		self.dof_list = []
		self.set_dof_list()

		self.dirichlet_bc_list = []
		self.set_dirichlet_bc_list()

		# Set the list of hanging nodes for this mesh. Will be needed during h-refinement
		self.hanging_node_list = []
		self.set_hanging_node_list()


		# Recursive back tracking when refining
		# Keep going out (like a depth first search) to find elements that are not
		# on the same refinement level. Then refine them (starting from the end)



	def mesh_element_leaf_collector(self, element, leaf_list):
		
		"""Collect all the leaves for the element tree starting
		at a given root. Get the elements recursively
		
		Args:
		    element (obj): The root point to start collecting elements
		    leaf_list (list): The list of leaves. Elements will be appended on
		"""

		if len(element.children) == 0:
			leaf_list.append(element)

		else:
			for i in range(2):
				for j in range(2):
					self.mesh_element_leaf_collector(element.children[i][j], leaf_list)


	def set_dirichlet_boundary_condition_element(self, element):
		
		"""Recursively move through the element to go its
		children (leaves) and set the dirichlet boundary condition.
		"""

		if len(element.children) == 0:
			# Base Case

			# Loop over the nodes
			for i in range(2):
				for j in range(2):

					bool_val, dirichlet_bc_val = self.dirichlet_bc_flag_function(element.node_matrix[i][j].x, 
						element.node_matrix[i][j].y)

					if bool_val:
						element.node_matrix[i][j].set_type("dirichlet_bc")
						element.node_matrix[i][j].set_value(dirichlet_bc_val)

		else:

			# Loop over the children
			for i in range(2):
				for j in range(2):
					self.set_dirichlet_boundary_condition_element(element.children[i][j])


	def set_dirichlet_boundary_condition(self):
		
		"""Flag all nodes that have a dirichlet boundary condition. 
		Do this by inputing the x and y coordinate into the function and
		then seeing the output. The function should return a tuple with the
		boolean (for whether or not this node has a dirichlet bc) and the 
		value of the bc (None if it is not on a dirichlet boundary)
		"""

		# Loop through the elements and apply the dirichlet BC on the nodes
		# for each element. Note, we will go through elements possibly
		# multiple times since dofs are shared

		for elem in self.element_list:
			self.set_dirichlet_boundary_condition_element(elem)


	def clear_node_to_element_connectivity_element(self, element):
		
		"""Used to recursively go through the leaves of the refined mesh
		and clear the node element lists. 
		
		Args:
		    element (obj): The element of interest. We will clear the element lists
		    	for the nodes for this element.
		"""

		if len(element.children) == 0:

			# Base Case

			# Loop through the nodes
			for i in range(2):
				for j in range(2):
					element.node_matrix[i][j].clear_element_list()

		else:

			# Loop through the children
			for i in range(2):
				for j in range(2):
					self.clear_node_to_element_connectivity_element(element.children[i][j])


	def set_node_to_element_connectivity_element(self, element):
		
		"""Used to recursively go through the leaves of the refined mesh
		and add each leaf ("child-most" element) into the element list for 
		each node (node to element connecitivity). This element list is what is 
		ultimately used to efficiently generate the contributions of each element 
		to the global finite element system.
		
		Args:
		    element (obj): The element of interest to add to the node to element lists.
		"""

		if len(element.children) == 0:
			
			# Base Case
			element.set_node_to_element_connectivity()

		else:

			# Loop through the children
			for i in range(2):
				for j in range(2):
					self.set_node_to_element_connectivity_element(element.children[i][j])


	def set_node_to_element_connectivity(self):
		
		"""Generate the connecivity of dofs to elements. 
		That is, for each dof, get the list of elements that hold this dof.
		This will be done to easily get the contributions of each element
		to the global system.

		Recursively go through the mesh and obtain the connectivity in order
		to handle h-refined meshes.

		TODO: Make slight adjustments here to handle h-refined meshes
		"""


		# Empty the element lists of each node.
		# - NOTE: we will recursively go through the elements to their leaves
		#		and do the clearing process there.
		# - NOTE: The same nodes will surely be cleared mutliple times. 
		#		This inefficiency is negligble
		for element in self.element_list:
			self.clear_node_to_element_connectivity_element(element)


		# Add each element to the node's element lists to generate the 
		# connectivity information. Do this recursively as we will
		# add only the leaves of the mesh ("child-most" elements) into
		# these connectivty lists
		for element in self.element_list:
			self.set_node_to_element_connectivity_element(element)


	def set_dof_hanging_node_list_element(self, element, node_list, node_type):
		
		"""Used to recursively to get the degrees of freedom or 
		hanging nodes of the mesh by traversing through the leaves of the tree.
		
		Args:
		    element (obj): The element that we are concentrating
		    	on to collect dofs from
		    node_list (list): List to hold the nodes that are being collected
		    	from the mesh.
		    node_type (str): Specifies what type of node_type are being
		    	collected. Options are "dof" and "hanging".
		"""

		if len(element.children) == 0:

			# Base Case

			# Loop through the nodes and collect the specified nodes
			for i in range(2):
				for j in range(2):
					node_ij = element.node_matrix[i][j]

					if node_ij.node_type == node_type:
						if node_ij not in node_list:
							node_list.append(node_ij)

		else:

			for i in range(2):
				for j in range(2):
					self.set_dof_hanging_node_list_element(element.children[i][j], node_list, node_type)


	def set_dof_list(self):
		
		"""Assemble the list of nodes that will serve as the degrees of 
		freedom for the mesh. Loop through all leaves of the refined mesh
		tree. 
		"""

		self.dof_list = []
		for element in self.element_list:
			self.set_dof_hanging_node_list_element(element, self.dof_list, "dof")


	def set_hanging_node_list(self):
		
		"""Assemble the list of nodes that will serve as the hanging nodes 
		for the mesh. Loop through all leaves of the refined mesh tree. 
		"""

		self.hanging_node_list = []
		for element in self.element_list:
			self.set_dof_hanging_node_list_element(element, self.hanging_node_list, "hanging")


	def get_solution_at_position_element(self, element, x, y):
		
		"""Used to recursively go through the leaves of the mesh
		and to get the element ("childmost") that holds this (x,y)
		coordinate and get the solution from it
		
		Args:
		    element (obj): The element of interest to check for the solution in.
		    x (float): x-position at which the solution is required
		    y (float): y-position at which the solution is required
		
		Returns:
		    float: Value of the solution at the specified position. If the
		    	point is not in the element, then returns None
		"""

		if len(element.children) == 0:
			
			# Base Case

			if element.x_range[0] <= x <= element.x_range[1] and \
				element.y_range[0] <= y <= element.y_range[1]:

				return element.get_solution_at_position(x,y)

			else:
				return None

		else:

			# Loop over the children
			for i in range(2):
				for j in range(2):

					val = self.get_solution_at_position_element(element.children[i][j], x, y)

					if val is not None:
						return val

			return None


	def get_solution_at_position(self, x, y):
		
		"""Get the solution at a specified position on the
		domain.
		
		TODO: Make adjustments here to handle h-refined
		meshes

		Args:
		    x (float): x-position at which the solution is required
		    y (float): y-position at which the solution is required
		
		Returns:
		    float: Value of the solution at the specified position.
		"""



		for element in self.element_list:

			val = self.get_solution_at_position_element(element, x, y)

			if val is not None:
				return val


	def plot_mesh_element(self, element, plot_nodes, linewidth):
		
		"""Recursively plot the mesh elements (the leaves of the tree)
		
		Args:
		    element (obj): The finite element structure to plot
		"""

		if len(element.children) == 0:

			# Base Case

			# Ordering of the vertices is counterclockwise
			vertex_pt_list = element.vertex_list

			x_vals_vertices = [node.x for node in vertex_pt_list]
			y_vals_vertices = [node.y for node in vertex_pt_list]

			# Append the first element onto the end as we want the vertices
			# to cycle around.
			x_vals_vertices.append(x_vals_vertices[0])
			y_vals_vertices.append(y_vals_vertices[0])

			if linewidth is None:
				plt.plot(x_vals_vertices, y_vals_vertices, c="k")
			else:
				plt.plot(x_vals_vertices, y_vals_vertices, c="k", linewidth=linewidth)


			if plot_nodes:

				# Plot the nodes on the mesh as well
				hanging_nodes = []
				dof_nodes = []
				dirichlet_bc_nodes = []

				for i in range(2):
					for j in range(2):
						
						node_pt = (element.node_matrix[i][j].x, element.node_matrix[i][j].y)

						if element.node_matrix[i][j].node_type == "dof":
							dof_nodes.append(node_pt)
						elif element.node_matrix[i][j].node_type == "dirichlet_bc":
							dirichlet_bc_nodes.append(node_pt)
						elif element.node_matrix[i][j].node_type == "hanging":
							hanging_nodes.append(node_pt)

				plt.scatter([x[0] for x in dof_nodes], [x[1] for x in dof_nodes], c="blue")
				plt.scatter([x[0] for x in hanging_nodes], [x[1] for x in hanging_nodes], c="red")
				plt.scatter([x[0] for x in dirichlet_bc_nodes], [x[1] for x in dirichlet_bc_nodes], c="orange")


		else:

			for i in range(2):
				for j in range(2):
					self.plot_mesh_element(element.children[i][j], plot_nodes)


	def plot_mesh(self, plot_nodes=True, linewidth=None):
		
		"""Plot the mesh to visualize all elements. Do this
		by progressing through the unstructured list of elements

		TODO: Make adjustments to handle h-refined meshes

		NOTE: For now, all elements are of maximum order p = 1
		"""


		for element in self.element_list:
			self.plot_mesh_element(element, plot_nodes, linewidth)


	def project_solution(self):
		
		"""Project the solution onto a grid. Return the x, y and u grids.

		TODO: Make adjustments to handle h-refined meshes
		"""
		
		n_plot_vals = 50

		x_plot_vals = np.linspace(self.x_range[0], self.x_range[1], n_plot_vals)
		y_plot_vals = np.linspace(self.y_range[0], self.y_range[1], n_plot_vals)
		x_plot_vals, y_plot_vals = np.meshgrid(x_plot_vals, y_plot_vals)
		
		u_plot_vals = np.zeros((n_plot_vals, n_plot_vals))

		for i in range(n_plot_vals):
			for j in range(n_plot_vals):

				x_ij = x_plot_vals[i,j]
				y_ij = y_plot_vals[i,j]

				u_plot_vals[i,j] = self.get_solution_at_position(x_ij, y_ij)

		return x_plot_vals, y_plot_vals, u_plot_vals


class TriMeshGMSH(object):

	"""Read the mesh from the gmsh file. The file should hold
	Tris. For now, we will ignore the boundary condition flags and
	set those in this object.
	"""

	def __init__(self, p, mesh_properties, gmsh_file_path):
		
		"""Constructor for the tri mesh. Read the file from gmsh and generate the given
		mesh.
		
		NOTE: For now, we set the dirichlet BC using the function flag. All other
			edges are zero neumann BCs.

		Args:
		    p (int): The order of the elements (initially)
		    mesh_properties (dict): Properties of the mesh. Holds the following parameters:
		    	"dirichlet_bc_flag": The function for setting the dirichlet bc.
		    gmsh_file_path (str): Path to the gmsh file to load. We will get the vertices
		    	and elements from here. 
		"""

		self.p = p
		if self.p != 1:
			raise ValueError("Unsupported P Value")

		self.dirichlet_bc_flag_function = mesh_properties["dirichlet_bc_flag"]

		
		# ===========================
		#      Load GMSH Data
		# ===========================

		self.gmsh_file_path = gmsh_file_path
		mesh_nodes, element_node_lists = self.read_gmsh_file()


		# ===========================
		#   Generate the elements
		# ===========================

		self.element_list = []
		for element_node_list in element_node_lists:
			node_list = [mesh_nodes[x] for x in element_node_list]
			self.element_list.append(tri_finite_element.TriFiniteElement(self.p, node_list))


		# Compute the connectivity between the elements. For now, the procedure will
		# be quite innefficient as a first design pass
		for element_i in self.element_list:
			for element_j in self.element_list:

				# Only check connections between elements other than this one
				if element_j == element_i:
					continue

				tri_finite_element.TriFiniteElement.set_element_connectivity(element_i, element_j)


		# Set the connectivity of nodes to elements
		self.set_node_to_element_connectivity()

		
		# ===========================
		#      Dirichlet BC Setup 
		# ===========================

		# # Set the Dirichlet BC on the mesh and its nodes
		self.set_dirichlet_boundary_condition()


		# ===========================
		#      Global DOF List
		# ===========================

		# Set the list of dofs for this mesh. This will be used for the system solve
		self.dof_list = []
		self.set_dof_list()


		self.dirichlet_bc_list = []
		self.set_dirichlet_bc_list()


	def h_refine_mesh(self, refine_element_list):
		
		""" Refine the mesh using h-adaptivity. Refine the
		elements from the refine_element_list

		Args:
		    element_list (list): The list of elements to be refined. Note
		    	that this list will be modified to remove elements that
		    	no longer exist.
		"""

		# Progress through the list of elements to refine until it is empty
		while True:

			if len(refine_element_list) == 0:
				break

			element_to_refine = refine_element_list[0]
			newly_refined_elements, outdated_refined_elements = element_to_refine.refine_element()
			
			# Add the newly generated elements to the element list and
			# remove the outdated elements

			for e_add in newly_refined_elements:
				self.element_list.append(e_add)

			for e_remove in outdated_refined_elements:
				if e_remove in self.element_list:
					self.element_list.remove(e_remove)


			# Modify the list of elements to be refined.
			# Remove this  element from the list of elements to refine
			if element_to_refine in refine_element_list:
				refine_element_list.remove(element_to_refine)

			# Remove any element from the list of outdated refined elements
			# from the list of elements to refine
			for e_remove in outdated_refined_elements:
				if e_remove in refine_element_list:
					refine_element_list.remove(e_remove)


		# Set the connectivity of nodes to elements
		self.set_node_to_element_connectivity()

		
		# ===========================
		#      Dirichlet BC Setup 
		# ===========================

		# # Set the Dirichlet BC on the mesh and its nodes
		self.set_dirichlet_boundary_condition()


		# ===========================
		#      Global DOF List
		# ===========================

		# Set the list of dofs for this mesh. This will be used for the system solve
		self.dof_list = []
		self.set_dof_list()


		self.dirichlet_bc_list = []
		self.set_dirichlet_bc_list()


		# # Have created a new h-adapated mesh. Now set the dirichlet bc nodes
		# # and clear and set the degree of freedom list

		# # Set the connectivity of nodes to elements
		# self.set_node_to_element_connectivity()

		# self.set_dirichlet_boundary_condition()

		# self.dof_list = []
		# self.set_dof_list()



	def set_node_to_element_connectivity(self):

		# Clear all element lists for each node
		for element in self.element_list:
			for node in element.node_list:
				node.clear_element_list()

		# Add each element to their node's element lists
		for element in self.element_list:
			for node in element.node_list:
				node.add_element(element)


	def set_dirichlet_boundary_condition(self):
		
		"""Go through all nodes and set the dirichlet boundary
		condition nodes.

		NOTE: For now, we will only have dirichlet and zero Neumann
		BCs. Perhaps build off gmsh later by flagging edges that are
		different types of BCs.
		"""

		# If we have no dirichlet BC imposed, then there is nothing to apply
		if self.dirichlet_bc_flag_function is None:
			return

		for element in self.element_list:
			for node in element.node_list:

				bool_val, dirichlet_bc_val = self.dirichlet_bc_flag_function(node.x, node.y)

				if bool_val:
					node.set_type("dirichlet_bc")
					node.set_value(dirichlet_bc_val)


	def set_dirichlet_bc_list(self):
		
		"""Collect the list of dirichlet BC nodes
		"""

		self.dirichlet_bc_list = []
		
		for element in self.element_list:
			for node in element.node_list:
				if node.node_type == "dirichlet_bc" and node not in self.dirichlet_bc_list:
					self.dirichlet_bc_list.append(node)


	def set_dof_list(self):
		
		"""Go through all elements and build the list of degrees of freedom.
		We will not empty the list here. All that will be done is that Nodes
		that are dofs, and not present in the list, will be added.
		"""

		self.dof_list = []
		
		for element in self.element_list:
			for node in element.node_list:
				if node.node_type == "dof" and node not in self.dof_list:
					self.dof_list.append(node)


	def plot_mesh(self, plot_nodes=True, linewidth=None):
		
		"""Plot the mesh to visualize all elements. Do this
		by progressing through the unstructured list of elements

		NOTE: For now, all elements are of maximum order p = 1
		"""

		# Hold node objects for the dof and dirichlet BCs
		nodes_dof = []
		nodes_dirichlet = []

		for i_element in range(len(self.element_list)):
			element = self.element_list[i_element]

			node_list = element.node_list
			node_x_vals, node_y_vals = [node.x for node in node_list], [node.y for node in node_list] 

			node_x_vals.append(node_x_vals[0])
			node_y_vals.append(node_y_vals[0])

			if linewidth is None:
				plt.plot(node_x_vals, node_y_vals, c="k")
			else:
				plt.plot(node_x_vals, node_y_vals, c="k", linewidth=linewidth, alpha=0.9)

			for node in node_list:

				if node.node_type == "dof" and node not in nodes_dof:
					nodes_dof.append(node)

				if node.node_type == "dirichlet_bc" and node not in nodes_dirichlet:
					nodes_dirichlet.append(node)


		if plot_nodes is True:
			dof_pts_x, dof_pts_y = [node.x for node in nodes_dof], [node.y for node in nodes_dof]
			dirichlet_pts_x, dirichlet_pts_y = [node.x for node in nodes_dirichlet], [node.y for node in nodes_dirichlet]

			plt.scatter(dof_pts_x, dof_pts_y, c="blue")
			plt.scatter(dirichlet_pts_x, dirichlet_pts_y, c="red")



	def read_gmsh_file(self):
		
		"""Read the gmsh file and load the nodes and elements. Load
		the data into the mesh_vertex_nodes and element_connectivity_list 
		data structures.

		NOTE: We will only read tri elements from the mesh so we can
			only handle tri meshes for this case
		"""

		mesh_nodes = []
		element_node_lists = []

		if self.gmsh_file_path is None:
			return mesh_nodes, element_node_lists


		with open(self.gmsh_file_path, "r") as fp:

			while True:

				line = fp.readline().rstrip("\n")

				if line == "$Nodes":
					# Have entered the Nodes section. Read the node data

					num_nodes = int(fp.readline().rstrip("\n"))
					mesh_nodes = [None]*num_nodes

					for i in range(num_nodes):
						node_data_line = fp.readline().rstrip("\n").split()
						mesh_nodes[int(node_data_line[0])-1] = \
									node.Node("dof", float(node_data_line[1]), float(node_data_line[2]))


				if line == "$Elements":
					# Have entered the Element section. Read only the
					# triangle connectivity data

					num_elem = int(fp.readline().rstrip("\n"))

					for i in range(num_elem):
						elem_data_ints = [int(x) for x in fp.readline().rstrip("\n").split()]

						if elem_data_ints[1] == 2:
							# Found a triangle

							tri_node_info = []
							for i_tri_node in range(len(elem_data_ints)-3, len(elem_data_ints)):
								# Get the node information (use 0 based indexing to be consistent)
								tri_node_info.append(elem_data_ints[i_tri_node]-1)

							element_node_lists.append(tri_node_info)


				if line == "":
					break

		return mesh_nodes, element_node_lists


	def get_solution_at_position(self, x, y):
		
		"""Get the solution at a specified position on the
		domain.

		Args:
		    x (float): x-position at which the solution is required
		    y (float): y-position at which the solution is required
		
		Returns:
		    float: Value of the solution at the specified position.
		"""

		for element in self.element_list:

			# If the point is in the element, then return the value there
			if element.mapping_physical_to_reference_domain(x,y)[1]:
				return element.get_solution_at_position(x,y)


	def plot_basis_at_point(self, dof, x, y):
		
		"""Plot the basis function (nodal) at the specified point
		
		Args:
		    dof (obj): The node object corresponding to the nodal basis 
		   		function of interest 
		    x (float): x-position at which the solution is required
		    y (float): y-position at which the solution is required
		
		Returns:
		    float: Value of the solution at the specified position.
		"""

		for element in self.element_list:

			# If the point is in the element, then return the value there
			if element.mapping_physical_to_reference_domain(x,y)[1]:
				if dof in element.node_list:

					psi_hat_i, grad_psi_hat_i = element.get_basis_funcs_given_dof(dof)
					xi_eta_pt = element.mapping_physical_to_reference_domain(x,y)[0]
					return psi_hat_i(xi_eta_pt[0], xi_eta_pt[1])

				else:
					return 0.0


	def project_solution(self, x_plot_vals, y_plot_vals, n_plot_vals):
		
		"""Project the solution onto a grid. Return the x, y and u grids.

		NOTE: This function will get the x and y values to plot the solution
			at from the user since we shouldn't know the geometry from 
			this function.

		NOTE: Add in the capability to output paraview files to visualize 
			the solution to get some practice with the software.
		"""
		
		x_plot_vals_grid, y_plot_vals_grid = np.meshgrid(x_plot_vals, y_plot_vals)
		u_plot_vals = np.zeros((n_plot_vals, n_plot_vals))

		for i in range(n_plot_vals):
			for j in range(n_plot_vals):

				x_ij = x_plot_vals_grid[i,j]
				y_ij = y_plot_vals_grid[i,j]

				u_plot_vals[i,j] = self.get_solution_at_position(x_ij, y_ij)

		return x_plot_vals_grid, y_plot_vals_grid, u_plot_vals


	def project_error_indicator(self, x_plot_vals, y_plot_vals, n_plot_vals):
		
		"""Get the error indicator distribution over the mesh
		
		Args:
		    x_plot_vals (TYPE): Description
		    y_plot_vals (TYPE): Description
		    n_plot_vals (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""

		x_plot_vals_grid, y_plot_vals_grid = np.meshgrid(x_plot_vals, y_plot_vals)
		e_plot_vals = np.zeros((n_plot_vals, n_plot_vals))

		for i in range(n_plot_vals):
			for j in range(n_plot_vals):

				x_ij = x_plot_vals_grid[i,j]
				y_ij = y_plot_vals_grid[i,j]

				for elem in self.element_list:
					if elem.mapping_physical_to_reference_domain(x_ij, y_ij)[1]:
						# If the point is in this element
						e_plot_vals[i,j] = elem.error_indicator

		return x_plot_vals_grid, y_plot_vals_grid, e_plot_vals


	def plot_nodal_basis(self, dof, x_plot_vals, y_plot_vals, n_plot_vals):
		
		"""Plot the nodal basis function for a given degree of freedom.
		
		Args:
		    dof (obj): The node for which we wish to plot the nodal
		    	basis for
		"""

		x_plot_vals_grid, y_plot_vals_grid = np.meshgrid(x_plot_vals, y_plot_vals)
		u_plot_vals = np.zeros((n_plot_vals, n_plot_vals))

		for i in range(n_plot_vals):
			for j in range(n_plot_vals):

				x_ij = x_plot_vals_grid[i,j]
				y_ij = y_plot_vals_grid[i,j]

				u_plot_vals[i,j] = self.plot_basis_at_point(dof, x_ij, y_ij)

		return x_plot_vals_grid, y_plot_vals_grid, u_plot_vals




