"""
Module: quad_finite_element.py

Holds the data structure that will have the finite element structure. Will 
be used to generate quadrilateral finite elements.
"""


import reference_element
import node
import vertex


class QuadFiniteElement(object):

	"""Holds the data for a given finite element. 

	NOTE: For now, we will only be able to handle linear 
		elements. We will then add capabilities of higher
		order element geometries.
	"""
	
	def __init__(self, p, element_x_range, element_y_range, vertex_pt_matrix, refinement_level):
		
		"""Constructor for the finite element for the FEM mesh. Each element
		has the following ordering for the vertices and faces (using 0 based indexing)

				   3  f=2   2
					 -----
				f=3	|	  | f=1
					|  	  | 
					 -----
				   0  f=0  1

		Note that all i,j indexing on the element is dictated through the vertices.
		i increases from vertex 0 to 1 and j from 0 to 3.

		Args:
		    element_x_range (list): List of floats with then x limits 
		    	of the element.
		    element_y_range (list): List of floats with the y limits
		    	of the element.
		    vertex_pt_matrix (list): Matrix of the vertices for this element. Ordering
		    	is consistent with the figure above
		"""

		self.p = p

		# NOTE: For now, we will be working exclusively with bilinear elements.
		if self.p != 1:
			raise ValueError("Unsupported P value")

		self.refinement_level = refinement_level

		# Total number of generations of refinement starting from this element
		self.num_generation_refinement_levels = 0  

		
		# Structures to allow for h-refinement
		self.parent = None  # The parent to this element
		self.children = []  # The children of this element


		# TODO: Define a pc term to specify the order of the cubature. Then
		# all values will be computed on the cubature nodes (GL).

		# TODO: Define a pg term to specify the order of the geometry. This way,
		# we will be able to handle superparameteric meshes.

		
		# Vertex matrix. This provides the i,j indexing for the element.
		self.vertex_matrix = vertex_pt_matrix

		# Store the vertices that mark this element. Ordering given in the 
		# comment above (counterclockwise ordering)
		self.vertex_list = []
		self.vertex_list.append(self.vertex_matrix[0][0])
		self.vertex_list.append(self.vertex_matrix[1][0])
		self.vertex_list.append(self.vertex_matrix[1][1])
		self.vertex_list.append(self.vertex_matrix[0][1])


		# The neighbors for the element (ordering is given by the
		# face ordering above). Held in this list are the elements
		# that neighbor this element. 

		# - NOTE: We hold a list of maximum size two as, with
		# adaptation, each element will be allowed to neighbor two elements.
		self.neighbors = []
		# The index for the face of the neighboring element that is neighbors
		# with this element
		self.neighbors_face_index = []
		for i in range(4):
			self.neighbors.append(None)
			self.neighbors_face_index.append(None)

		

		# Generate the reference element of the given order
		self.reference_element = reference_element.QuadReferenceElement(self.p)


		# For now, take in the x range and y range. We will eventually remove
		# these to be able to handle arbitrary geometries
		self.x_range = element_x_range
		self.delta_x = element_x_range[1] - element_x_range[0]

		self.y_range = element_y_range
		self.delta_y = element_y_range[1] - element_y_range[0]


		# TODO: Compute all metric terms (the cofactor matrix and jacobian)
		# Do this using the order pg geometry shape functions and compute the
		# values at the cubature nodes (of order pc)

		# Metric terms
		self.Jac = 0.5*0.5*self.delta_x*self.delta_y


		# Generate the node matrix using the given order p. 
		# Ordering of the basis functions is from vertex 0 to vertex 1
		# (p+1 nodes along the i direction), and then layered above (increase j).
		# Thus, as always, the ij indexing of the vertices is dictated by the 
		# ordering of the vertices.

		# NOTE: For now, fill the node_matrix with Nones. Then, once the connectivity
		# between the elements has been set, the nodes will be generated (so that
		# nodes can be shared between elements)
		self.node_matrix = []
		for i in range(2):
			col = []
			for j in range(2):
				#col.append(node.Node("dof", self.vertex_matrix[i][j].x, self.vertex_matrix[i][j].y))
				col.append(None)
			self.node_matrix.append(col)



	def __str__(self):

		return "x_range : [%.3f, %.3f], y_range : [%.3f, %.3f]" % \
			(self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1])


	# ==============================
	#        Class Methods
	# ==============================


	def refine_element(self):
		
		"""Refine the given element. Do this by splitting it
		into 4 elements. 

		All refinement will be done locally using the element and its
		neighbors (that is all). This way, we will be able to do robust
		refinment. This means, during the refinement, there should be 
		no interaction with any other elements.
		"""

		# TODO:
		# First, check refinement level of other elements


		# =============================
		#     Split Parent Element
		# =============================

		# Split this element into 4 child elements. Store it in the children
		# data structure. Make each child have equal dimensions.

		self.children = [[None, None], [None, None]]

		child_dx = 0.5*self.delta_x
		child_dy = 0.5*self.delta_y

		for i in range(2):
			for j in range(2):

				# x and y range for this child
				child_x_range = [self.x_range[0] + i*child_dx, self.x_range[0] + (i+1)*child_dx]
				child_y_range = [self.y_range[0] + j*child_dy, self.y_range[0] + (j+1)*child_dy]

				
				# Create the matrix of vertex elements
				vertex_pt_matrix = [[None, None], [None, None]]
				for i_vertex in range(2):
					for j_vertex in range(2):
						vertex_pt_matrix[i_vertex][j_vertex] = vertex.Vertex(child_x_range[i_vertex], child_y_range[j_vertex])

				
				# Create the refined element
				h_refined_element = QuadFiniteElement(self.p, child_x_range, child_y_range, 
					vertex_pt_matrix, self.refinement_level+1)


				self.children[i][j] = h_refined_element


		# =============================
		#      Set the neighbors
		# =============================

		# Set the neighbor references. Set the reference to the neighbors for
		# each child element as well as all outer neighbors (the neighbors to 
		# this parent element)

		# NOTE: We do not need to check neighbor refinement levels here. This is
		# because we will, before refining this element, make sure that the mesh 
		# follows the one element rule. So, if an element has only one neighbor,
		# that neighbor is of the same size as this parent element.

		# NOTE: When setting neighbors, we will set the neighboring faces for both
		# elements involved.


		# Set the neighboring faces between the children

		for i in range(2):
			for j in range(2):

				child_ij = self.children[i][j]

				if i+1 <= 1:
					# Set the right face (face = 1)
					child_ij.set_neighbor([self.children[i+1][j]], 1, [3])

				if i-1 >= 0:
					# Set the left face (face = 3)
					child_ij.set_neighbor([self.children[i-1][j]], 3, [1])

				if j+1 <= 1:
					# Set the top face (face = 2)
					child_ij.set_neighbor([self.children[i][j+1]], 2, [0])

				if j-1 >= 0:
					# Set the bottom face (face = 0)
					child_ij.set_neighbor([self.children[i][j-1]], 0, [2])


		# Set the neighboring faces to outer elements. These are the elements that
		# neighbor/touch the parent element that is being split

		for face_index in range(4):

			# Get the child elements that are on this face. The ordering of the
			# faces in each list is counterclockwise around the element.
			face_child_elements = []

			# Get the indeces for what face on the child elements is touching 
			# this face. Will be the same as the parent element
			face_child_elements_face_indeces = [face_index, face_index]		

			if face_index == 0:
				face_child_elements = [self.children[0][0], self.children[1][0]]
			elif face_index == 1:
				face_child_elements = [self.children[1][0], self.children[1][1]]
			elif face_index == 2:
				face_child_elements = [self.children[1][1], self.children[0][1]]
			elif face_index == 3:
				face_child_elements = [self.children[0][1], self.children[0][0]]


			# Get the index of the face of the neighbors that touches this face on this element
			neighbor_elements = self.neighbors[face_index]
			neighbors_face_index = self.neighbors_face_index[face_index]


			if neighbor_elements is None:
				# The neighbor is a boundary (so do nothing)
				pass

			elif len(neighbor_elements) == 1:
				# Only a single outer neighbor on this face. Set the reference to the face child
				# neighbors. 
				# - Note: We flip the order of the elements when setting the neighbor. This is done
				# 	to stay consistent with the ordering of multiple neighbors for elements.
				neigh_elem = neighbor_elements[0]
				neigh_elem_face_index = neighbors_face_index[0]
				neigh_elem.set_neighbor([face_child_elements[1],  face_child_elements[0]], 
					neigh_elem_face_index, [face_child_elements_face_indeces[1], face_child_elements_face_indeces[0]])

				# Set neighbor references for the children
				for face_child in face_child_elements:
					face_child.set_neighbor([neigh_elem], face_index, [neigh_elem_face_index])

			else:
				# 2 neighbors on this face
				raise ValueError("Unsupported right now")



		# =============================
		#       Set the nodes
		# =============================

		# Set nodes for the refined elements. With the neighbors set, 
		# go through each element and set the nodes (regular and hanging)
		# based on the number of neighbors. Only generate nodes that are not
		# already existing. Then, set the dof to element connecitivity.


		# 1) Get the nodes that form the corner of the element. We will take these
		# 	from the parent element. Allocate the correct node to each child
		#	element.

		for i in range(2):
			for j in range(2):
				self.children[i][j].node_matrix[i][j] = self.node_matrix[i][j]


		# 2) Create the node at the middle of the parent element. This will be 
		# 	shared by all child elements. This will be a dof for the system.

		mid_node_x, mid_node_y = self.x_range[0] + child_dx, self.y_range[0] + child_dy
		mid_node = node.Node("dof", mid_node_x, mid_node_y)

		self.children[0][0].node_matrix[1][1] = mid_node  # Bottom left element
		self.children[1][0].node_matrix[0][1] = mid_node  # Bottom right element
		self.children[1][1].node_matrix[0][0] = mid_node  # Top right element
		self.children[0][1].node_matrix[1][0] = mid_node  # Top left element


		# 3) Create or get the nodes at the middle outer edges of the split element.
		# 	There are two cases here:
		# 	- Case 1: A single outer element is on the boundary. In this case, 
		# 		the middle node on the face of the parent element that has been split
		#		will need to be generated and will be a hanging node.
		#	- Case 2: Two outer elements are on the boundary. In this case, the
		#		middle node on the face of the parent element is already existing
		#		as it is a hanging node. Convert this hanging node into a dof and 
		#		store its reference for each new child element.
		#	- Case 3: The outer boundary has no neighboring element and is therefore
		#		on a boundary of the geometry. In this case, generate the middle node.
		#		Then, check if this node is on a dirichlet BC. If it is, set it to be
		#		a dirichlet BC node. If it is on a Neumann BC, set it to be a hanging 
		#		node. 

		for face_index in range(4):

			# Get the child elements on this given face



			face_outer_neighbors = self.neighbors[face_index]


			if face_outer_neighbors is None:
				# Case 3:

				if face_index == 0:

					hanging_node_x, hanging_node_y = self.children[0][0].vertex_matrix[1][0].x, \
						self.children[0][0].vertex_matrix[1][0].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[0][0].node_matrix[0][0], 
						self.children[1][0].node_matrix[1][0]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[0][0].node_matrix[1][0] = hanging_node
					self.children[1][0].node_matrix[0][0] = hanging_node

				elif face_index == 1:

					hanging_node_x, hanging_node_y = self.children[1][0].vertex_matrix[1][1].x, \
						self.children[1][0].vertex_matrix[1][1].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[1][0].node_matrix[1][0], 
						self.children[1][1].node_matrix[1][1]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[1][0].node_matrix[1][1] = hanging_node
					self.children[1][1].node_matrix[1][0] = hanging_node
					
				elif face_index == 2:
					
					hanging_node_x, hanging_node_y = self.children[1][1].vertex_matrix[0][1].x, \
						self.children[1][1].vertex_matrix[0][1].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[1][1].node_matrix[1][1], 
						self.children[0][1].node_matrix[0][1]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[1][1].node_matrix[0][1] = hanging_node
					self.children[0][1].node_matrix[1][1] = hanging_node

				elif face_index == 3:

					hanging_node_x, hanging_node_y = self.children[0][1].vertex_matrix[0][0].x, \
						self.children[0][1].vertex_matrix[0][0].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[0][1].node_matrix[0][1], 
						self.children[0][0].node_matrix[0][0]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[0][1].node_matrix[0][0] = hanging_node
					self.children[0][0].node_matrix[0][1] = hanging_node


			elif len(face_outer_neighbors) == 1:
				# Case 1:

				if face_index == 0:

					hanging_node_x, hanging_node_y = self.children[0][0].vertex_matrix[1][0].x, \
						self.children[0][0].vertex_matrix[1][0].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[0][0].node_matrix[0][0], 
						self.children[1][0].node_matrix[1][0]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[0][0].node_matrix[1][0] = hanging_node
					self.children[1][0].node_matrix[0][0] = hanging_node

				elif face_index == 1:

					hanging_node_x, hanging_node_y = self.children[1][0].vertex_matrix[1][1].x, \
						self.children[1][0].vertex_matrix[1][1].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[1][0].node_matrix[1][0], 
						self.children[1][1].node_matrix[1][1]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[1][0].node_matrix[1][1] = hanging_node
					self.children[1][1].node_matrix[1][0] = hanging_node
					
				elif face_index == 2:
					
					hanging_node_x, hanging_node_y = self.children[1][1].vertex_matrix[0][1].x, \
						self.children[1][1].vertex_matrix[0][1].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[1][1].node_matrix[1][1], 
						self.children[0][1].node_matrix[0][1]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[1][1].node_matrix[0][1] = hanging_node
					self.children[0][1].node_matrix[1][1] = hanging_node

				elif face_index == 3:

					hanging_node_x, hanging_node_y = self.children[0][1].vertex_matrix[0][0].x, \
						self.children[0][1].vertex_matrix[0][0].y
					hanging_node = node.Node("hanging", hanging_node_x, hanging_node_y)

					constraint_nodes = [
						self.children[0][1].node_matrix[0][1], 
						self.children[0][0].node_matrix[0][0]
						]

					hanging_node.set_hanging_nodes_constraint_nodes(constraint_nodes)

					# Allocate the hanging node
					self.children[0][1].node_matrix[0][0] = hanging_node
					self.children[0][0].node_matrix[0][1] = hanging_node

			elif len(face_outer_neighbors) == 2:
				# Case 2:

				raise ValueError("To Implement Still")


		# # 4) Remove the parent element from the node lists. This is
		# # 	because this parent element no longer be a leaf of the tree
		# #	since it has children
		# for i in range(2):
		# 	for j in range(2):
		# 		self.node_matrix[i][j].remove_element(self)


		# # 5) Add the children elements to the node element lists
		# for i in range(2):
		# 	for j in range(2):

		# 		child_ij = self.children[i][j]

		# 		for i_node in range(2):
		# 			for j_node in range(2):
		# 				child_ij.node_matrix[i_node][j_node].add_element(child_ij)


	def print_element_properties(self):
		
		"""Print the properties of this element. This include the 
		neighbors
		"""
		
		print "============================="
		print self

		for i in range(len(self.neighbors)):
			print " face i : %d" % i

			if self.neighbors[i] is not None:
				for neigh in self.neighbors[i]:
					print "    %s" % neigh

			else:
				print "    None"

		print "============================="


	# ==============================
	#           Getters
	# ==============================


	def get_ij_index_node(self, node):
		
		"""Get the i,j index associated to a given node. If the node
		is not present in the node matrix, then simply return None
		
		Args:
		    node (obj): The node object that is being searched for
		"""

		for i in range(2):
			for j in range(2):
				if self.node_matrix[i][j] == node:
					return (i,j)

		return None


	def get_nodes_on_face(self, face_index):
		
		"""Get the nodes on a given face. Note that the ordering
		of the list of nodes returned is always counterclockwise
		around the element.
		
		Args:
		    face_index (int): Index of which face we wish to get the nodes
		    	from
		
		Returns:
		    list: List with the nodes on the face. Will be of size 2 (since
		    	we are only dealing with P=1 elements for now)
		"""

		if face_index == 0:
			return [self.node_matrix[0][0], self.node_matrix[1][0]]
		elif face_index == 1:
			return [self.node_matrix[1][0], self.node_matrix[1][1]]
		elif face_index == 2:
			return [self.node_matrix[1][1], self.node_matrix[0][1]]
		elif face_index == 3:
			return [self.node_matrix[0][1], self.node_matrix[0][0]]
		else:
			raise ValueError("Unknown face index")


	def get_solution_at_position(self, x, y):

		"""Get the solution at the position x,y in the element
		
		Args:
		    x (float): x-position at which the solution is required
		    y (float): y-position at which the solution is requiredn
		
		Returns:
		    float: Value of the solution at the specified location in the element.
		"""

		if self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]:

			# Get the basis functions on the reference domain
			psi_hat_basis = self.reference_element.psi_hat_basis

			# Map the point to the reference domain
			x1, x2 = self.x_range[0], self.x_range[1]
			y1, y2 = self.y_range[0], self.y_range[1]

			xi_mapped = (2./(x2-x1)) * (x - (x1+x2)*0.5)
			eta_mapped = (2./(y2-y1)) * (y - (y1+y2)*0.5)

			u_val = 0.0
			for i in range(2):
				for j in range(2):
					u_val += self.node_matrix[i][j].value * psi_hat_basis[i][j](xi_mapped, eta_mapped)
			
			return u_val
			
		else:
			raise ValueError("Position not in element domain")


	# ==============================
	#           Setters
	# ==============================


	def set_node_to_element_connectivity(self):
		
		for i in range(2):
			for j in range(2):
				self.node_matrix[i][j].add_element(self)


	def set_neighbor(self, element_list, face_index, neighbor_face_index_list):
		
		"""Set a neighbors for the element. A maximum of two neighbors may be set
		for a given element. In the case where multiple neighbors are given, the
		following is the indexing convention for the element list 

			
				  3  n1 n0   2
					 -----
				n0	|	  | n1
				n1	|  	  | n0
					 -----
				  0  n0  n1  1
		
		Each face holds a list of elements. Therefore, the ordering of the elements in each
		list is moving in the clockwise direction around the element.

		Args:
		    element_list (list): List of elements to set for the neighbors
		    	of a given face. Has maximum size 2 and minimum size of 1
		    face_index (int): face index for which to set neighbors for
		    neighbor_face_index_list (int): The list of the index for what face on 
		    	the neighboring elements touch this element
		"""

		if len(element_list) > 2:
			raise ValueError("Too many Neighbors")

		# Store the neighboring element's reference
		self.neighbors[face_index] = element_list
		self.neighbors_face_index[face_index] = neighbor_face_index_list


	def set_nodes_on_face(self, face_index, node_list):
		
		"""Set the nodes on a given face. The input node_list is of
		maximum size 2 as we are consider bilinear elements for now. 
		The input node ordering is counterclockwise.
		
		Args:
		    face_index (int): Index for which face we will set the nodes
		    node_list (list): List of nodes to set for the given face
		"""
		
		if len(node_list) != 2:
			raise ValueError("Unsupported number of face nodes")

		if face_index == 0:
			self.node_matrix[0][0], self.node_matrix[1][0] = node_list[0], node_list[1]
		elif face_index == 1:
			self.node_matrix[1][0], self.node_matrix[1][1] = node_list[0], node_list[1]
		elif face_index == 2:
			self.node_matrix[1][1], self.node_matrix[0][1] = node_list[0], node_list[1]
		elif face_index == 3:
			self.node_matrix[0][1], self.node_matrix[0][0] = node_list[0], node_list[1]
		else:
			raise ValueError("Unknown face index")








