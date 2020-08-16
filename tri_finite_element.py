"""
Module: tri_finite_element.py

Holds the data structure that will have the finite element structure 
for creating triangular finite elements.
"""

import reference_element
import node
import vertex
import numpy as np
import pdb


class TriFiniteElement(object):

	# NOTE: For tris, will need to properly generate the metric
	# 	terms using the vertex points and the shape functions. 
	#	Use an isoparameteric mapping for now. Will compute results
	#	at the cubature nodes.


	def __init__(self, p, node_list):
		
		"""Constructor for the tri finite element. 
			
			3
			|\
		f=3	| f=2
			|  \
			|   \
		   1 ---- 2
			  f=1

		The orderig of the nodes and faces (for the node list and face list)
		is shown above.
		
		Args:
		    p (int): Order of the element
		    node_list (list): List with the nodes for the element.
		"""

		self.p = p
		self.node_list = node_list


		# Rearrange the nodes to accordingly have a positive Jacobian. For this
		# rearranging process, we use the case of a P = 1 isoparametric mapping, 
		# where the Jacobian only depends on the triangle node locations.
		x1, y1 = node_list[0].x, node_list[0].y
		x2, y2 = node_list[1].x, node_list[1].y
		x3, y3 = node_list[2].x, node_list[2].y

		if (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1) < 0:
			temp = self.node_list[1]
			self.node_list[1] = self.node_list[2]
			self.node_list[2] = temp



		# Generate the reference element of the given order
		self.reference_element = reference_element.TriReferenceElement(self.p)


		# The neighboring elements. This list holds, in the ith position, a tuple of the form
		# (neighbor_element, neighbor_element_face_index). This means that the i face for this
		# element touches the neighbor_element_face_index face for the neighbor_element.
		# A value of None means there is no neighbor (the face is on the boundary). The ordering
		# of the neighbors is given by 
		self.neighbors = [None, None, None]

		
		# Compute metric terms. Store these as functions
		# defined over the reference domain.
		self.x_xi, self.x_eta, self.y_xi, self.y_eta, self.J = None, None, None, None, None
		self.compute_metrics()


		# Compute the mapping functions from the reference domain
		# onto the computational domain. Use an isoparametric mapping.

		node_x_vals = [node.x for node in self.node_list]
		node_y_vals = [node.y for node in self.node_list]		

		self.x_mapping_func = lambda xi, eta, node_x_vals=node_x_vals, \
			reference_element=self.reference_element: \
				self.basis_expansion(reference_element.psi_hat_basis, node_x_vals, xi, eta)

		self.y_mapping_func = lambda xi, eta, node_y_vals=node_y_vals, \
			reference_element=self.reference_element: \
				self.basis_expansion(reference_element.psi_hat_basis, node_y_vals, xi, eta)



		# Compute the mapping function from the physical domain to the 
		# reference domain. Note that this will only work for p=1 isoparametric
		# elements,
		self.inverse_mat_phys_to_ref_dom = np.zeros((2,2))
		
		self.inverse_mat_phys_to_ref_dom[0,0] = node_x_vals[1] - node_x_vals[0]
		self.inverse_mat_phys_to_ref_dom[0,1] = node_x_vals[2] - node_x_vals[0]
		self.inverse_mat_phys_to_ref_dom[1,0] = node_y_vals[1] - node_y_vals[0]
		self.inverse_mat_phys_to_ref_dom[1,1] = node_y_vals[2] - node_y_vals[0]

		self.inverse_mat_phys_to_ref_dom = np.linalg.inv(self.inverse_mat_phys_to_ref_dom)
		
		self.shift_vect_phys_to_ref_dom = np.zeros((2,1))
		self.shift_vect_phys_to_ref_dom[0,0] = node_x_vals[0]
		self.shift_vect_phys_to_ref_dom[1,0] = node_y_vals[0]



		# Refinement properties
		# - Dictionary to hold properties used for the refinement.
		# 	Red - Green refinement is used. For Green refinement, 
		# 	we will need to know the neighbor (other green element),
		#	the face index of this element that touches the green neighbor
		#	and the neighboring green element's face index that touches this
		#	element.
		# - All elements will start off as being red.
		self.element_refinement_info = {
			"color" : "red",  # The "color" of the element for the Red-Green adaptation algorithm
			"element_face_index" : None,  # Index of the face on the element touching the green element
			"neighbor" : None,   # The reference to the neighbor green element
			"neighbor_face_index" : None,  # The face on the neighbor green element touching this element
			"shared_hanging_node" : None  # Reference to the node on the bisected edge
			}


		# Error parameters
		self.local_L2_error = None  # The L2 error of the solution over this element (without the square root)
		self.local_Linf_error = None  # The L2 error of the solution over this element (without the square root)
		self.error_indicator = None  # The error indicator used for refinement estimates


	@staticmethod
	def set_element_connectivity(element_i, element_j):
		
		"""Set the connectivity for two elements using their 
		nodes. If a common face is found, then set the corresponding
		neighbors for both elements.

		If we are dealing with two green elements, and a match is found,
		then these elements must be "green element brothers". Set the
		information regarding green elements in this case.

		NOTE: We will most likely create neighbors multiple times but that is
		alright (just inefficient) as that all that happens is the same
		data is written multiple times to the same locations.
		
		Args:
		    element_i (obj): Element i to check connectivity for
		    element_j (obj): Element j to check connectivity for
		"""
		
		# Loop over the faces of element i
		for face_i in range(3):

			# Get the two nodes that form this face
			if face_i < 2:
				elem_i_nodes = [element_i.node_list[face_i], element_i.node_list[face_i+1]]
			else:
				elem_i_nodes = [element_i.node_list[face_i], element_i.node_list[0]]

			# Loop over the faces of element j
			for face_j in range(3):

				# Get the two nodes that form this face
				if face_j < 2:
					elem_j_nodes = [element_j.node_list[face_j], element_j.node_list[face_j+1]]
				else:
					elem_j_nodes = [element_j.node_list[face_j], element_j.node_list[0]]

				# Check for a match. If one is found, then set the neighbors for both elements
				# accordingly
				if (elem_i_nodes[0] == elem_j_nodes[0] and elem_i_nodes[1] == elem_j_nodes[1]) or \
					(elem_i_nodes[0] == elem_j_nodes[1] and elem_i_nodes[1] == elem_j_nodes[0]):

					# Found a matching face
					element_i.set_neighbor(face_i, element_j, face_j)
					element_j.set_neighbor(face_j, element_i, face_i)

					# if element_i.element_refinement_info["color"] == "green" and \
					# 	element_j.element_refinement_info["color"] == "green":
					# 	# The new connection found is between two green elements. 
					# 	# Set the additional connectivity information for this case

					# 	# Element i information
					# 	element_i.element_refinement_info["element_face_index"] = face_i
					# 	element_i.element_refinement_info["neighbor"] = element_j
					# 	element_i.element_refinement_info["neighbor_face_index"] = face_j

					# 	# Element j information
					# 	element_j.element_refinement_info["element_face_index"] = face_j
					# 	element_j.element_refinement_info["neighbor"] = element_i
					# 	element_j.element_refinement_info["neighbor_face_index"] = face_i



	def basis_expansion(self, basis_funcs, coeffs, xi, eta):
		
		"""Get the value of a basis expansion at a specified point. That is,
		we compute each basis function at xi,eta and multiply by the coefficients
		and add the expression up.
		
		Args:
		    basis_funcs (list): List of lambda expressions for the basis function.
		    coeffs (list): List with the coefficients multiplying each basis function.
		    xi (float): xi position to evaluate the basis at
		    eta (float): eta position to evaluate the basis at
		
		Returns:
		    float: Value of the expansion at the specified point
		"""

		val = 0.0
		for i in range(len(basis_funcs)):
			val += basis_funcs[i](xi, eta)*coeffs[i]

		return val



	def set_green_element_connectivity_info(element_i, element_j):
		
		"""Set the information for the green elements. That is,
		set the information to be used for the green refinement which
		is stored in the element_refinement_info dictionary
		
		Args:
		    element_i (obj): Element i to check connectivity for
		    element_j (obj): Element j to check connectivity for
		"""
		
		# Loop over the faces of element i
		for face_i in range(3):

			# Get the two nodes that form this face
			if face_i < 2:
				elem_i_nodes = [element_i.node_list[face_i], element_i.node_list[face_i+1]]
			else:
				elem_i_nodes = [element_i.node_list[face_i], element_i.node_list[0]]

			# Loop over the faces of element j
			for face_j in range(3):

				# Get the two nodes that form this face
				if face_j < 2:
					elem_j_nodes = [element_j.node_list[face_j], element_j.node_list[face_j+1]]
				else:
					elem_j_nodes = [element_j.node_list[face_j], element_j.node_list[0]]

				# Check for a match. If one is found, then set the neighbors for both elements
				# accordingly
				if (elem_i_nodes[0] == elem_j_nodes[0] and elem_i_nodes[1] == elem_j_nodes[1]) or \
					(elem_i_nodes[0] == elem_j_nodes[1] and elem_i_nodes[1] == elem_j_nodes[0]):

					# Found a matching face
					if element_i.element_refinement_info["color"] == "green" and \
						element_j.element_refinement_info["color"] == "green":
						# The new connection found is between two green elements. 
						# Set the additional connectivity information for this case

						# Element i information
						element_i.element_refinement_info["element_face_index"] = face_i
						element_i.element_refinement_info["neighbor"] = element_j
						element_i.element_refinement_info["neighbor_face_index"] = face_j

						# Element j information
						element_j.element_refinement_info["element_face_index"] = face_j
						element_j.element_refinement_info["neighbor"] = element_i
						element_j.element_refinement_info["neighbor_face_index"] = face_i



	def __str__(self):

		node_list_points = [(node.x, node.y) for node in self.node_list]
		return "node_points : %s, color : %s" % (node_list_points, self.element_refinement_info["color"])


	def print_element_refinement_info(self):

		info_str = "c = %s, element_face_index = %d, neighbor = %s, neighbor_face_index = %d, shared_hanging_node = %s" %\
			(self.element_refinement_info["color"], self.element_refinement_info["element_face_index"], 
				self.element_refinement_info["neighbor"], self.element_refinement_info["neighbor_face_index"],
				self.element_refinement_info["shared_hanging_node"])

		print info_str



	def refine_element(self):
		
		"""Refine an element by following the red-green algorithm. In this
		algorithm, we will refine the given element and accordingly the
		elements that neighbor it to remove all hanging nodes.
		
		Returns:
		    Tuple: List of newly generated elements and List of outdated elements 
		    	(that have now been refined)
		
		Raises:
		    ValueError: Description
		"""
		
		if self.element_refinement_info["color"] == "red":
			newly_generated_elements, outdated_refined_elements = self.refine_red_element()
			return newly_generated_elements, outdated_refined_elements

		elif self.element_refinement_info["color"] == "green":
			newly_generated_elements, outdated_refined_elements = self.refine_green_element()
			return newly_generated_elements, outdated_refined_elements

		else:
			raise ValueError("Unknown element type to refine")


	def refine_red_element(self):
		
		"""Refine an element using the "red" cell approach if it 
		is a red element. This will generate 3 new hanging nodes which
		will then need to be dealt with by creating green elements.

				3
				/\
			   /  \ 
              / e3 \
             / ---- \
            / \e4 /  \
		   /e1 \/  e2 \
		 1 ----------- 2

		"""
		
		# The list to hold the newly generated elements and the elements 
		# that have now been refined and which should be removed.
		newly_generated_elements = []
		outdated_refined_elements = []


		# =============================
		#        Node Generation
		# =============================

		# Get the existing nodes
		node_1 = self.node_list[0]
		node_2 = self.node_list[1]
		node_3 = self.node_list[2]

		# Create the new nodes at the midpoints of each face. By default,
		# first set all the nodes to be dof nodes.
		node_12_x, node_12_y = 0.5*(node_2.x - node_1.x) + node_1.x, \
								0.5*(node_2.y - node_1.y) + node_1.y
		node_12 = node.Node("dof", node_12_x, node_12_y)

		node_23_x, node_23_y = 0.5*(node_3.x - node_2.x) + node_2.x, \
								0.5*(node_3.y - node_2.y) + node_2.y
		node_23 = node.Node("dof", node_23_x, node_23_y)		

		node_31_x, node_31_y = 0.5*(node_1.x - node_3.x) + node_3.x, \
								0.5*(node_1.y - node_3.y) + node_3.y
		node_31 = node.Node("dof", node_31_x, node_31_y)


		# =============================
		#       Element Generation
		# =============================

		# Create the 4 new elements.
		# - Note that the ordering of the nodes is "counterclockwise"
		# 	and always starting at the node that preexisted. For the middle element, 
		# 	the first node will be node_12. This can be visualized in the schematic shown above.
		# - All elements are of order P = 1.

		# Element with node 1
		node_list_elem_1 = [node_1, node_12, node_31]
		elem_1 = TriFiniteElement(1, node_list_elem_1)

		# Element with node 2
		node_list_elem_2 = [node_2, node_23, node_12]
		elem_2 = TriFiniteElement(1, node_list_elem_2)

		# Element with node 3
		node_list_elem_3 = [node_3, node_31, node_23]
		elem_3 = TriFiniteElement(1, node_list_elem_3)

		# Element in middle
		node_list_elem_4 = [node_12, node_23, node_31]
		elem_4 = TriFiniteElement(1, node_list_elem_4)


		# Set the connectivity of the interior elements first. The connectivity of these
		# child elements with the green elements will be set automatically in the method
		# that generates the green elements

		child_elements = [elem_1, elem_2, elem_3, elem_4]
		for child_elem_i in child_elements:
			for child_elem_j in child_elements:
				if child_elem_i != child_elem_j:
					TriFiniteElement.set_element_connectivity(child_elem_i, child_elem_j)


		# Add these new child elements to the cumulative list of newly 
		# generated elements
		for child in child_elements:
			newly_generated_elements.append(child)


		# This parent element was refined into four children. It is now
		# outdated and will be removed from the mesh element list
		outdated_refined_elements.append(self)


		# =============================
		#     Hanging Node Removal
		# =============================

		# Create green elements on the other outer neighboring
		# elements in order to remove the hanging nodes

		# Loop over all the faces and bisect the outer neighbors
		for face_index in range(3):

			# Get the hanging node at the middle of the face and the child elements
			# that have been generated and that touch the given face of the 
			# original parent element.

			if face_index == 0:
				
				mid_node = node_12
				child_elements_face = [elem_1, elem_2]

			elif face_index == 1:
				
				mid_node = node_23
				child_elements_face = [elem_2, elem_3]

			elif face_index == 2:

				mid_node = node_31
				child_elements_face = [elem_3, elem_1]
	

			# Get the element that neighbors this parent element (that has been refined) 
			# at the given face. If the face is None, then we are at a boundary and 
			# nothing needs to be done. Otherwise, split the element that is there into 
			# two green elements to remove the hanging node.

			if self.neighbors[face_index] is not None:

				outer_neighbor_element_to_split_to_greens = self.neighbors[face_index][0]
				outer_neighbor_element_to_split_to_greens_face_index = self.neighbors[face_index][1]

				# Split the outer element by bisecting it and creating two green
				# elements. Then, get the list of new elements created (which are green)
				# and the list of elements to be removed (the old parent element)
				new_element_list_hanging_node, remove_element_list_hanging_node = \
					outer_neighbor_element_to_split_to_greens.refine_element_remove_hanging_node(mid_node, 
						child_elements_face, outer_neighbor_element_to_split_to_greens_face_index)


				# Add the newly created elements to the list of newly
				# generated elements
				for e in new_element_list_hanging_node:
					newly_generated_elements.append(e)

				# Add the elements to be removed
				for e in remove_element_list_hanging_node:
					outdated_refined_elements.append(e)


		return newly_generated_elements, outdated_refined_elements


	def refine_element_remove_hanging_node(self, hanging_node, child_elements, face_index):
		
		"""Refine an element to remove a hanging node. That is, turn an 
		element into two green elements. Do this by bisecting
		a given element using the new hanging node at the element midpoint
		and the vertex opposite the given face.
		
		Args:
		    hanging_node (obj): The new hanging node that must be removed
		    child_elements (list): List of elements (new children) on the
		    	face that contains the hanging node.
		    face_index (int): Index of the face on which the hanging node lies (this
		    	is the face which must be bisected)
		
		Returns:
		    tuple: (List of newly generated elements, List of outdated refined elements)
		
		Raises:
		    ValueError: If the input face_index is bigger than 2
		"""

		if self.element_refinement_info["color"] == "red":

			# Refining a red element to remove a hanging node

			# Lists to hold the new elements generated and the elements that have 
			# been refined and need to be removed
			newly_generated_elements = []
			outdated_refined_elements = []


			# Get the node opposite to the hanging node
			if face_index == 0:
				oppposite_face_node = self.node_list[2]

			elif face_index == 1:
				oppposite_face_node = self.node_list[0]

			elif face_index == 2:
				oppposite_face_node = self.node_list[1]

			else:
				raise ValueError("Unsupported face index")


			# The remaining nodes (2) other than the one that is opposite to 
			# the hanging node.
			remaining_nodes = [] 
			for n in self.node_list:
				if n != oppposite_face_node:
					remaining_nodes.append(n)


			# =============================
			#    Green Element Generation
			# =============================
			new_green_element_list = []

			# Create the green elements. Each green element will have the hanging node,
			# the opposite_face_node and one of the remaining_nodes as its nodes.
			for green_element_index in range(2):
				
				green_element_node_list = [remaining_nodes[green_element_index], 
					hanging_node, oppposite_face_node]

				green_element = TriFiniteElement(1, green_element_node_list)
				
				# Set the element's refinement color to be "green".
				green_element.element_refinement_info["color"] = "green"
				green_element.element_refinement_info["shared_hanging_node"] = hanging_node

				new_green_element_list.append(green_element)

			TriFiniteElement.set_green_element_connectivity_info(new_green_element_list[0], new_green_element_list[1])


			
			# =============================
			#      Connectivity Setup
			# =============================

			# Set the connectivity information for the new green elements

			# 1) Set connectivity information between the green elements
			TriFiniteElement.set_element_connectivity(new_green_element_list[0],
				new_green_element_list[1])

			# 2) Set connectivity information between the green elements and the
			# 	outer elements. When doing this, it is important to note what are the
			# 	outer elements. These are the elements that touched the original parent
			#	element that is currently being split into two green elements. On 2 sides
			# 	of the element (assuming the parent was not on a boundary) there should be
			#	a single element touching the face. On a single side (with the hanging node)
			# 	there will be two elements.

			# Collect (maximum number of 4) outer neighbor elements 
			outer_neighbor_elements = []

			# The child elements on the face 
			for c in child_elements:
				outer_neighbor_elements.append(c)

			# The remaining two elements
			for i in range(3):
				if i != face_index:
					if self.neighbors[i] is not None:
						outer_neighbor_elements.append(self.neighbors[i][0])

			# Set the connectivity with the outer neighbor elements
			for green_element in new_green_element_list:
				for outer_neighbor_element in outer_neighbor_elements:
					if outer_neighbor_element != green_element:
						TriFiniteElement.set_element_connectivity(green_element, outer_neighbor_element)


			# Get the final list of newly generated elements and refined (and now
			# outdated) elements

			for e in new_green_element_list:
				newly_generated_elements.append(e)

			outdated_refined_elements.append(self)

			return newly_generated_elements, outdated_refined_elements

		elif self.element_refinement_info["color"] == "green":
			
			# Refining a green element to remove a hanging node. There are
			# two cases to consider here:

			# Case 1: The node being added is on a face that does not have the
			#	original hanging node (the node that was removed by the generation
			#	of these green elements).

			# Case 2: The node being added is on a face that also holds the
			# 	hanging node. 


			# =============================
			#            Setup
			# =============================


			# Lists to hold the new elements generated and the elements that have 
			# been refined and need to be removed
			newly_generated_elements = []
			outdated_refined_elements = []

			#pdb.set_trace()

			self_green_neighbor_face_index = self.element_refinement_info["element_face_index"]
			green_neighbor_self_face_index = self.element_refinement_info["neighbor_face_index"]
			green_neighbor_element = self.element_refinement_info["neighbor"]

			shared_hanging_node = self.element_refinement_info["shared_hanging_node"]

			# Find what case we are in
			refinement_case = "Case 1"

			# If one of the child elements holds the shared hanging node then we 
			# are in case 2
			for e in child_elements:
				if shared_hanging_node in e.node_list:
					refinement_case = "Case 2"


			if refinement_case == "Case 1":

				# We are refining a green element to remove a hanging node on a face that
				# does not hold the original hanging node.

				# =============================
				#       Node Generation
				# =============================

				# 1) The hanging node on this green element's outer face. It is given to us.
				self_outer_hanging_node = hanging_node


				# 2) The hanging node on the green element's neighbor outer face
				green_neighbor_outer_hanging_node = None


				for green_neighbor_face_i in range(3):

					# Get the nodes on this face of the neighbor element. If it doesn't
					# hold the shared hanging node, then we are on the right outer face
					if shared_hanging_node not in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):

						face_nodes = green_neighbor_element.get_nodes_on_face(green_neighbor_face_i)

						new_node_x, node_node_y = 0.5*(face_nodes[1].x - face_nodes[0].x) + face_nodes[0].x, \
												0.5*(face_nodes[1].y - face_nodes[0].y) + face_nodes[0].y

						green_neighbor_outer_hanging_node = node.Node("dof", new_node_x, node_node_y)


				# =============================
				#       Element Generation
				# =============================

				# Generate the 4 elements that will replace these two green elements

				# Element 1: This element is on the current green element
				# 	and it is also on the bisected edge with the hanging node.
				# 	Is made up of shared_hanging_node, self_outer_hanging_node
				# 	and the correct corner node on the bisected face

				elem_1_node_list = []
				elem_1_node_list.append(shared_hanging_node)
				elem_1_node_list.append(self_outer_hanging_node)

				for self_face_i in range(3):
					
					# Loop over the faces of this green element

					# We wish to now land on the face that was bisected due
					# to the shared hanging node. Then, get the node on this
					# face (other than the shared hanging node) as this is the last
					# node for element 1

					if self_face_i != self_green_neighbor_face_index:
						if shared_hanging_node in self.get_nodes_on_face(self_face_i):
							for n in self.get_nodes_on_face(self_face_i):
								if n != shared_hanging_node:
									elem_1_node_list.append(n)

				elem_1 = TriFiniteElement(1, elem_1_node_list)


				# Element 2: This element is on the neighbor green element
				# 	and it is also on the bisected edge with the shared hanging node.
				# 	Is made up of shared_hanging_node, green_neighbor_outer_hanging_node
				# 	and the corner element on the bisected face

				elem_2_node_list = []
				elem_2_node_list.append(shared_hanging_node)
				elem_2_node_list.append(green_neighbor_outer_hanging_node)

				for green_neighbor_face_i in range(3):
					
					# Loop over the faces of the neighbor green element

					# As before, we wish now to land on the face that was bisected due
					# to the original hanging node. Then, get the node on this
					# face (other than the hanging node) as this forms the last
					# node for element 2

					if green_neighbor_face_i != green_neighbor_self_face_index:
						if shared_hanging_node in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
							for n in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
								if n != shared_hanging_node:
									elem_2_node_list.append(n)

				elem_2 = TriFiniteElement(1, elem_2_node_list)


				# Element 3: This element is half in one green element and half 
				# 	in the other. It is made up of the green_neighbor_outer_hanging_node, 
				#	self_outer_hanging_node and shared node between the green 
				#	elements that is not the shared_hanging_node.

				elem_3_node_list = []
				elem_3_node_list.append(green_neighbor_outer_hanging_node)
				elem_3_node_list.append(self_outer_hanging_node)

				for n in self.node_list:
					
					# Loop over the nodes of this green element. If it is in the
					# neighbor green element and it is not the shared_hanging_node
					# then we have found the third node

					if n != shared_hanging_node:
						if n in green_neighbor_element.node_list:
							elem_3_node_list.append(n)

				elem_3 = TriFiniteElement(1, elem_3_node_list)


				# Element 4: This element is directly in the middle of the original
				# 	element. It consists of all the hanging nodes.

				elem_4_node_list = []
				elem_4_node_list.append(green_neighbor_outer_hanging_node)
				elem_4_node_list.append(self_outer_hanging_node)
				elem_4_node_list.append(shared_hanging_node)

				elem_4 = TriFiniteElement(1, elem_4_node_list)


				# Set the connectivity of the new interior elements first. The connectivity of these
				# child elements with the green elements will be set automatically in the method
				# that generates the green elements

				child_elements_from_greens = [elem_1, elem_2, elem_3, elem_4]
				for child_elem_i in child_elements_from_greens:
					for child_elem_j in child_elements_from_greens:
						if child_elem_i != child_elem_j:
							TriFiniteElement.set_element_connectivity(child_elem_i, child_elem_j)


				# Set the connectivity between the child elements and the 
				# outer element child elements (the face with two nodes)

				outer_element_child_elements = []

				# The child elements for the new children generated (that
				# just bisected the face resulting in the spliting of this element
				# to remove the hanging node)
				for e in child_elements:
					outer_element_child_elements.append(e)


				# The child elements for the original face that was bisected, creating 
				# a hanging node, which was removed by the generation of these
				# green elements in the past.
				for self_face_i in range(3):
					
					self_face_i_nodes = self.get_nodes_on_face(self_face_i)
					if shared_hanging_node in self_face_i_nodes:
						# We are on a face with the shared hanging node. 

						if self_face_i_nodes[0] == shared_hanging_node:
							face_other_node = self_face_i_nodes[1]
						else:
							face_other_node = self_face_i_nodes[0]

						if face_other_node not in green_neighbor_element.node_list:

							# Found the face that borders one of the child elements
							if self.neighbors[self_face_i] is not None:
								outer_element_child_elements.append(self.neighbors[self_face_i][0])
						

				for green_neighbor_face_i in range(3):
					
					green_neighbor_face_i_nodes = green_neighbor_element.get_nodes_on_face(green_neighbor_face_i)
					if shared_hanging_node in green_neighbor_face_i_nodes:
						# We are on a face with the shared hanging node. 

						if green_neighbor_face_i_nodes[0] == shared_hanging_node:
							face_other_node = green_neighbor_face_i_nodes[1]
						else:
							face_other_node = green_neighbor_face_i_nodes[0]

						if face_other_node not in self.node_list:

							# Found the face that borders one of the child elements
							if green_neighbor_element.neighbors[green_neighbor_face_i] is not None:
								outer_element_child_elements.append(green_neighbor_element.neighbors[green_neighbor_face_i][0])


				for child_elem_i in child_elements_from_greens:
					for outer_child_element in outer_element_child_elements:
						if child_elem_i != outer_child_element:
							TriFiniteElement.set_element_connectivity(child_elem_i, outer_child_element)


				# Add these new child elements to the list of new elements and add the 
				# old green elements to the list of elements to be removed
				for n in child_elements_from_greens:
					newly_generated_elements.append(n)

				outdated_refined_elements.append(self)
				outdated_refined_elements.append(green_neighbor_element)


				# =============================
				#     Hanging Node Removal
				# =============================

				# Create new green elements for the outer elements that now need to be split
				# due to the new hanging node. Only one hanging node has been generated in
				# this process (green_neighbor_outer_hanging_node)

				# Green Element 1: The outer element for the neighbor green element will 
				#	need to be split. It's middle node is green_neighbor_outer_hanging_node. It's two
				#	child elements will be given by the elements that are in the neighbor green element

				mid_node = green_neighbor_outer_hanging_node
				
				child_elements_face = []
				child_elements_face.append(elem_2)
				child_elements_face.append(elem_3)

				# Identify the index of the outer face for the green element
				for green_neighbor_face_i in range(3):
					if shared_hanging_node not in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
						face_to_outer_elem = green_neighbor_face_i

				# Get the element to be split and split it (if it exists (we are not on a boundary))
				if green_neighbor_element.neighbors[face_to_outer_elem] is not None:

					outer_neighbor_element_to_split_to_greens = green_neighbor_element.neighbors[face_to_outer_elem][0]
					outer_neighbor_element_to_split_to_greens_face_index = green_neighbor_element.neighbors[face_to_outer_elem][1]

					# Split the outer element by bisecting it and creating two green
					# elements. Then, get the list of new elements created (which are green)
					# and the list of elements to be removed (the old parent element)
					new_element_list_hanging_node, remove_element_list_hanging_node = \
						outer_neighbor_element_to_split_to_greens.refine_element_remove_hanging_node(mid_node, 
							child_elements_face, outer_neighbor_element_to_split_to_greens_face_index)


					# Add the newly created elements to the list of newly
					# generated elements
					for e in new_element_list_hanging_node:
						newly_generated_elements.append(e)

					# Add the elements to be removed
					for e in remove_element_list_hanging_node:
						outdated_refined_elements.append(e)

				return newly_generated_elements, outdated_refined_elements


			elif refinement_case == "Case 2":

				# First, we will combine the green elements and split it into 4.
				# Then, once this is done, the hanging node that has been input into
				# this function will be removed


				# =============================
				#       Node Generation
				# =============================

				# 1) The hanging node on this green element's outer face.
				self_outer_hanging_node = None

				for self_face_i in range(3):

					# Get the nodes on this face of the self element. If it
					# doesn't hold the shared hanging node, then we are on the
					# right outer face.

					if shared_hanging_node not in self.get_nodes_on_face(self_face_i):

						face_nodes = self.get_nodes_on_face(self_face_i)
						new_node_x, node_node_y = 0.5*(face_nodes[1].x - face_nodes[0].x) + face_nodes[0].x, \
												0.5*(face_nodes[1].y - face_nodes[0].y) + face_nodes[0].y
						self_outer_hanging_node = node.Node("dof", new_node_x, node_node_y)


				# 2) The hanging node on the green element's neighbor outer face
				green_neighbor_outer_hanging_node = None

				for green_neighbor_face_i in range(3):

					# Get the nodes on this face of the neighbor element. If it doesn't
					# hold the shared hanging node, then we are on the right outer face
					if shared_hanging_node not in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):

						face_nodes = green_neighbor_element.get_nodes_on_face(green_neighbor_face_i)
						new_node_x, node_node_y = 0.5*(face_nodes[1].x - face_nodes[0].x) + face_nodes[0].x, \
												0.5*(face_nodes[1].y - face_nodes[0].y) + face_nodes[0].y
						green_neighbor_outer_hanging_node = node.Node("dof", new_node_x, node_node_y)


				# =============================
				#       Element Generation
				# =============================


				# Generate the 4 elements that will replace these two green elements

				# Element 1: This element is on the current green element
				# 	and it is also on the bisected edge with the hanging node.
				# 	Is made up of shared_hanging_node, self_outer_hanging_node
				# 	and the correct corner node on the bisected face

				elem_1_node_list = []
				elem_1_node_list.append(shared_hanging_node)
				elem_1_node_list.append(self_outer_hanging_node)

				for self_face_i in range(3):
					
					# Loop over the faces of this green element

					# We wish to now land on the face that was bisected due
					# to the shared hanging node. Then, get the node on this
					# face (other than the shared hanging node) as this is the last
					# node for element 1

					if self_face_i != self_green_neighbor_face_index:
						if shared_hanging_node in self.get_nodes_on_face(self_face_i):
							for n in self.get_nodes_on_face(self_face_i):
								if n != shared_hanging_node:
									elem_1_node_list.append(n)

				elem_1 = TriFiniteElement(1, elem_1_node_list)


				# Element 2: This element is on the neighbor green element
				# 	and it is also on the bisected edge with the shared hanging node.
				# 	Is made up of shared_hanging_node, green_neighbor_outer_hanging_node
				# 	and the corner element on the bisected face

				elem_2_node_list = []
				elem_2_node_list.append(shared_hanging_node)
				elem_2_node_list.append(green_neighbor_outer_hanging_node)

				for green_neighbor_face_i in range(3):
					
					# Loop over the faces of the neighbor green element

					# As before, we wish now to land on the face that was bisected due
					# to the original hanging node. Then, get the node on this
					# face (other than the hanging node) as this forms the last
					# node for element 2

					if green_neighbor_face_i != green_neighbor_self_face_index:
						if shared_hanging_node in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
							for n in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
								if n != shared_hanging_node:
									elem_2_node_list.append(n)

				elem_2 = TriFiniteElement(1, elem_2_node_list)


				# Element 3: This element is half in one green element and half 
				# 	in the other. It is made up of the green_neighbor_outer_hanging_node, 
				#	self_outer_hanging_node and shared node between the green 
				#	elements that is not the shared_hanging_node.

				elem_3_node_list = []
				elem_3_node_list.append(green_neighbor_outer_hanging_node)
				elem_3_node_list.append(self_outer_hanging_node)

				for n in self.node_list:
					
					# Loop over the nodes of this green element. If it is in the
					# neighbor green element and it is not the shared_hanging_node
					# then we have found the third node

					if n != shared_hanging_node:
						if n in green_neighbor_element.node_list:
							elem_3_node_list.append(n)

				elem_3 = TriFiniteElement(1, elem_3_node_list)


				# Element 4: This element is directly in the middle of the original
				# 	element. It consists of all the hanging nodes.

				elem_4_node_list = []
				elem_4_node_list.append(green_neighbor_outer_hanging_node)
				elem_4_node_list.append(self_outer_hanging_node)
				elem_4_node_list.append(shared_hanging_node)

				elem_4 = TriFiniteElement(1, elem_4_node_list)


				# Set the connectivity of the new interior elements first. The connectivity of these
				# child elements with the green elements will be set automatically in the method
				# that generates the green elements

				child_elements_from_greens = [elem_1, elem_2, elem_3, elem_4]
				for child_elem_i in child_elements_from_greens:
					for child_elem_j in child_elements_from_greens:
						if child_elem_i != child_elem_j:
							TriFiniteElement.set_element_connectivity(child_elem_i, child_elem_j)


				# On the face with the shared hanging node, one side has been refined (generating the child
				# elements input to the function) and the other still has a full element. 
				# Find this element (unrefined) and set the connectivity with it. This element is
				# touching the green_neighbor element

				outer_element_child_elements = []
						
				for green_neighbor_face_i in range(3):
					
					green_neighbor_face_i_nodes = green_neighbor_element.get_nodes_on_face(green_neighbor_face_i)
					if shared_hanging_node in green_neighbor_face_i_nodes:
						# We are on a face with the shared hanging node. 

						if green_neighbor_face_i_nodes[0] == shared_hanging_node:
							face_other_node = green_neighbor_face_i_nodes[1]
						else:
							face_other_node = green_neighbor_face_i_nodes[0]

						if face_other_node not in self.node_list:

							# Found the face that borders one of the old child elements
							if green_neighbor_element.neighbors[green_neighbor_face_i] is not None:
								outer_element_child_elements.append(green_neighbor_element.neighbors[green_neighbor_face_i][0])


				for child_elem_i in child_elements_from_greens:
					for outer_child_element in outer_element_child_elements:
						if child_elem_i != outer_child_element:
							TriFiniteElement.set_element_connectivity(child_elem_i, outer_child_element)


				# =============================
				#     Hanging Node Removal
				# =============================

				# Create new green elements for the outer elements that now need to be split
				# due to the new hanging node. Two hanging nodes have been generated in this 
				# process.


				# Green Element 1: The outer element for the current green element will need 
				#	to be split. It's middle node is self_outer_hanging_node. It's two 
				#	child elements will be given by the elements that are in this green element (self)

				mid_node = self_outer_hanging_node

				child_elements_face = []
				child_elements_face.append(elem_1)
				child_elements_face.append(elem_3)

				# Get the face on this (self) green element that bounds the outer element.
				# This face can be identified by finding the one that does not hold the 
				# shared hanging node.
				for self_face_i in range(3):
					if shared_hanging_node not in self.get_nodes_on_face(self_face_i):
						face_to_outer_elem = self_face_i

				# Get the element to be split and split it (if it exists (we are not on a boundary))
				if self.neighbors[face_to_outer_elem] is not None:

					outer_neighbor_element_to_split_to_greens = self.neighbors[face_to_outer_elem][0]
					outer_neighbor_element_to_split_to_greens_face_index = self.neighbors[face_to_outer_elem][1]

					# Split the outer element by bisecting it and creating two green
					# elements. Then, get the list of new elements created (which are green)
					# and the list of elements to be removed (the old parent element)
					new_element_list_hanging_node, remove_element_list_hanging_node = \
						outer_neighbor_element_to_split_to_greens.refine_element_remove_hanging_node(mid_node, 
							child_elements_face, outer_neighbor_element_to_split_to_greens_face_index)

					# Add the newly created elements to the list of newly
					# generated elements
					for e in new_element_list_hanging_node:
						newly_generated_elements.append(e)

					# Add the elements to be removed
					for e in remove_element_list_hanging_node:
						outdated_refined_elements.append(e)


				# Green Element 2: The outer element for the neighbor green element will also 
				#	need to be split. It's middle node is green_neighbor_outer_hanging_node. It's two
				#	child elements will be given by the elements that are in the neighbor's green element

				mid_node = green_neighbor_outer_hanging_node
				
				child_elements_face = []
				child_elements_face.append(elem_2)
				child_elements_face.append(elem_3)

				# As before, identify the index of the outer face for the green element
				for green_neighbor_face_i in range(3):
					if shared_hanging_node not in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
						face_to_outer_elem = green_neighbor_face_i

				# Get the element to be split and split it (if it exists (we are not on a boundary))
				if green_neighbor_element.neighbors[face_to_outer_elem] is not None:

					outer_neighbor_element_to_split_to_greens = green_neighbor_element.neighbors[face_to_outer_elem][0]
					outer_neighbor_element_to_split_to_greens_face_index = green_neighbor_element.neighbors[face_to_outer_elem][1]

					# Split the outer element by bisecting it and creating two green
					# elements. Then, get the list of new elements created (which are green)
					# and the list of elements to be removed (the old parent element)
					new_element_list_hanging_node, remove_element_list_hanging_node = \
						outer_neighbor_element_to_split_to_greens.refine_element_remove_hanging_node(mid_node, 
							child_elements_face, outer_neighbor_element_to_split_to_greens_face_index)


					# Add the newly created elements to the list of newly
					# generated elements
					for e in new_element_list_hanging_node:
						newly_generated_elements.append(e)

					# Add the elements to be removed
					for e in remove_element_list_hanging_node:
						outdated_refined_elements.append(e)


				# =============================
				# Original Hanging Node Removal
				# =============================

				# Remove the original hanging node now. This will be done by splitting one of the
				# child elements (which is red) into two green elements

				child_element_from_greens_to_split = None

				for c_green in child_elements_from_greens:

					# Go through the nodes of the child element that has been
					# split from the green elements. We have found a match if
					# two of the nodes in the child elements from the greens
					# is in the child elements (outer face) that we are removing
					# the hanging node for

					matches_found = 0

					for c_outer in child_elements:
						for n in c_outer.node_list:
							if n in c_green.node_list:
								matches_found += 1

					if matches_found == 2:
						child_element_from_greens_to_split = c_green
						break


				if child_element_from_greens_to_split is None:
					raise ValueError("Child Element Match not found")

				
				# Find the face on the child_element_from_greens_to_split that will
				# need to be bisected due to the hanging node to be removed
				face_to_bisect_child_elements_from_greens_to_split = None

				for i in range(3):

					face_nodes = child_element_from_greens_to_split.get_nodes_on_face(i)

					matches_found = 0
					for c_outer in child_elements:
						for n in c_outer.node_list:
							if n in face_nodes:
								matches_found += 1

					if matches_found == 2:
						face_to_bisect_child_elements_from_greens_to_split = i
						break


				if face_to_bisect_child_elements_from_greens_to_split is None:
					raise ValueError("Could not find face to bisect")


				new_element_list_hanging_node, remove_element_list_hanging_node = \
					child_element_from_greens_to_split.refine_element_remove_hanging_node(hanging_node, 
						child_elements, face_to_bisect_child_elements_from_greens_to_split)


				# Finally, now collect all the new elements that have been generated
				# and the elements that have been removed. The green elements from the
				# other faces have already been added to these lists.

				# New child elements (3 red triangles and 2 green triangles)
				for c_green in child_elements_from_greens:
					if c_green != child_element_from_greens_to_split:
						newly_generated_elements.append(c_green)

				for n in new_element_list_hanging_node:
					newly_generated_elements.append(n)


				# Elements to remove
				outdated_refined_elements.append(self)
				outdated_refined_elements.append(green_neighbor_element)


				return newly_generated_elements, outdated_refined_elements


		else:
			raise ValueError("Unsupported Element Color")


	def refine_green_element(self):
		
		"""Refine a green element. Do this by finding the green element's
		partner (the other green element) and temporarily combine both. Then,
		split the resulting element as a red element would be split. In this case
		however, make sure to not generate three new nodes (we will use 
		a prexisting node).
		"""

		# Lists to hold the new elements generated and the elements that have 
		# been refined and need to be removed
		newly_generated_elements = []
		outdated_refined_elements = []


		self_green_neighbor_face_index = self.element_refinement_info["element_face_index"]
		green_neighbor_self_face_index = self.element_refinement_info["neighbor_face_index"]
		green_neighbor_element = self.element_refinement_info["neighbor"]
		
		
		# =============================
		#  Node Collection/Generation
		# =============================

		# Collect and generate the hanging nodes. We say hanging, but these
		# nodes just correspond to those that bisect the element faces.

		# 1) Collect the shared hanging node between the green elements. This
		# 	is the node that was a hanging node that had to be removed using
		#	these green elements

		shared_hanging_node = self.element_refinement_info["shared_hanging_node"]

		# Now, create the new hanging nodes (that will need to be removed) on 
		# the outer faces of the two green elements

		# 2) The hanging node on this green element's outer face
		self_outer_hanging_node = None

		for self_face_i in range(3):

			# Get the nodes on this face. If it doesn't hold the 
			# shared hanging node, then we are on an outer face of the element
			if shared_hanging_node not in self.get_nodes_on_face(self_face_i):

				face_nodes = self.get_nodes_on_face(self_face_i)

				new_node_x, node_node_y = 0.5*(face_nodes[1].x - face_nodes[0].x) + face_nodes[0].x, \
										0.5*(face_nodes[1].y - face_nodes[0].y) + face_nodes[0].y
				self_outer_hanging_node = node.Node("dof", new_node_x, node_node_y)


		# 3) The hanging node on the green element's neighbor outer face
		green_neighbor_outer_hanging_node = None

		for green_neighbor_face_i in range(3):

			# Get the nodes on this face of the neighbor element. If it doesn't
			# hold the shared hanging node, then we are on the right outer face
			if shared_hanging_node not in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):

				face_nodes = green_neighbor_element.get_nodes_on_face(green_neighbor_face_i)

				new_node_x, node_node_y = 0.5*(face_nodes[1].x - face_nodes[0].x) + face_nodes[0].x, \
										0.5*(face_nodes[1].y - face_nodes[0].y) + face_nodes[0].y
				green_neighbor_outer_hanging_node = node.Node("dof", new_node_x, node_node_y)


		# =============================
		#       Element Generation
		# =============================

		# Generate the 4 elements that will replace these two green elements


		# Element 1: This element is on the current green element
		# 	and it is also on the bisected edge with the hanging node.
		# 	Is made up of shared_hanging_node, self_outer_hanging_node
		# 	and the correct corner node on the bisected face

		elem_1_node_list = []
		elem_1_node_list.append(shared_hanging_node)
		elem_1_node_list.append(self_outer_hanging_node)

		for self_face_i in range(3):
			
			# Loop over the faces of this green element

			# We wish to now land on the face that was bisected due
			# to the shared hanging node. Then, get the node on this
			# face (other than the shared hanging node) as this is the last
			# node for element 1

			if self_face_i != self_green_neighbor_face_index:
				if shared_hanging_node in self.get_nodes_on_face(self_face_i):
					for n in self.get_nodes_on_face(self_face_i):
						if n != shared_hanging_node:
							elem_1_node_list.append(n)

		elem_1 = TriFiniteElement(1, elem_1_node_list)


		# Element 2: This element is on the neighbor green element
		# 	and it is also on the bisected edge with the shared hanging node.
		# 	Is made up of shared_hanging_node, green_neighbor_outer_hanging_node
		# 	and the corner element on the bisected face

		elem_2_node_list = []
		elem_2_node_list.append(shared_hanging_node)
		elem_2_node_list.append(green_neighbor_outer_hanging_node)

		for green_neighbor_face_i in range(3):
			
			# Loop over the faces of the neighbor green element

			# As before, we wish now to land on the face that was bisected due
			# to the original hanging node. Then, get the node on this
			# face (other than the hanging node) as this forms the last
			# node for element 2

			if green_neighbor_face_i != green_neighbor_self_face_index:
				if shared_hanging_node in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
					for n in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
						if n != shared_hanging_node:
							elem_2_node_list.append(n)

		elem_2 = TriFiniteElement(1, elem_2_node_list)


		# Element 3: This element is half in one green element and half 
		# 	in the other. It is made up of the green_neighbor_outer_hanging_node, 
		#	self_outer_hanging_node and shared node between the green 
		#	elements that is not the shared_hanging_node.

		elem_3_node_list = []
		elem_3_node_list.append(green_neighbor_outer_hanging_node)
		elem_3_node_list.append(self_outer_hanging_node)

		for n in self.node_list:
			
			# Loop over the nodes of this green element. If it is in the
			# neighbor green element and it is not the shared_hanging_node
			# then we have found the third node

			if n != shared_hanging_node:
				if n in green_neighbor_element.node_list:
					elem_3_node_list.append(n)

		elem_3 = TriFiniteElement(1, elem_3_node_list)


		# Element 4: This element is directly in the middle of the original
		# 	element. It consists of all the hanging nodes.

		elem_4_node_list = []
		elem_4_node_list.append(green_neighbor_outer_hanging_node)
		elem_4_node_list.append(self_outer_hanging_node)
		elem_4_node_list.append(shared_hanging_node)

		elem_4 = TriFiniteElement(1, elem_4_node_list)


		# Set the connectivity of the new interior elements first. The connectivity of these
		# child elements with the green elements will be set automatically in the method
		# that generates the green elements

		child_elements = [elem_1, elem_2, elem_3, elem_4]
		for child_elem_i in child_elements:
			for child_elem_j in child_elements:
				if child_elem_i != child_elem_j:
					TriFiniteElement.set_element_connectivity(child_elem_i, child_elem_j)


		# Set the connectivity between the child elements and the 
		# outer element child elements (the face with two nodes)
		outer_element_child_elements = []

		for self_face_i in range(3):
			
			self_face_i_nodes = self.get_nodes_on_face(self_face_i)
			if shared_hanging_node in self_face_i_nodes:
				# We are on a face with the shared hanging node. 

				if self_face_i_nodes[0] == shared_hanging_node:
					face_other_node = self_face_i_nodes[1]
				else:
					face_other_node = self_face_i_nodes[0]

				if face_other_node not in green_neighbor_element.node_list:

					# Found the face that borders one of the child elements
					if self.neighbors[self_face_i] is not None:
						outer_element_child_elements.append(self.neighbors[self_face_i][0])
				
		for green_neighbor_face_i in range(3):
			
			green_neighbor_face_i_nodes = green_neighbor_element.get_nodes_on_face(green_neighbor_face_i)
			if shared_hanging_node in green_neighbor_face_i_nodes:
				# We are on a face with the shared hanging node. 

				if green_neighbor_face_i_nodes[0] == shared_hanging_node:
					face_other_node = green_neighbor_face_i_nodes[1]
				else:
					face_other_node = green_neighbor_face_i_nodes[0]

				if face_other_node not in self.node_list:

					# Found the face that borders one of the child elements
					if green_neighbor_element.neighbors[green_neighbor_face_i] is not None:
						outer_element_child_elements.append(green_neighbor_element.neighbors[green_neighbor_face_i][0])


		for child_elem_i in child_elements:
			for outer_child_element in outer_element_child_elements:
				if child_elem_i != outer_child_element:
					TriFiniteElement.set_element_connectivity(child_elem_i, outer_child_element)


		# Add these new child elements to the list of new elements and and the 
		# old green elements to the list of elements to be removed
		for n in child_elements:
			newly_generated_elements.append(n)

		outdated_refined_elements.append(self)
		outdated_refined_elements.append(green_neighbor_element)


		# =============================
		#     Hanging Node Removal
		# =============================

		# Create new green elements for the outer elements that now need to be split
		# due to the new hanging node

		# Green Element 1: The outer element for the current green element will need 
		#	to be split. It's middle node is self_outer_hanging_node. It's two 
		#	child elements will be given by the elements that are in this green element (self)

		mid_node = self_outer_hanging_node

		child_elements_face = []
		child_elements_face.append(elem_1)
		child_elements_face.append(elem_3)

		# Get the face on this (self) green element that bounds the outer element.
		# This face can be identified by finding the one that does not hold the 
		# shared hanging node.
		for self_face_i in range(3):
			if shared_hanging_node not in self.get_nodes_on_face(self_face_i):
				face_to_outer_elem = self_face_i

		# Get the element to be split and split it (if it exists (we are not on a boundary))
		if self.neighbors[face_to_outer_elem] is not None:

			outer_neighbor_element_to_split_to_greens = self.neighbors[face_to_outer_elem][0]
			outer_neighbor_element_to_split_to_greens_face_index = self.neighbors[face_to_outer_elem][1]

			# Split the outer element by bisecting it and creating two green
			# elements. Then, get the list of new elements created (which are green)
			# and the list of elements to be removed (the old parent element)
			new_element_list_hanging_node, remove_element_list_hanging_node = \
				outer_neighbor_element_to_split_to_greens.refine_element_remove_hanging_node(mid_node, 
					child_elements_face, outer_neighbor_element_to_split_to_greens_face_index)


			# Add the newly created elements to the list of newly
			# generated elements
			for e in new_element_list_hanging_node:
				newly_generated_elements.append(e)

			# Add the elements to be removed
			for e in remove_element_list_hanging_node:
				outdated_refined_elements.append(e)


		# Green Element 2: The outer element for the neighbor green element will also 
		#	need to be split. It's middle node is green_neighbor_outer_hanging_node. It's two
		#	child elements will be given by the elements that are in the neighbors green element

		mid_node = green_neighbor_outer_hanging_node
		
		child_elements_face = []
		child_elements_face.append(elem_2)
		child_elements_face.append(elem_3)

		# As before, identify the index of the outer face for the green element
		for green_neighbor_face_i in range(3):
			if shared_hanging_node not in green_neighbor_element.get_nodes_on_face(green_neighbor_face_i):
				face_to_outer_elem = green_neighbor_face_i

		# Get the element to be split and split it (if it exists (we are not on a boundary))
		if green_neighbor_element.neighbors[face_to_outer_elem] is not None:

			outer_neighbor_element_to_split_to_greens = green_neighbor_element.neighbors[face_to_outer_elem][0]
			outer_neighbor_element_to_split_to_greens_face_index = green_neighbor_element.neighbors[face_to_outer_elem][1]

			# Split the outer element by bisecting it and creating two green
			# elements. Then, get the list of new elements created (which are green)
			# and the list of elements to be removed (the old parent element)
			new_element_list_hanging_node, remove_element_list_hanging_node = \
				outer_neighbor_element_to_split_to_greens.refine_element_remove_hanging_node(mid_node, 
					child_elements_face, outer_neighbor_element_to_split_to_greens_face_index)


			# Add the newly created elements to the list of newly
			# generated elements
			for e in new_element_list_hanging_node:
				newly_generated_elements.append(e)

			# Add the elements to be removed
			for e in remove_element_list_hanging_node:
				outdated_refined_elements.append(e)



		return newly_generated_elements, outdated_refined_elements


	def compute_metrics(self):
		
		"""Compute the metric terms for this element. For now, set these
		terms using lambda expressions. If slow, then simply compute them 
		at the cubature nodes.
		"""
		
		# Get the gradient of the basis functions on the reference domain
		grad_psi_hat_basis = self.reference_element.grad_psi_hat_basis

		psi_hat_xi  = [psi_grad[0] for psi_grad in grad_psi_hat_basis]
		psi_hat_eta = [psi_grad[1] for psi_grad in grad_psi_hat_basis]

		x_node_vals = [node.x for node in self.node_list]
		y_node_vals = [node.y for node in self.node_list]

		self.x_xi = lambda xi, eta, x_node_vals=x_node_vals, psi_hat_xi=psi_hat_xi: \
			self.basis_expansion(psi_hat_xi, x_node_vals, xi, eta)
		self.y_xi = lambda xi, eta, y_node_vals=y_node_vals, psi_hat_xi=psi_hat_xi: \
			self.basis_expansion(psi_hat_xi, y_node_vals, xi, eta)

		self.x_eta = lambda xi, eta, x_node_vals=x_node_vals, psi_hat_eta=psi_hat_eta: \
			self.basis_expansion(psi_hat_eta, x_node_vals, xi, eta)
		self.y_eta = lambda xi, eta, y_node_vals=y_node_vals, psi_hat_eta=psi_hat_eta: \
			self.basis_expansion(psi_hat_eta, y_node_vals, xi, eta)


		self.J = lambda xi, eta, x_xi=self.x_xi, x_eta=self.x_eta, y_xi=self.y_xi, y_eta=self.y_eta: \
			x_xi(xi, eta)*y_eta(xi, eta) - y_xi(xi, eta)*x_eta(xi, eta)


		# Check that the Jacobian is non-negative at the nodes
		if self.J(0.0, 0.0) <= 0 or self.J(1.0, 0.0) <= 0 or self.J(0.0, 1.0) <= 0: 
			raise ValueError("Negative Jacobian : %f %f %f -> n : %s, %s, %s" % \
				(self.J(0.0, 0.0), self.J(1.0, 0.0), self.J(0.0, 1.0), self.node_list[0], 
					self.node_list[1], self.node_list[2]))


	def mapping_physical_to_reference_domain(self, x, y):
	
		"""Map the point x,y from the physical domain to the reference
		domain on the element
		
		Args:
		    x (float): x-position
		    y (float): y-position
		
		Returns:
		    list, boolean: [xi, eta] values for the mapped point and
		    	True or false for whether or not the point is in the element
		"""

		x_y_vec = np.zeros((2,1))
		x_y_vec[0,0] = x
		x_y_vec[1,0] = y

		xi_eta_vec = np.dot(self.inverse_mat_phys_to_ref_dom, x_y_vec-self.shift_vect_phys_to_ref_dom)

		xi, eta = xi_eta_vec[0,0], xi_eta_vec[1,0]

		in_elem_bool = False
		if xi >= (0 - 1E-8) and eta >= (0 - 1E-8) and (xi + eta) <= (1 + 1E-8):
			in_elem_bool = True

		return [xi, eta], in_elem_bool


	def get_solution_at_position(self, x, y):

		"""Get the solution at the position x,y in the element
		
		Args:
		    x (float): x-position at which the solution is required
		    y (float): y-position at which the solution is requiredn
		
		Returns:
		    float: Value of the solution at the specified location in the element.
		"""

		xi_eta_list, in_elem_bool = self.mapping_physical_to_reference_domain(x, y)

		if in_elem_bool is False:
			raise ValueError("Position not in the element domain")


		# The point is in the domain, so compute the solution at the specified position
		psi_hat_basis = self.reference_element.psi_hat_basis

		u_val = 0.0
		for i in range(3):
			u_val += psi_hat_basis[i](xi_eta_list[0], xi_eta_list[1]) * self.node_list[i].value
		return u_val


	def print_connectivity(self):
		
		"""For testing purpuses. This function simply
		prints the connectivity information for the given element.
		"""

		for face_i in range(3):
			if self.neighbors[face_i] is not None:
				print " f = %d -> %s" % (face_i, self.neighbors[face_i][0])
			else:
				print " f = %d -> None" % (face_i)


	def set_neighbor(self, face_index, element_neighbor, element_neighbor_face_index):
		
		"""Set the neighbor for the given element. The ordering of the face indeces
		is consistent with what is described in the constructor.
		
		Args:
		    face_index (int): The index of the face for which we would like to
		    	assign a neighbor for.
		    element_neighbor (obj): The element that is neighboring this one.
		    element_neighbor_face_index (int): The index of the face on the neighboring
		    	element that is touching this element.
		"""
		
		self.neighbors[face_index] = (element_neighbor, element_neighbor_face_index)


	def get_nodes_on_face(self, face_i):
		
		"""Get the nodes that make up the given face. The ordering
		of the nodes and faces is consistent with the diagram
		shown in the constructor.
		
		Args:
		    face_i (int): The index of the face
		
		Returns:
		    list: List with the nodes for the face (size of 2)
		"""

		if face_i == 0:
			return [self.node_list[0], self.node_list[1]]

		elif face_i == 1:
			return [self.node_list[1], self.node_list[2]]

		elif face_i == 2:
			return [self.node_list[2], self.node_list[0]]

		else:
			raise ValueError("Unknown face index")


	def get_basis_funcs_given_dof(self, dof):
		
		"""Given the degree of freedom, return the basis function and 
		gradient of the basis function (defined on the reference domain).
		Note that the dof should be one of the nodes on the element, otherwise
		raise a ValueError.

		This function is used in the global system assembly.

		Args:
		    dof (obj): The node object of interest.
		
		Returns:
		    tuple: The basis function and gradient of the basis function
		    	defined on the reference element.
		"""

		for i in range(3):
			if self.node_list[i] == dof:
				return self.reference_element.psi_hat_basis[i], self.reference_element.grad_psi_hat_basis[i]

		raise ValueError("dof not in the element")





