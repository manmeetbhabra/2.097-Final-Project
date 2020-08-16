"""
Module: node.py

Contains the node information for the mesh. The nodes are all instances of 
a class. This structure is used as different elements will share nodes 
(if they are interior to the domain) and so an efficient way will be 
needed to get the contributions from each element.
"""


class Node(object):

	"""Class holding the data for the nodes in the mesh. 
	
	The node types include standard degrees of freedom, hanging nodes
	and boundary nodes.
	
	Attributes:
	    element_list (list): Description
	    node_type (TYPE): Description
	    value (TYPE): Description
	"""
	
	def __init__(self, node_type, x, y):
		
		"""Constructor for the Node class object
		
		Args:
		    node_type (str): Type of node. Options include 
		    	- "dof" : (Degree of freedom) The value may be adjusted
		    	- "hanging" : The node is constrained (it is a hanging node
		    		generated through h refinement)
		    	- "dirichlet_bc" : A node on the dirichlet boundary (so the value
		    		is fixed as prescribed).
		    x: The x value for the node
		    y: The y value for the node
		"""
	
		self.x, self.y = x, y
		self.node_type = node_type

		# Initialize the value stored by this degree of freedom to None 
		self.value = None

		# Initialize the error at this node to None. This error is used 
		# when solving the auxiliary system
		self.error_value = None

		# List holding what elements contain this node
		self.element_list = []

		# The list of nodes that will be use to constrain the value
		# of a given hanging node (used only if the node is a hanging node)
		self.hanging_node_constraint_nodes = None


	def __str__(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		return "[(x,y) = (%f, %f), type: %s]" % (self.x, self.y, self.node_type)


	def set_value(self, value):
		
		"""Set the value held by this node
		
		Args:
		    value (float): The state at this given node
		"""

		self.value = value


	def set_error_value(self, error_val):

		self.error_value = error_val


	def set_type(self, node_type):
		"""Summary
		
		Args:
		    node_type (TYPE): Description
		"""
		self.node_type = node_type


	def set_hanging_nodes_constraint_nodes(self, constraint_nodes):
		
		"""Set the constraint nodes if this node is a hanging node. Note that
		we are only working with P=1 elements so only two constraint nodes
		exist.

		NOTE: Since only P=1 elements are being worked with, the ordering
			of the constraint nodes doesn't matter (we just average
			the values).
		
		Args:
		    constraint_nodes (list): List of nodes used to constrain the
		    	value of this hanging node
		"""
		self.hanging_node_constraint_nodes = constraint_nodes


	def clear_element_list(self):
		
		"""Clear the list of element that this dof points to.
		"""
		
		self.element_list = []


	def add_element(self, element):
		
		"""Add an element to the element_list
		
		Args:
		    element (TYPE): The finite element object
		"""
	
		self.element_list.append(element)


	def remove_element(self, element):
		
		"""Remove an element from the element list. If it is
		not in the list, raise a value error for now
		for debugging purposes
		
		Args:
		    element (obj): The finite element object to remove
		"""

		if element not in self.element_list:
			raise ValueError("Element to remove not in dof element list")

		self.element_list.remove(element)



