"""
Module: vertex.py

Contains the vertex information for the mesh. The vertices are used
to be able to find what the connectivity of the elements are 
for the unstructured mesh
"""


class Vertex(object):


	"""Class holding the data for the vertices in the mesh. 
	These vertices are used to be able to generate the 
	connectivity of the elements in the mesh.
	"""
	
	def __init__(self, x, y):
		
		"""Constructor for the Vertex class object
		
		Args:
		    x_loc (float): x location of the vertex point
		    y_loc (float): y location of the vertex point
		    index (int): The index of the vertex (global ordering). Obsolete
		    	parameter for the vertices now
		"""
		
		self.x, self.y = x, y
	
	def __str__(self):
		
		"""Summary
		
		Returns:
		    TYPE: Description
		"""

		return "[(x,y) = (%f, %f)]" % (self.x, self.y)