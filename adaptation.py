"""
Module: adaptation.py

Adapts the mesh using the computed error indicators
"""


def h_adapt_mesh(mesh_obj, error_indicator_fraction):
	
	"""h-adapt the mesh using the error indicator on
	each element. Compute the maximum indicator value
	and then refine all elements with a value of

		eta_i > error_indicator_fraction * eta_i_max
	

	Args:
	    mesh_obj (obj): The mesh object
	    error_indicator_fraction (float): The fraction of the maximum
	    	error indicator value to determine which elements to mark for
	    	adaptation
	"""

	# Get eta_max
	eta_max = 0.0
	for element in mesh_obj.element_list:
		eta_max = max(element.error_indicator, eta_max)

	# Mark elements to refine
	refine_element_list = []
	for element in mesh_obj.element_list:
	    if element.error_indicator >= error_indicator_fraction*eta_max:
	        refine_element_list.append(element)
	        
	# Refine elements
	for i in range(len(refine_element_list)):
	    
	    # First, check that the given element is still in the mesh (if it 
	    # isn't, then it has already been refined)
	    if refine_element_list[i] in mesh_obj.element_list:
	        try:
	            mesh_obj.h_refine_mesh([refine_element_list[i]])
	        except RuntimeError :
	        	# Temporary addition until a fix for the maximum recursion depth is placed
	            "Maximum Recursion Depth Reached for Element : %s" % refine_element_list[i]


