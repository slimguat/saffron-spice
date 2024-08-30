
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2):
	l=3
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	mask_array_33292806 = (x > 900) & (x < 1000);sub_x_33292806 = x[mask_array_33292806];sum[mask_array_33292806] += (arg_0 + sub_x_33292806 * (arg_1 + sub_x_33292806 * arg_2))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2):
	l=3
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	mask_array_22652443 = (x > 900) & (x < 1000);sub_x_22652443 = x[mask_array_22652443]
	
	jac[mask_array_22652443,0]= 1
	jac[mask_array_22652443,1]= sub_x_22652443
	jac[mask_array_22652443,2]= (sub_x_22652443*sub_x_22652443)
	
	
	return jac