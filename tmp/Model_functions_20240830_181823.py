
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_58002923 = (x > 700) & (x < 1300);sub_x_58002923 = x[mask_array_58002923];sum[mask_array_58002923] += (arg_3 + sub_x_58002923 * (arg_4 + sub_x_58002923 * arg_5))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_56913571 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_40884356 = (x > 700) & (x < 1300);sub_x_40884356 = x[mask_array_40884356]
	
	jac[:,0]= exp_56913571
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_56913571
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_56913571
	jac[mask_array_40884356,3]= 1
	jac[mask_array_40884356,4]= sub_x_40884356
	jac[mask_array_40884356,5]= (sub_x_40884356*sub_x_40884356)
	
	
	return jac