
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_43842887 = (x > 700) & (x < 1300);sub_x_43842887 = x[mask_array_43842887];sum[mask_array_43842887] += (arg_3 + sub_x_43842887 * (arg_4 + sub_x_43842887 * arg_5))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_44563662 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_54167915 = (x > 700) & (x < 1300);sub_x_54167915 = x[mask_array_54167915]
	
	jac[:,0]= exp_44563662
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_44563662
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_44563662
	jac[mask_array_54167915,3]= 1
	jac[mask_array_54167915,4]= sub_x_54167915
	jac[mask_array_54167915,5]= (sub_x_54167915*sub_x_54167915)
	
	
	return jac