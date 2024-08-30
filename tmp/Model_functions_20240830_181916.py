
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_65516759 = (x > 700) & (x < 1300);sub_x_65516759 = x[mask_array_65516759];sum[mask_array_65516759] += (arg_3 + sub_x_65516759 * (arg_4 + sub_x_65516759 * arg_5))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_85172963 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_77609520 = (x > 700) & (x < 1300);sub_x_77609520 = x[mask_array_77609520]
	
	jac[:,0]= exp_85172963
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_85172963
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_85172963
	jac[mask_array_77609520,3]= 1
	jac[mask_array_77609520,4]= sub_x_77609520
	jac[mask_array_77609520,5]= (sub_x_77609520*sub_x_77609520)
	
	
	return jac