
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_94231464 = (x > 700) & (x < 1300);sub_x_94231464 = x[mask_array_94231464];sum[mask_array_94231464] += (arg_3 + sub_x_94231464 * (arg_4 + sub_x_94231464 * arg_5))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5):
	l=6
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_55436320 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_31196886 = (x > 700) & (x < 1300);sub_x_31196886 = x[mask_array_31196886]
	
	jac[:,0]= exp_55436320
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_55436320
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_55436320
	jac[mask_array_31196886,3]= 1
	jac[mask_array_31196886,4]= sub_x_31196886
	jac[mask_array_31196886,5]= (sub_x_31196886*sub_x_31196886)
	
	
	return jac