
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4):
	l=5
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	sum+=arg_3*np.exp(-(x-(arg_1+100))**2/(2*arg_4**2))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4):
	l=5
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_36052229 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_39824335 = np.exp(-(x-(arg_1+100))**2/(2*arg_4**2))
	
	jac[:,0]= exp_36052229
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_36052229 + arg_3*(x-(arg_1+100))   /(arg_4**2) * exp_39824335
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_36052229
	jac[:,3]= exp_39824335
	jac[:,4]= arg_3*(x-(arg_1+100))**2/(arg_4**3) * exp_39824335
	
	
	return jac