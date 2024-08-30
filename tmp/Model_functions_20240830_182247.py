
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2):
	l=3
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2):
	l=3
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_40065182 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	
	jac[:,0]= exp_40065182
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_40065182
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_40065182
	
	
	return jac