
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8):
	l=9
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	sum+=arg_3*np.exp(-(x-arg_4)**2/(2*arg_5**2))
	sum += (arg_6 + x * (arg_7 + x * arg_8))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8):
	l=9
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_95468627 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_9280687 = np.exp(-(x-arg_4)**2/(2*arg_5**2))
	
	jac[:,0]= exp_95468627
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_95468627
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_95468627
	jac[:,3]= exp_9280687
	jac[:,4]= arg_3*(x-arg_4)   /(arg_5**2) * exp_9280687
	jac[:,5]= arg_3*(x-arg_4)**2/(arg_5**3) * exp_9280687
	jac[:,6]= 1
	jac[:,7]= x
	jac[:,8]= (x*x)
	
	
	return jac