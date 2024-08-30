
# Function definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14,arg_15,arg_16,arg_17,arg_18,arg_19,arg_20):
	l=21
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	sum+=arg_3*np.exp(-(x-(arg_1+100))**2/(2*arg_4**2))
	sum+=arg_5*np.exp(-(x-(arg_1+200))**2/(2*arg_6**2))
	sum+=arg_7*np.exp(-(x-(arg_1+300))**2/(2*arg_8**2))
	mask_array_42293016 = (x > 900) & (x < 1000);sub_x_42293016 = x[mask_array_42293016];sum[mask_array_42293016] += (arg_9 + sub_x_42293016 * (arg_10 + sub_x_42293016 * arg_11))
	mask_array_95068710 = (x > 1000) & (x < 1100);sub_x_95068710 = x[mask_array_95068710];sum[mask_array_95068710] += (arg_12 + sub_x_95068710 * (arg_13 + sub_x_95068710 * arg_14))
	mask_array_84106433 = (x > 1100) & (x < 1200);sub_x_84106433 = x[mask_array_84106433];sum[mask_array_84106433] += (arg_15 + sub_x_84106433 * (arg_16 + sub_x_84106433 * arg_17))
	mask_array_26219535 = (x > 1200) & (x < 1300);sub_x_26219535 = x[mask_array_26219535];sum[mask_array_26219535] += (arg_18 + sub_x_26219535 * (arg_19 + sub_x_26219535 * arg_20))
	return sum





# Jacobian definition
import numpy as np
from numba import jit
@jit(nopython=True, inline = 'always', cache = True)
def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14,arg_15,arg_16,arg_17,arg_18,arg_19,arg_20):
	l=21
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_50325124 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_25199698 = np.exp(-(x-(arg_1+100))**2/(2*arg_4**2))
	exp_82954374 = np.exp(-(x-(arg_1+200))**2/(2*arg_6**2))
	exp_29497666 = np.exp(-(x-(arg_1+300))**2/(2*arg_8**2))
	mask_array_27086638 = (x > 900) & (x < 1000);sub_x_27086638 = x[mask_array_27086638]
	mask_array_63323980 = (x > 1000) & (x < 1100);sub_x_63323980 = x[mask_array_63323980]
	mask_array_14041263 = (x > 1100) & (x < 1200);sub_x_14041263 = x[mask_array_14041263]
	mask_array_74079398 = (x > 1200) & (x < 1300);sub_x_74079398 = x[mask_array_74079398]
	
	jac[:,0]= exp_50325124
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_50325124 + arg_3*(x-(arg_1+100))   /(arg_4**2) * exp_25199698 + arg_5*(x-(arg_1+200))   /(arg_6**2) * exp_82954374 + arg_7*(x-(arg_1+300))   /(arg_8**2) * exp_29497666
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_50325124
	jac[:,3]= exp_25199698
	jac[:,4]= arg_3*(x-(arg_1+100))**2/(arg_4**3) * exp_25199698
	jac[:,5]= exp_82954374
	jac[:,6]= arg_5*(x-(arg_1+200))**2/(arg_6**3) * exp_82954374
	jac[:,7]= exp_29497666
	jac[:,8]= arg_7*(x-(arg_1+300))**2/(arg_8**3) * exp_29497666
	jac[mask_array_27086638,9]= 1
	jac[mask_array_27086638,10]= sub_x_27086638
	jac[mask_array_27086638,11]= (sub_x_27086638*sub_x_27086638)
	jac[mask_array_63323980,12]= 1
	jac[mask_array_63323980,13]= sub_x_63323980
	jac[mask_array_63323980,14]= (sub_x_63323980*sub_x_63323980)
	jac[mask_array_14041263,15]= 1
	jac[mask_array_14041263,16]= sub_x_14041263
	jac[mask_array_14041263,17]= (sub_x_14041263*sub_x_14041263)
	jac[mask_array_74079398,18]= 1
	jac[mask_array_74079398,19]= sub_x_74079398
	jac[mask_array_74079398,20]= (sub_x_74079398*sub_x_74079398)
	
	
	return jac