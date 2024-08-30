
from typing import Callable
from types import FunctionType
from datetime import datetime
from pathlib import Path
import numpy as np
import copy
import sys
import importlib.util
from ..utils import ModelCodec

class ModelFactory():
  def __init__(self,functions = None,jit_activated = True,cache=True) -> None:
    """
      Initializes the Model object, setting up dictionaries to store functions 
      and configurations for generating function and Jacobian strings.
      

      A class for constructing, managing, and fitting mathematical models consisting of Gaussian and polynomial components.

      The `Model` class allows users to define a model comprising Gaussian and polynomial functions, manage the parameters,
      and generate code for the model and its Jacobian matrix, which can be used for fitting procedures. It also supports
      locking parameters between different components, ensuring that certain parameters maintain fixed relationships 
      during optimization.

      Attributes:
          functions (dict): A dictionary that stores the Gaussian and polynomial functions that make up the model.
          string_to_function_generator_list (dict): A mapping of model types ('gaussian', 'polynome') to their respective
              function generation methods.
          string_to_jacobian_generator_list (dict): A mapping of model types ('gaussian', 'polynome') to their respective
              Jacobian generation methods.
          module_imports (list): A list of required imports for the generated function and Jacobian strings.
          function_string (list): A list containing the name and code of the generated function.
          jacobian_string (list): A list containing the name and code of the generated Jacobian function.
          lockParameterIndexMapping (np.ndarray): A mapping of locked parameters to their corresponding indices in the
              parameter vector.

      Methods:
          __init__(): Initializes the Model class, setting up empty function containers and necessary attributes.
          __add__(other): Combines two models into a new model by merging their functions.
          __repr__(): Provides a string representation of the model, detailing the components and their parameters.
          add_func_dict(functions_dict): Adds functions to the model from a given dictionary.
          add_gaussian(I, x, s): Adds a Gaussian function to the model with specified parameters.
          add_polynome(*Bi, lims=[-np.inf, np.inf]): Adds a polynomial function to the model with specified coefficients
              and limits.
          get_locked_index(arr, target): Finds the index of a target object in a numpy array.
          lock(param_1, param_2, lock_protocol): Locks a parameter in one component of the model to another, enforcing
              a fixed relationship.
          gen_mappings(): Generates a mapping of parameters to their indices in the parameter vector.
          get_lock_prameter_vector(): Returns a vector of the current values of the locked parameters.
          gen_fit_function(): Generates the code for the model function and its Jacobian matrix, storing them as strings.
          dump_function(directory=Path('./tmp')): Saves the generated function and Jacobian code to a file.
          encode_model(): Encodes the model object into a string for serialization.
          decode_model(string): Decodes a string into a model object, reconstructing the model's functions.
          lock_to_unlock(locked_array: np.ndarray) -> np.ndarray: Converts an array of locked parameter values to their
              corresponding unlocked values.
          unlock_to_lock(unlocked_array: np.ndarray) -> np.ndarray: Converts an array of unlocked parameter values to their
              corresponding locked values.
          _lock_arg_string_generator(model_type, ind_func, key): Generates the code string for a locked parameter.
          _gaussian_string_to_function(ind_func): Generates the code string for a Gaussian function.
          _polynome_string_to_function(ind_func): Generates the code string for a polynomial function using Horner's method.
          _lock_arg_string_jac_generator(model_type, ind_func, key): Generates the Jacobian code for a locked parameter.
          _gaussian_string_to_jacobian(ind_func): Generates the Jacobian code string for a Gaussian function.
          _polynome_string_to_jacobian(ind_func): Generates the Jacobian code string for a polynomial function, optimizing
              repeated multiplications.
    """
    self.string_to_function_generator_list = {'gaussian':self._gaussian_string_to_function,'polynome':self._polynome_string_to_function}
    self.string_to_jacobian_generator_list = {'gaussian':self._gaussian_string_to_jacobian,'polynome':self._polynome_string_to_jacobian}
    self.module_imports = ['import numpy as np','from numba import jit',]
    self.function_string = None
    self.jacobian_string = None
    self.func_path = None
    
    
    self._function_string_template = "{0}\n"+(("@jit(nopython=True, inline = 'always', cache = "+f"{str(cache)}"+")" ) if jit_activated else "" )+"\ndef {1}(x,{2}):\n\tl={3}\n\tif isinstance(x,float):x = np.array([x],dtype=np.float64)\n\tsum = np.zeros((len(x), ),dtype=np.float64)"
    self._jacobian_string_template = "{0}\n"+(("@jit(nopython=True, inline = 'always', cache = "+f"{str(cache)}"+")" ) if jit_activated else "" )+"\ndef {1}(x,{2}):\n\tl={3}\n\tif isinstance(x,float):x = np.array([x],dtype=np.float64)\n\tjac = np.zeros((len(x),l),dtype=np.float64)"
    
    self.bounds = None #set in self.set_bounds
    self.lockParameterIndexMapping = None #set by gen_mapping
    self.unlockParameterIndexMapping = None #set by gen_mapping
    
    #here lies all the parameters as well as their values 
    if functions is not None:
      self.functions = functions
      self.gen_mappings()
      self.gen_fit_function()
      self.set_bounds()
      
    else:
      self.functions = {"gaussian":{},"polynome":{}}
  def __add__(self, other: 'Model') -> 'Model':
    """
    Combines two Model objects by adding their functions together.
    
    Args:
        other (Model): Another Model object to add to the current one.
        
    Returns:
        Model: A new Model object containing the combined functions of both models.
    
    Raises:
        ValueError: If the other object is not a Model.
    """
    if not isinstance(other,Model):raise ValueError("Can only add Model to Model")
    a_dict = copy.deepcopy(self.functions)
    b_dict = copy.deepcopy(other.functions)
    _res = a_dict.copy()
    a_lengths = {key: len(a_dict[key]) for key in a_dict}
    
    for key in b_dict:
      if key not in _res:
        _res[key] = {}
        a_lengths[key] = 0
        
      for key2 in b_dict[key]:
        func_prams = {}
        for key3 in b_dict[key][key2]:
          if not isinstance(b_dict[key][key2][key3],dict):
            func_prams[key3] = b_dict[key][key2][key3]
          else:
            func_prams[key3] = copy.deepcopy(b_dict[key][key2][key3])
            model_type = b_dict[key][key2][key3]['reference']['model_type']
            # print("-------------------")
            # print(func_prams[key3]['reference']['element_index'],"+")
            # print(a_lengths[model_type])
            # print("-------------------")
            func_prams[key3]['reference']['element_index'] += a_lengths[model_type]
        _res[key][key2+a_lengths[key]] = func_prams
    
    
    return Model(_res)
  def __repr__(self) -> str:
    """
    Provides a string representation of the Model, showing its functions and parameters.
    
    Returns:
        str: A formatted string that represents the Model's functions and their parameters.
    """
    output = ''
    for key in self.functions:
      for key2 in self.functions[key]:
        _repr = {key3:self.functions[key][key2][key3] for key3 in self.functions[key][key2]}
        for key3 in _repr:
          if isinstance(_repr[key3],dict):
            if _repr[key3]["constraint"] == 'lock':
              new_repr = f'locked to {_repr[key3]['reference']['model_type'].upper()}[{_repr[key3]["reference"]["element_index"]},{_repr[key3]["reference"]["parameter"]}]{"+" if _repr[key3]["operation"] == "add" else "*"}{_repr[key3]["value"]} = {
              _repr[key3]["value"]+self.functions[_repr[key3]['reference']['model_type']][_repr[key3]["reference"]["element_index"]][_repr[key3]["reference"]["parameter"]]
              if _repr[key3]['operation'] == 'add'
              else 
              _repr[key3]["value"]*self.functions[_repr[key3]['reference']['model_type']][_repr[key3]["reference"]["element_index"]][_repr[key3]["reference"]["parameter"]]
              }'
              _repr[key3] = new_repr
        output += '\n-----------------------'
        output += (f'\n{key.upper()}[{key2}]:\n   '+'\n   '.join([f"{key3}={_repr[key3]}" for key3 in _repr]))
    output += '\n-----------------------'
    return output
  def add_gaussian(self, I: float, x: float, s: float) -> None:
    """
    Adds a Gaussian function to the Model.
    
    Args:
        I (float): Amplitude of the Gaussian.
        x (float): Mean of the Gaussian.
        s (float): Standard deviation of the Gaussian.
    """
    index = len(self.functions['gaussian'])
    self.functions['gaussian'][index] = {'I':I,'x':x,'s':s}
    self.gen_mappings()
    self.gen_fit_function()
    self.set_bounds()
  def add_polynome(self, *Bi: float, lims: list = [-np.inf, np.inf]) -> None:
    """
    Adds a polynomial function to the Model.
    
    Args:
        *Bi (float): Coefficients of the polynomial.
        lims (list, optional): Limits within which the polynomial is valid. Defaults to [-np.inf, np.inf].
    """
    index = len(self.functions['polynome'])
    self.functions['polynome'][index] = {**{f'B{i}':Bi[i] for i in range(len(Bi))},'lims':lims}
    self.gen_mappings()
    self.gen_fit_function()
    self.set_bounds()
  def get_locked_index(self, arr: np.ndarray, target: list) -> int:
    """
    Finds the index of a target object in a numpy array.
    
    Args:
        arr (np.ndarray): The array to search.
        target (list): The object to find in the array.
        
    Returns:
        int: The index of the target in the array.
        
    Raises:
        ValueError: If the target object is not found in the array.
    """
    # Loop through the array to find the index manually
    for i, row in enumerate(arr):
        if list(row) == list(target):
            return i
    raise ValueError(f"Target {target} not found in array")
  def lock(self, param_1: dict, param_2: dict, lock_protocol: dict) -> None:
    """
    Locks a parameter in one function to another parameter in a different function.
    
    Args:
        param_1 (dict): Dictionary specifying the first parameter to lock.
        param_2 (dict): Dictionary specifying the second parameter to lock.
        lock_protocol (dict): Dictionary specifying the operation and value for the lock.
    
    Raises:
        ValueError: If the locking involves polynomials or if the second parameter is already locked.
    """
    assert "model_type" in param_1.keys() and "element_index" in param_1.keys() and "parameter" in param_1.keys() 
    assert "model_type" in param_2.keys() and "element_index" in param_2.keys() and "parameter" in param_2.keys() 
    assert "operation" in lock_protocol.keys() and "value" in lock_protocol.keys()
    
    #locking anything to polynoms is forbidden for now so I have to prohibit it
    if param_1['model_type'] == 'polynome' or param_2['model_type'] == 'polynome':
      raise ValueError("Locking into polynoms is forbidden for now for mask conflict reasons") 
    
    index1 = param_1['element_index']
    index2 = param_2['element_index']
    model_type1 = param_1['model_type']
    model_type2 = param_2['model_type']
    p1 = param_1['parameter']
    p2 = param_2['parameter']
    model1 = self.functions[model_type1][index1]
    model2 = self.functions[model_type2][index2]
    operation = lock_protocol['operation']
    value = lock_protocol['value']
  
    #make sure that p2 value is set and not locked
    if model2[p2] == 'lock':raise ValueError(f"Parameter {p2} is locked")
    self.functions[model_type1][index1][p1] = {
      "constraint":"lock",
      'operation':operation,
      'value':value,
      'reference':{'model_type':model_type2,'element_index':index2,'parameter':p2}
      }
    self.gen_mappings()
    self.gen_fit_function()
  def gen_mappings(self) -> None:
    """
    Generates mappings of parameters for the Model, storing them for use in function and Jacobian generation.
    """
    self.lockParameterIndexMapping = []
    for model_type in self.functions:
      for element_index in range(len(self.functions[model_type])):
        for parameter in self.functions[model_type][element_index]:
          if (isinstance(self.functions[model_type][element_index][parameter],dict) or parameter =="lims"):
            continue
          self.lockParameterIndexMapping.append([model_type,element_index,parameter])
    self.lockParameterIndexMapping = np.array(self.lockParameterIndexMapping,dtype= object)

    self.unlockParameterIndexMapping = []
    for model_type in self.functions:
      for element_index in range(len(self.functions[model_type])):
        for parameter in self.functions[model_type][element_index]:
          if ( parameter =="lims"):
            continue
          self.unlockParameterIndexMapping.append([model_type,element_index,parameter])
    self.unlockParameterIndexMapping = np.array(self.unlockParameterIndexMapping,dtype= object)
  def get_lock_parameter_vector(self,unlocked_params=None) -> np.ndarray:
    """
    Retrieves a vector of parameters that are locked in the Model.
    
    Returns:
        np.ndarray: An array of locked parameters in the Model.
    """
    if unlocked_params is None:
      locked_params =  np.array([
        self.functions[i[0]][i[1]][i[2]] for i in self.lockParameterIndexMapping
      ])
    else:
      if len(unlocked_params) != len(self.unlockParameterIndexMapping):raise ValueError("Unlocked parameters must be equal to the number of unlocked parameters")
      
      locked_params = np.zeros(len(self.lockParameterIndexMapping))
      for index_lock , lock in enumerate(self.lockParameterIndexMapping):
        
        index_unlock = self.get_locked_index(self.unlockParameterIndexMapping,lock)
        locked_params[index_lock] = unlocked_params[index_unlock]
    return locked_params
  def get_unlock_parameter_vector(self,locked_params=None) -> np.ndarray:
    """
    Generates a vector of unlocked parameters in the Model.
    
    Returns:
        np.ndarray: An array of unlocked parameters in the Model.
    """
    if locked_params is None:
      unlocked_params = np.zeros(len(self.unlockParameterIndexMapping))
      for index_unlock,lock in enumerate(self.unlockParameterIndexMapping):
        if not isinstance(self.functions[lock[0]][lock[1]][lock[2]],dict):
          unlocked_params[index_unlock] = self.functions[lock[0]][lock[1]][lock[2]] 
        else: 
          constraint = self.functions[lock[0]][lock[1]][lock[2]]['constraint']
          if constraint == 'lock':
            reference = self.functions[lock[0]][lock[1]][lock[2]]['reference']
            operation = self.functions[lock[0]][lock[1]][lock[2]]['operation']
            value = self.functions[lock[0]][lock[1]][lock[2]]['value']
            if operation == 'add':
              unlocked_params[index_unlock] = value + self.functions[reference['model_type']][reference['element_index']][reference['parameter']]
            else:
              unlocked_params[index_unlock] = value * self.functions[reference['model_type']][reference['element_index']][reference['parameter']] 
          else:
            raise ValueError(f"Constraint {constraint} not supported")
            
        
    else:
      unlocked_params = np.zeros(len(self.unlockParameterIndexMapping))
      for index_unlock,lock in enumerate(self.unlockParameterIndexMapping):
        if not isinstance(self.functions[lock[0]][lock[1]][lock[2]],dict):
          index_lock = self.get_locked_index(self.lockParameterIndexMapping,lock)
          unlocked_params[index_unlock] = locked_params[index_lock]
        else: 
          constraint = self.functions[lock[0]][lock[1]][lock[2]]['constraint']
          if constraint == 'lock':
            reference = self.functions[lock[0]][lock[1]][lock[2]]['reference']
            operation = self.functions[lock[0]][lock[1]][lock[2]]['operation']
            value = self.functions[lock[0]][lock[1]][lock[2]]['value']
            if operation == 'add':
              index_lock_reference = self.get_locked_index(self.lockParameterIndexMapping,reference.values())
              unlocked_params[index_unlock] = value + locked_params[index_lock_reference]
            else:
              unlocked_params[index_unlock] = value * locked_params[index_lock_reference]
          else:
            raise ValueError(f"Constraint {constraint} not supported")
    return unlocked_params
  def gen_fit_function(self) -> None:
    """
    Generates the function and Jacobian strings for the Model, including the handling of locked parameters.
    """
    if True: #Generate the function string
      #function name is function_<dateYYYYMMDD>_<timeHHMMSS>
      now_date = datetime.now()
      _name  =  'model_function'#_{}_{}'.format(now_date.strftime("%Y%m%d"),now_date.strftime("%H%M%S"))
      self.function_string = [_name ,None]
      self.function_string[1] = self._function_string_template.format(
        '\n'.join(self.module_imports),
        _name,
        ','.join([f'arg_{i}' for i in range(len(self.lockParameterIndexMapping))]),
        len(self.lockParameterIndexMapping),
        )
      
      for model_type in self.functions:
        for ind_func in (self.functions[model_type]):
          
          self.function_string[1] += "\n\t"+self.string_to_function_generator_list[model_type](ind_func) 
      self.function_string[1] += '\n\treturn sum'
    
    if True:  #Generate the jacobian string
      _name2 =  'model_jacobian'#_{}_{}'.format(now_date.strftime("%Y%m%d"),now_date.strftime("%H%M%S"))
      self.jacobian_string = [_name2,None]
      self.jacobian_string[1] = self._jacobian_string_template.format(
        '\n'.join(self.module_imports),
        _name2,
        ','.join([f'arg_{i}' for i in range(len(self.lockParameterIndexMapping))]),
        len(self.lockParameterIndexMapping),
        )
      
      #setting each line with its jacobian expression 
      lines = ["jac[{}"+f",{i}]=" for i in range(len(self.lockParameterIndexMapping))]
      mask_declarations = [] 
      for model_type in self.functions:
        for ind_func in (self.functions[model_type]):
          params_jac = self.string_to_jacobian_generator_list[model_type](ind_func)
          """
          params_jac = [
            [p_expression1,index1,if lims is defined, mask_array_name],
            in case lims is defined [lim_expression,index=-1] this part has to be in the beginning of the function declaration so it will declare the mask_array and sub_x 
            ...
            ...
          """
          for pram_jac in params_jac:
            #case index is not -1 (simple jacobian expression for derivative of element i)
            if pram_jac[1] != -1:
              lines[pram_jac[1]] += " + "+pram_jac[0]
              if len(pram_jac)>2: #in case lims is defined
                lines[pram_jac[1]] = lines[pram_jac[1]].format(pram_jac[2])
              else:
                lines[pram_jac[1]] = lines[pram_jac[1]].format(':')
            #case index is -1 (mask array declaration)
            else:
              mask_declarations.append(pram_jac[0])
            
      self.jacobian_string[1] +=  '\n\t'+'\n\t'.join(mask_declarations) + '\n\t'+'\n\t' + '\n\t'.join(lines) + '\n\t'+'\n\t' + '\n\treturn jac'
      self.jacobian_string[1] = self.jacobian_string[1].replace('= +', '=')
  def dump_function(self, directory: Path = Path('./tmp')) -> None:
    """
    Saves the generated function and Jacobian strings to a file in the specified directory.
    
    Args:
        directory (Path, optional): The directory to save the function file in. Defaults to './tmp'.
    """
    directory.mkdir(exist_ok=True)
    now_date = datetime.now()
    self.func_path = directory/f'Model_functions_{now_date.strftime("%Y%m%d")}_{now_date.strftime("%H%M%S")}.py' 
    with open(self.func_path,'w') as f:
      f.write('\n# Function definition\n'+self.function_string[1] + '\n'*5+'\n# Jacobian definition\n' + self.jacobian_string[1]) 
  def _lock_arg_string_generator(self, model_type: str, ind_func: int, key: str) -> str:
    """
    Generates a string representing a locked argument for use in function or Jacobian generation.
    
    Args:
        model_type (str): The type of model (e.g., 'gaussian', 'polynome').
        ind_func (int): The index of the function within its type.
        key (str): The parameter key within the function.
        
    Returns:
        str: The generated argument string.
    
    Raises:
        ValueError: If an unsupported operation is encountered.
    """
    if self.functions[model_type][ind_func][key]['operation'] == 'add':
      operation = '+'
    elif self.functions[model_type][ind_func][key]['operation'] == 'mul':
      operation = '*'
    else:
      raise ValueError(f"Operation {self.functions[model_type][ind_func][key]['operation']} not implemented")
    
    reference = self.functions[model_type][ind_func][key]['reference']
    value = self.functions[model_type][ind_func][key]['value']
    ref_param_string = f"arg_{(
      self.get_locked_index(
        self.lockParameterIndexMapping,list(reference.values())
        )
      )}"
      
    return '('+ref_param_string + operation + f'{value})'
  def _gaussian_string_to_function(self, ind_func: int) -> str:
    """
    Generates a string representing a Gaussian function for use in function generation.
    
    Args:
        ind_func (int): The index of the Gaussian function.
        
    Returns:
        str: The generated Gaussian function string.
    
    Raises:
        ValueError: If an unsupported constraint is encountered.
    """
    gaussian_code_string_template = "{0}*np.exp(-(x-{1})**2/(2*{2}**2))"
    # string_to_concatenate += gaussian_code_string_template.format()
    str_params = []
    for key,item in self.functions["gaussian"][ind_func].items():
      if not isinstance(item,dict):
        str_params.append(f'arg_{self.get_locked_index(self.lockParameterIndexMapping,["gaussian",ind_func,key])}')
      
      elif self.functions["gaussian"][ind_func][key]["constraint"] == 'lock':
        str_params.append(self._lock_arg_string_generator("gaussian",ind_func,key))
      else: 
        raise ValueError(f"Constraint {self.functions['gaussian'][ind_func][key]['constraint']} not implemented")
    return "sum+="+gaussian_code_string_template.format(*str_params)
  def _polynome_string_to_function(self, ind_func: int) -> str:
    """
    Generates a string representing a polynomial function using Horner's method for use in function generation.
    
    Args:
        ind_func (int): The index of the polynomial function.
        
    Returns:
        str: The generated polynomial function string.
    
    Raises:
        ValueError: If an unsupported constraint is encountered.
    """
    if True:
      lims = self.functions["polynome"][ind_func]['lims']
      
      if not (lims[0] == -np.inf and lims[1] == np.inf):
          rand = np.random.randint(0, 10**8)
          mask_array_name = f"mask_array_{rand}"
          mask_array = "mask_array_{2} = (x > {0}) & (x < {1})".format(*lims, rand)
          sub_x = f"sub_x_{rand} = x[{mask_array_name}]"
          x_var = f"sub_x_{rand}"
      else:
          mask_array = None
          sub_x = None
          x_var = "x"

      # Start with the last coefficient (highest power)
      str_params = []
      for key, item in sorted(self.functions["polynome"][ind_func].items(), reverse=True):
          if key != 'lims':
              if not isinstance(item, dict):
                  str_params.append(f'arg_{self.get_locked_index(self.lockParameterIndexMapping, ["polynome", ind_func, key])}')
              elif self.functions["polynome"][ind_func][key]["constraint"] == 'lock':
                  str_params.append(self._lock_arg_string_generator("polynome", ind_func, key))
              else:
                  raise ValueError(f"Constraint {self.functions['polynome'][ind_func][key]['constraint']} not implemented")
      
      # Generate the Horner's method polynomial evaluation string
      horner_expression = str_params[0]  # Start with the coefficient of the highest power
      for param in str_params[1:]:
          horner_expression = f"({param} + {x_var} * {horner_expression})"
      
      if mask_array is not None:
          horner_expression = f"{mask_array};{sub_x};sum[{mask_array_name}] += {horner_expression}"
      else:
          horner_expression = f"sum += {horner_expression}"
      return horner_expression
  def _lock_arg_string_jac_generator(self, model_type: str, ind_func: int, key: str) -> list:
    """
    Generates the argument string for a locked parameter for use in Jacobian generation.
    
    Args:
        model_type (str): The type of model (e.g., 'gaussian', 'polynome').
        ind_func (int): The index of the function within its type.
        key (str): The parameter key within the function.
        
    Returns:
        list: A list containing the value and index of the locked parameter.
    """
    operation = self.functions[model_type][ind_func][key]['operation']
    reference = self.functions[model_type][ind_func][key]['reference']
    value     = self.functions[model_type][ind_func][key]['value']
    _ind = self.get_locked_index(
      self.lockParameterIndexMapping,
      [
        reference["model_type"],
        reference["element_index"],
        reference["parameter"],
        ])
    _val = f'{value}' if (value != 1 and operation != "add") else ''
    return [_val,_ind]
  def _gaussian_string_to_jacobian(self, ind_func: int) -> list:
    """
    Generates a list of expressions and indices representing the Jacobian for a Gaussian function.
    
    Args:
        ind_func (int): The index of the Gaussian function.
        
    Returns:
        list: A list of Jacobian expressions and their corresponding indices.
    """
    # gaussian derivative sample  
    # dA = dI                                  * np.exp(-((x - x0) ** 2) / (2 * s ** 2))
    # dx0 =dx0* I * (x - x0)       / (s ** 2)  * np.exp(-((x - x0) ** 2) / (2 * s ** 2))
    # ds = ds * I * (x - x0) ** 2) / (s ** 3)  * np.exp(-((x - x0) ** 2) / (2 * s ** 2))
    _rand = np.random.randint(0,10**8)
    exponent_template =  f'exp_{_rand} = '+'np.exp(-(x-{1})**2/(2*{2}**2))'
    
    I_jacobian_template = "{3}"+f"exp_{_rand}"
    x_jacobian_template = "{4}{0}*(x-{1})   /({2}**2) * "+f"exp_{_rand}"
    s_jacobian_template = "{5}{0}*(x-{1})**2/({2}**3) * "+f"exp_{_rand}"
    
    # get the values of the parameters
    I = self.functions["gaussian"][ind_func]['I']
    x = self.functions["gaussian"][ind_func]['x']
    s = self.functions["gaussian"][ind_func]['s']
    
    # get string of the parameters (jacobian template 0,1,2)
    if not isinstance(I,dict):Ival = f'arg_{self.get_locked_index(self.lockParameterIndexMapping,["gaussian",ind_func,"I"])}'
    else:Ival = self._lock_arg_string_generator("gaussian",ind_func,"I")
    if not isinstance(x,dict):xval = f'arg_{self.get_locked_index(self.lockParameterIndexMapping,["gaussian",ind_func,"x"])}'
    else:xval = self._lock_arg_string_generator("gaussian",ind_func,"x")
    if not isinstance(s,dict):sval = f'arg_{self.get_locked_index(self.lockParameterIndexMapping,["gaussian",ind_func,"s"])}'
    else:sval = self._lock_arg_string_generator("gaussian",ind_func,"s")
    
    # add derivation of lock values, and its typical index in the jacobian matrix
    if not isinstance(I,dict):
      dIval = ""
      dIval_ind = self.get_locked_index(self.lockParameterIndexMapping,["gaussian",ind_func,"I"])
    else: 
      _ = self._lock_arg_string_jac_generator("gaussian",ind_func,"I")
      dIval = _[0] + ("*" if _[0]!="" else "")
      dIval_ind = _[1]

    if not isinstance(x,dict):
      dxval = ""
      dxval_ind = self.get_locked_index(self.lockParameterIndexMapping,["gaussian",ind_func,"x"])
    else: 
      _ = self._lock_arg_string_jac_generator("gaussian",ind_func,"x")
      dxval = _[0] + ("*" if _[0]!="" else "")
      dxval_ind = _[1]

    if not isinstance(s,dict):
      dsval = ""
      dsval_ind = self.get_locked_index(self.lockParameterIndexMapping,["gaussian",ind_func,"s"])
    else: 
      _ = self._lock_arg_string_jac_generator("gaussian",ind_func,"s")
      dsval = _[0] + ("*" if _[0]!="" else "")
      dsval_ind = _[1]

    I_string_expression = I_jacobian_template.format(Ival,xval,sval,dIval,dxval,dsval)
    x_string_expression = x_jacobian_template.format(Ival,xval,sval,dIval,dxval,dsval)
    s_string_expression = s_jacobian_template.format(Ival,xval,sval,dIval,dxval,dsval)
    
    list_expressions_and_indices = [
      [I_string_expression            ,dIval_ind],
      [x_string_expression            ,dxval_ind],
      [s_string_expression            ,dsval_ind], 
      [exponent_template.format(Ival,xval,sval,dIval,dxval,dsval),-1       ], 
    ]
    return list_expressions_and_indices
  def _polynome_string_to_jacobian(self, ind_func: int) -> list:
    """
    Generates a list of expressions and indices representing the Jacobian for a polynomial function.
    
    Args:
        ind_func (int): The index of the polynomial function.
        
    Returns:
        list: A list of Jacobian expressions and their corresponding indices.
    """
    if True:
      # polynome derivative sample
      # dpi = dBi * i * x**(i-1)
      list_expressions_and_indices = []

      lims = self.functions["polynome"][ind_func]["lims"]

      if not (lims[0] == -np.inf and lims[1] == np.inf):
          rand = np.random.randint(0, 10**8)
          mask_array_name = f"mask_array_{rand}"
          mask_array = f"{mask_array_name} = (x > {lims[0]}) & (x < {lims[1]})"
          sub_x = f"sub_x_{rand} = x[{mask_array_name}]"
          x_var = f"sub_x_{rand}"
      else:
          mask_array = None
          sub_x = None
          x_var = "x"

      # Get the values of the parameters
      for key in self.functions["polynome"][ind_func]:
          if key != 'lims':
              B = self.functions["polynome"][ind_func][key]
              if not isinstance(B, dict):
                  Bval = f'arg_{self.get_locked_index(self.lockParameterIndexMapping, ["polynome", ind_func, key])}'
              else:
                  Bval = self._lock_arg_string_generator("polynome", ind_func, key)

              # Determine the index for the Jacobian matrix
              if not isinstance(B, dict):
                  dBval = ""
                  dBval_ind = self.get_locked_index(self.lockParameterIndexMapping, ["polynome", ind_func, key])
              else:
                  _ = self._lock_arg_string_jac_generator("polynome", ind_func, key)
                  dBval = _[0] + ("*" if _[0] != "" else "")
                  dBval_ind = _[1]

              # Optimize the Jacobian expression using repeated multiplication for exponents
              exponent = int(key[1:])  # Extract the exponent from the key (e.g., 'B1' -> 1)
              if exponent == 0:
                  B_string_expression = f"{dBval}" if dBval != "" else "1"
              elif exponent == 1:
                  B_string_expression = f"{dBval}{x_var}"
              else:
                  repeated_mul = f"({'*'.join([x_var] * exponent)})"
                  B_string_expression = f"{dBval}{repeated_mul}"

              list_expressions_and_indices.append([B_string_expression, dBval_ind])

              if mask_array is not None:
                  list_expressions_and_indices[-1].append(mask_array_name)

      if mask_array is not None:
          list_expressions_and_indices.append([f"{mask_array};{sub_x}", -1])

      return list_expressions_and_indices
  def encode_model(self,version = "latest") -> str:
    """
    Encodes the Model object into a Base64 encoded string using ModelSerializer.
    
    Returns:
        str: The encoded model string.
    """
    _ = ModelCodec(version='latest')
    
    return _.serialize(self.functions)
  @classmethod
  def decode_model(cls, string: str) -> 'Model':
    """
    Decodes a Base64 encoded string into a Model object using ModelSerializer.
    
    Args:
        string (str): The encoded model string.
        
    Returns:
        Model: The decoded Model object.
    """
    model = cls(ModelCodec.decode(string))
    return model
  def get_model_function(self):
    """
    Generates callable functions for the model's function and its Jacobian by dumping them to a file
    and then importing them.

    Returns:
        dict: A dictionary with 'function' and 'jacobian' as keys, and their respective callable functions as values.
    """
    # Dump the function and jacobian strings to a file
    self.dump_function()

    # File path where the functions are dumped
    func_file_path = self.func_path

    # Load the module from the file
    spec = importlib.util.spec_from_file_location(self.function_string[0], func_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[self.function_string[0]] = module
    spec.loader.exec_module(module)

    # Retrieve the function and Jacobian from the module
    model_function = getattr(module, self.function_string[0])
    model_jacobian = getattr(module, self.jacobian_string[0])

    # Store them in the callables attribute
    self.callables = {'function': model_function, 'jacobian': model_jacobian}
    
    return self.callables
  def set_bounds(self, kwargs = {"I":[0,1000],"x":[0,1000],"s":[0,3],"B":[-10,10]}) -> None:
    self.bounds = [
      #lower bounds
      [- np.inf for i in  self.lockParameterIndexMapping],
      #upper bounds
      [+ np.inf for i in  self.lockParameterIndexMapping],
    ]  
    for ind,(model_type,element_index,parameter) in enumerate(self.lockParameterIndexMapping):
      if parameter in kwargs:
        self.bounds[0][ind] = kwargs[parameter][0]
        self.bounds[1][ind] = kwargs[parameter][1] 
      if parameter == "B0" and 'B' in kwargs:
        self.bounds[0][ind] = kwargs['B'][0]
        self.bounds[1][ind] = kwargs['B'][1]  
  def copy(self):
    return copy.deepcopy(self)
