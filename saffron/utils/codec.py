import numpy as np
import base64
from abc import abstractmethod
class BaseModelCodec:
  def __init__(self):
    self.fractional_size = None
    self.integer_size = None
    self.available_dict_level = None
    self.available_functions = None
    self.available_function_number = None
    self.available_parameters = None
    self.available_constraints = None
    self.available_operations = None
    self.dict_level_mapping = None
    self.functions_mapping = None
    self.function_number_mapping = None
    self.parameters_mapping = None
    self.constraints_mapping = None
    self.operations_mapping = None
  @abstractmethod
  def encode(self, func_dict):...#Encode a function dictionary into a binary string.
  @abstractmethod
  def serialize(self, value):...#Serialize the codec state to a string or byte representation.
  @staticmethod
  def deserialize( serial: str):
    """
    Deserialize the codec state from a base64 encoded string to a binary string.

    Args:
      serial (str): The base64 encoded string.
    """
    index_length = serial.find('_')
    length_ascii = serial[:index_length]
    binary_data_length = base64.b64decode(length_ascii.encode('ascii'))
    str_binary_length = bin(int.from_bytes(binary_data_length, byteorder='big'))[2:].zfill(len(binary_data_length) * 8)
    length = int(str_binary_length,2)
    binary_data = base64.b64decode(serial[index_length+1:].encode('ascii'))
    str_binary = bin(int.from_bytes(binary_data, byteorder='big'))[2:].zfill(len(binary_data) * 8)
    # print(str_binary)
    if len(str_binary)<length:
      str_binary = ''.zfill(length-len(str_binary))+str_binary
    elif len(str_binary)>length:
      str_binary = str_binary[len(str_binary)-length:]
      # raise ValueError("The length of the binary string is greater than the expected length")
      
    return str_binary
  @abstractmethod
  def decode(self,binarycode):...#Decode a binary string back to a function dictionary.
  @staticmethod
  def int2base(n:int,base:int)->str:
    if n == 0:return '0'
    digits = []
    while n:
      digits.append(str(n % base))
      n //= base
    return ''.join(digits[::-1])
  @staticmethod
  def base2int(s: str, base: int) -> int:
      n = 0
      for char in s:
          n = n * base + int(char)
      return n
  @staticmethod
  def float2mybinary(value,fractional_size=2,integer_size=4,verbose= 0):
    value_int = int(value*10**fractional_size)
    max_bits = BaseModelCodec.max_basimal(10**(fractional_size+integer_size),2)
    if 2**(max_bits)<value_int+1:
      raise ValueError("Integer size is too small than the value")
    if value*10**fractional_size - value_int != 0:
      if verbose>=1: print("Fractional part size of the value is more precise than the fractional size")
    value= BaseModelCodec.int2base(value_int,2).zfill(int(max_bits))
    return value
  @staticmethod
  def mybinary2float(binary_str: str, fractional_size=2, integer_size=4) -> float:
      value_int = BaseModelCodec.base2int(binary_str, 2)

      return value_int / 10**fractional_size
  @staticmethod
  def max_basimal(number,base):
    logn = np.log(number)/np.log(base)
    # print(logn)
    ceil_base = np.ceil(logn)
    # print(ceil_base)
    return int(ceil_base) 
  @staticmethod
  def reverse_dict(d):
    return {v: k for k, v in d.items()}
class ModelCodecV1(BaseModelCodec):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.list_versions = ["1.0"]
    self.version = "1.0"
    self.fractional_size = 2
    self.integer_size = 4
    
    self.available_dict_level      = ["model_type","element_index","parameter"]
    self.available_functions       = ['gaussian',"polynome" ]
    self.available_function_number = [i for i in range(32)]
    self.available_parameters      = ['I', 's', 'x', 'lims', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B9','B10','B11']
    self.available_constraints     = ['free','lock']
    self.available_operations      = ['add','mul']
    self.dict_level_mapping        = {self.available_dict_level      [i]:self.int2base(i,2).zfill((self.max_basimal(len(self.available_dict_level      ),2))) for i in range(len(self.available_dict_level      ))}
    self.functions_mapping         = {self.available_functions       [i]:self.int2base(i,2).zfill((self.max_basimal(len(self.available_functions       ),2))) for i in range(len(self.available_functions       ))}
    self.function_number_mapping   = {self.available_function_number [i]:self.int2base(i,2).zfill((self.max_basimal(len(self.available_function_number ),2))) for i in range(len(self.available_function_number ))}
    self.parameters_mapping        = {self.available_parameters      [i]:self.int2base(i,2).zfill((self.max_basimal(len(self.available_parameters      ),2))) for i in range(len(self.available_parameters      ))}
    self.constraints_mapping       = {self.available_constraints     [i]:self.int2base(i,2).zfill((self.max_basimal(len(self.available_constraints     ),2))) for i in range(len(self.available_constraints     ))}
    self.operations_mapping        = {self.available_operations      [i]:self.int2base(i,2).zfill((self.max_basimal(len(self.available_operations      ),2))) for i in range(len(self.available_operations      ))}
  def encode(self, func_dict):
    code_version = (self.float2mybinary(float(self.version),2,2))
    str_binary = f"{code_version}"
    for model_type in func_dict:
      str_binary+= self.dict_level_mapping['model_type']
      str_binary+= self.functions_mapping[model_type] 
      for element_index in func_dict[model_type]:
        str_binary+= self.dict_level_mapping['element_index']
        str_binary+= self.function_number_mapping[element_index]
        for parameter in func_dict[model_type][element_index]:
          str_binary+= self.dict_level_mapping['parameter']
          str_binary+= self.parameters_mapping[parameter]
          if isinstance(func_dict[model_type][element_index][parameter],dict):
            str_binary += self.constraints_mapping[func_dict[model_type][element_index][parameter]['constraint']]
            str_binary += self.operations_mapping[func_dict[model_type][element_index][parameter]['operation']]
            str_binary += self.float2mybinary(func_dict[model_type][element_index][parameter]['value'],fractional_size=self.fractional_size,integer_size=self.integer_size)
            str_binary += self.functions_mapping[func_dict[model_type][element_index][parameter]['reference']['model_type']]
            str_binary += self.function_number_mapping[func_dict[model_type][element_index][parameter]['reference']['element_index']]
            str_binary += self.parameters_mapping[func_dict[model_type][element_index][parameter]['reference']['parameter']]
          elif parameter == 'lims':
            lims = func_dict[model_type][element_index][parameter]
            code_lims = []
            for lim in lims:
              if lim == -np.inf:
                code_lims.append('11111111111111111110')
              elif lim == np.inf:
                code_lims.append('11111111111111111111')
              else:
                code_lims.append(self.float2mybinary(lim,fractional_size=self.fractional_size,integer_size=self.integer_size))
            
            str_binary += code_lims[0]
            str_binary += code_lims[1]
          else:
            str_binary += self.constraints_mapping['free']
            str_binary += self.float2mybinary(func_dict[model_type][element_index][parameter],fractional_size=self.fractional_size,integer_size=self.integer_size)
    return str_binary
  def serialize(self, value):
    if isinstance(value,dict):
      value = self.encode(value)
    elif isinstance(value,str):
      pass
    value2 = BaseModelCodec.int2base(len(value),2)
    binary_length = int(value2, 2).to_bytes((len(value2) + 7) // 8, byteorder='big')
    base64_length = base64.b64encode(binary_length)
    ascii_length  = base64_length.decode('ascii')
    
    binary_data = int(value, 2).to_bytes((len(value) + 7) // 8, byteorder='big')
    base64_encoded = base64.b64encode(binary_data)
    ascii_representation = base64_encoded.decode('ascii')
    
    return ascii_length+"_"+ascii_representation
  def decode(self, binarycode,verbose=0):
    """
    Decode the binary string back to a function dictionary.

    Returns:
      dict: The function dictionary reconstructed from the binary string.
    """
    func_dict = {}
    idx = 0
    #check if it is serial or binary 
    for val in binarycode:
      if val not in ["0",'1']:
        print('This is not a bit string, treating it as a serial')
        binarycode = self.deserialize(binarycode)
        break
    # Decode version
    version_max_basimal = self.max_basimal(10**(2 + 2), 2)
    version_float = self.mybinary2float(binarycode[idx:idx + version_max_basimal], fractional_size=2, integer_size=2)
    if version_float != float(self.version):
      raise ValueError(f"This class ({self.__class__.__name__}) is not compatible with the version of the binary string ({version_float})")
    idx += version_max_basimal

    # Start decoding the function dictionary
    while idx < len(binarycode):
      #what is the level we are working with
      dict_level_code = binarycode[idx:idx + len(list(self.dict_level_mapping.items())[0][1])] 
      dict_level_value = BaseModelCodec.reverse_dict(self.dict_level_mapping)[dict_level_code]
      if verbose>=1: print("deciding level code:",dict_level_code,'decode: ',dict_level_value)
      idx += len(dict_level_code)
      
      if dict_level_value == 'model_type':
        # Decode model type
        model_type_code = binarycode[idx:idx + len(list(self.functions_mapping.items())[0][1])]
        model_type = BaseModelCodec.reverse_dict(self.functions_mapping)[model_type_code]
        if model_type not in func_dict: func_dict[model_type] = {}
        if verbose>=1: print("model type code:",model_type_code,'decode: ',model_type)
        
        idx += len(model_type_code)
      
      if dict_level_value == 'element_index':
        # Decode element index
        element_index_code = binarycode[idx:idx + len(list(self.function_number_mapping.items())[0][1])]
        element_index = BaseModelCodec.reverse_dict(self.function_number_mapping)[element_index_code]
        if element_index not in func_dict[model_type]: func_dict[model_type][element_index] = {}
        idx += len(element_index_code)
        if verbose>=1: print("element index code:",element_index_code,'decode: ',element_index)
      if dict_level_value == 'parameter':
        # Decode parameter
        parameter_code = binarycode[idx:idx + len(list(self.parameters_mapping.items())[0][1])]
        parameter = BaseModelCodec.reverse_dict(self.parameters_mapping)[parameter_code]
        idx += len(parameter_code)
        if parameter not in func_dict[model_type][element_index]: func_dict[model_type][element_index][parameter] = None
        if verbose>=1: print("parameter code:",parameter_code,'decode: ',parameter)
        
        if parameter == 'lims':
          if verbose>=1: print("lims code:")
          # Decode the lims parameter
          lower_lim_code = binarycode[idx:idx + BaseModelCodec.max_basimal(10**(self.fractional_size + self.integer_size), 2)]
          idx += 20
          upper_lim_code = binarycode[idx:idx + BaseModelCodec.max_basimal(10**(self.fractional_size + self.integer_size), 2)]
          idx += 20

          if lower_lim_code == '11111111111111111110':
            lower_lim = -np.inf
          elif lower_lim_code == '11111111111111111111':
            lower_lim = np.inf
          else:
            lower_lim = self.mybinary2float(lower_lim_code, self.fractional_size, self.integer_size)

          if upper_lim_code == '11111111111111111110':
            upper_lim = -np.inf
          elif upper_lim_code == '11111111111111111111':
            upper_lim = np.inf
          else:
            upper_lim = self.mybinary2float(upper_lim_code, self.fractional_size, self.integer_size)
          
          func_dict[model_type][element_index][parameter] = [lower_lim,upper_lim]
          if verbose>=1: print("    lower lim code:",lower_lim_code,'decode: ',lower_lim)
          if verbose>=1: print("    upper lim code:",upper_lim_code,'decode: ',upper_lim)
          
        else:
          # Decode whether parameter is locked or free
          constraint_code = binarycode[idx:idx + len(list(self.constraints_mapping.items())[0][1])]
          constraint = BaseModelCodec.reverse_dict(self.constraints_mapping)[constraint_code]
          idx += len(constraint_code)
          
          if constraint == 'free':
            if verbose>=1: print("free code:")
            # Decode free parameter value
            param_value_code = binarycode[idx:idx + self.max_basimal(10**(self.fractional_size + self.integer_size), 2)]
            idx += len(param_value_code)
            func_dict[model_type][element_index][parameter] = self.mybinary2float(param_value_code, self.fractional_size, self.integer_size)
            if verbose>=1: print("    parameter value code:",param_value_code,'decode: ',func_dict[model_type][element_index][parameter])
            
          elif constraint == 'lock':
            if verbose>=1: print("lock code:")
            # Decode lock parameters
            operation_code = binarycode[idx:idx + len(list(self.operations_mapping.items())[0][1])]
            operation = BaseModelCodec.reverse_dict(self.operations_mapping)[operation_code]
            idx += len(operation_code)
            if verbose>=1: print("    operation code:",operation_code,'decode: ',operation)
            
            value_code = binarycode[idx:idx + self.max_basimal(10**(self.fractional_size + self.integer_size), 2)]
            value = self.mybinary2float(value_code, self.fractional_size, self.integer_size)
            idx += len(value_code)
            if verbose>=1: print("    value code:",value_code,'decode: ',value)
            
            reference_model_type_code = binarycode[idx:idx + len(list(self.functions_mapping.items())[0][1])]
            reference_model_type = BaseModelCodec.reverse_dict(self.functions_mapping)[reference_model_type_code]
            idx += len(reference_model_type_code)
            if verbose>=1: print("    reference model type code:",reference_model_type_code,'decode: ',reference_model_type)
            
            reference_element_index_code = binarycode[idx:idx + len(list(self.function_number_mapping.items())[0][1])]
            reference_element_index = BaseModelCodec.reverse_dict(self.function_number_mapping)[reference_element_index_code]
            idx += len(reference_element_index_code)
            if verbose>=1: print("    reference element index code:",reference_element_index_code,'decode: ',reference_element_index)
            
            reference_parameter_code = binarycode[idx:idx + len(list(self.parameters_mapping.items())[0][1])]
            reference_parameter = BaseModelCodec.reverse_dict(self.parameters_mapping)[reference_parameter_code]
            idx += len(reference_parameter_code)
            if verbose>=1: print("    reference parameter code:",reference_parameter_code,'decode: ',reference_parameter)
            
            func_dict[model_type][element_index][parameter] = {
              "constraint": "lock",
              "operation": operation,
              "value": value,
              "reference": {
                "model_type": reference_model_type,
                "element_index": reference_element_index,
                "parameter": reference_parameter
              }
            }
            
            if verbose>=1: print("locks code:",[],'decode: ',operation)
    return func_dict
class ModelCodec:
  version_class_map = {
    '1.0': ModelCodecV1,
    }
  def __new__(cls, version='1.0', *args, **kwargs):
    if version == 'latest':
      version = sorted(cls.version_class_map.keys())[-1]
    if version in cls.version_class_map:
      return cls.version_class_map[version](*args, **kwargs)
    else:
      raise ValueError(f"Unsupported version {version}")
  @staticmethod
  def deserialize( serial: str):
    """
    Deserialize the codec state from a base64 encoded string to a binary string.

    Args:
      serial (str): The base64 encoded string.
    """
    index_length = serial.find('_')
    length_ascii = serial[:index_length]
    binary_data_length = base64.b64decode(length_ascii.encode('ascii'))
    str_binary_length = bin(int.from_bytes(binary_data_length, byteorder='big'))[2:].zfill(len(binary_data_length) * 8)
    length = int(str_binary_length,2)
    binary_data = base64.b64decode(serial[index_length+1:].encode('ascii'))
    str_binary = bin(int.from_bytes(binary_data, byteorder='big'))[2:].zfill(len(binary_data) * 8)
    # print(str_binary)
    if len(str_binary)<length:
      str_binary = ''.zfill(length-len(str_binary))+str_binary
    elif len(str_binary)>length:
      str_binary = str_binary[len(str_binary)-length:]
      # raise ValueError("The length of the binary string is greater than the expected length")
      
    return str_binary
  @staticmethod
  def decode(binarycode):
    for bit in binarycode:
      if bit in ['0',"1"]:
        continue
      else:
        print('This is not a bit string, treating it as a serial')
        binarycode = ModelCodec.deserialize(binarycode)
        break
    version_max_basimal = BaseModelCodec.max_basimal(10**(2+2),2)
    version_float = BaseModelCodec.mybinary2float(binarycode[:version_max_basimal],fractional_size=2,integer_size=2)
    version_str = f"{version_float}"
    print("recognized codec version:",version_float)
    codec = ModelCodec(version=version_str)
    return codec.decode(binarycode)
