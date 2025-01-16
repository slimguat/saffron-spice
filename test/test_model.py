import numpy as np
from saffron import ModelFactory 

def test_gaussian_model():
    model = ModelFactory()
    model.add_gaussian(I=10, x=1000, s=0.1)

    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_function = callables['function']
    model_jacobian = callables['jacobian']

    # Define test inputs
    x_values = np.array([900, 1000, 1100])
    
    # Expected Gaussian output
    expected_output = 10 * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    
    # Expected Jacobian components
    expected_jacobian_I = np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_jacobian_x = 10 * (x_values - 1000) / (0.1**2) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_jacobian_s = 10 * (x_values - 1000)**2 / (0.1**3) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    
    # Get model outputs
    model_output = model_function(x_values, 10, 1000, 0.1)
    model_jacobian_output = model_jacobian(x_values, 10, 1000, 0.1)
    
    # Assertions
    np.testing.assert_almost_equal(model_output, expected_output, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_jacobian_I, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_jacobian_x, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_jacobian_s, decimal=5)

def test_multiple_gaussians():
    model = ModelFactory()
    model.add_gaussian(I=5, x=900, s=0.1)
    model.add_gaussian(I=10, x=1000, s=0.2)
    model.add_polynome(5,2,5)
    
    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_function = callables['function']
    model_jacobian = callables['jacobian']

    # Define test inputs
    x_values = np.array([900, 950, 1000, 1050])
    
    # Expected Gaussian output
    expected_output = (5 * np.exp(-(x_values - 900)**2 / (2 * 0.1**2)) +
                       10 * np.exp(-(x_values - 1000)**2 / (2 * 0.2**2))+
                       5 + 2*x_values + 5*x_values**2)
    
    # Get model outputs
    model_output = model_function(x_values, 5, 900, 0.1, 10, 1000, 0.2, 5, 2, 5)
    
    # Assertions
    np.testing.assert_almost_equal(model_output, expected_output, decimal=5)

def test_constraint_model():
    model = ModelFactory()
    model.add_gaussian(I=10, x=1000, s=0.1)
    model.add_gaussian(I=5, x=0, s=0.1)

    # Lock the x parameter of the second Gaussian to the first one
    model.lock(
        param_1={'model_type': 'gaussian', 'element_index': 1, 'parameter': 'x'},
        param_2={'model_type': 'gaussian', 'element_index': 0, 'parameter': 'x'},
        lock_protocol={'operation': 'add', 'value': 100}
    )
    
    # Generate the model function
    callables = model.get_model_function()
    model_function = callables['function']

    # Define test inputs
    x_values = np.array([900, 950, 1000, 1050])
    
    # Expected Gaussian output with constraint applied
    expected_output = (10 * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2)) +
                       5 * np.exp(-(x_values - (1000 + 100))**2 / (2 * 0.1**2)))
    
    # Get model outputs
    model_output = model_function(x_values, 10, 1000, 0.1, 5, 0.1)
    
    # Assertions
    np.testing.assert_almost_equal(model_output, expected_output, decimal=5)

def test_gaussian_jacobian():
    model = ModelFactory()
    model.add_gaussian(I=10, x=1000, s=0.1)

    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_jacobian = callables['jacobian']

    # Define test inputs
    x_values = np.array([900, 1000, 1100])
    
    # Expected Jacobian components
    expected_jacobian_I = np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_jacobian_x = 10 * (x_values - 1000) / (0.1**2) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_jacobian_s = 10 * (x_values - 1000)**2 / (0.1**3) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))

    # Get model Jacobian output
    model_jacobian_output = model_jacobian(x_values, 10, 1000, 0.1)
    
    # Assertions
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_jacobian_I, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_jacobian_x, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_jacobian_s, decimal=5)

def test_multiple_gaussians_jacobian():
    model = ModelFactory()
    model.add_gaussian(I=5, x=900, s=0.1)
    model.add_gaussian(I=10, x=1000, s=0.2)
    
    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_jacobian = callables['jacobian']

    # Define test inputs
    x_values = np.array([900, 950, 1000, 1050])
    
    # Expected Jacobian for Gaussian 1
    expected_jacobian_1_I = np.exp(-(x_values - 900)**2 / (2 * 0.1**2))
    expected_jacobian_1_x = 5 * (x_values - 900) / (0.1**2) * np.exp(-(x_values - 900)**2 / (2 * 0.1**2))
    expected_jacobian_1_s = 5 * (x_values - 900)**2 / (0.1**3) * np.exp(-(x_values - 900)**2 / (2 * 0.1**2))
    
    # Expected Jacobian for Gaussian 2
    expected_jacobian_2_I = np.exp(-(x_values - 1000)**2 / (2 * 0.2**2))
    expected_jacobian_2_x = 10 * (x_values - 1000) / (0.2**2) * np.exp(-(x_values - 1000)**2 / (2 * 0.2**2))
    expected_jacobian_2_s = 10 * (x_values - 1000)**2 / (0.2**3) * np.exp(-(x_values - 1000)**2 / (2 * 0.2**2))

    # Get model Jacobian output
    model_jacobian_output = model_jacobian(x_values, 5, 900, 0.1, 10, 1000, 0.2)
    
    # Assertions for Gaussian 1
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_jacobian_1_I, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_jacobian_1_x, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_jacobian_1_s, decimal=5)
    
    # Assertions for Gaussian 2
    np.testing.assert_almost_equal(model_jacobian_output[:, 3], expected_jacobian_2_I, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 4], expected_jacobian_2_x, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 5], expected_jacobian_2_s, decimal=5)

def test_gaussian_polynomial_jacobian():
    model = ModelFactory()
    model.add_gaussian(I=10, x=1000, s=0.1)
    model.add_polynome(1, 2, 3, lims=[700, 1300])

    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_jacobian = callables['jacobian']

    # Define test inputs
    x_values = np.array([800, 900, 1000, 1100])

    # Expected Gaussian Jacobian components
    expected_gaussian_jacobian_I = np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_gaussian_jacobian_x = 10 * (x_values - 1000) / (0.1**2) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_gaussian_jacobian_s = 10 * (x_values - 1000)**2 / (0.1**3) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))

    # Expected Polynomial Jacobian components
    expected_polynomial_jacobian_B0 = 1
    expected_polynomial_jacobian_B1 = x_values
    expected_polynomial_jacobian_B2 = x_values**2

    # Get model Jacobian output
    model_jacobian_output = model_jacobian(x_values, 10, 1000, 0.1, 1, 2, 3)

    # Assertions for Gaussian
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_gaussian_jacobian_I, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_gaussian_jacobian_x, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_gaussian_jacobian_s, decimal=5)

    # Assertions for Polynomial
    np.testing.assert_almost_equal(model_jacobian_output[:, 3], expected_polynomial_jacobian_B0, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 4], expected_polynomial_jacobian_B1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 5], expected_polynomial_jacobian_B2, decimal=5)

def test_constraint_jacobian():
    model = ModelFactory()
    model.add_gaussian(I=10, x=1000, s=0.1)
    model.add_gaussian(I=5, x=0, s=0.1)

    # Lock the x parameter of the second Gaussian to the first one
    model.lock(
        param_1={'model_type': 'gaussian', 'element_index': 1, 'parameter': 'x'},
        param_2={'model_type': 'gaussian', 'element_index': 0, 'parameter': 'x'},
        lock_protocol={'operation': 'add', 'value': 100}
    )

    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_jacobian = callables['jacobian']

    # Define test inputs
    x_values = np.array([900, 1000, 1100, 1200])

    # Expected Jacobian components (considering the constraint)
    expected_jacobian_I1 = np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_jacobian_x1 = 10 * (x_values - 1000) / (0.1**2) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_jacobian_s1 = 10 * (x_values - 1000)**2 / (0.1**3) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    expected_jacobian_I2 = np.exp(-(x_values - (1000 + 100))**2 / (2 * 0.1**2))
    expected_jacobian_s2 = 5 * (x_values - (1000 + 100))**2 / (0.1**3) * np.exp(-(x_values - (1000 + 100))**2 / (2 * 0.1**2))
    # Get model Jacobian output
    model_jacobian_output = model_jacobian(x_values, 10, 1000, 0.1, 5, 0.1)

    # Assertions for Gaussian 1
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_jacobian_I1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_jacobian_x1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_jacobian_s1, decimal=5)
    
    # Assertions for Gaussian 2 with constraint
    np.testing.assert_almost_equal(model_jacobian_output[:, 3], expected_jacobian_I2, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 4], expected_jacobian_s2, decimal=5)

def test_polynomial_with_lims():
    # Create a model factory
    model = ModelFactory()
    
    # Add a polynomial with specific limits
    lims=[900, 1000]
    model.add_polynome(1, 2, 3, lims=lims)

    # Generate the model function
    callables = model.get_model_function()
    model_function = callables['function']

    # Define test inputs, including values inside and outside the limits
    x_values = np.array([800, 900, 950, 1000, 1050, 1100, 1200])
    x_values = np.arange(800,1200,1)
    
    # Expected output: The polynomial is applied only within the limits [900, 1100]
    expected_output = np.zeros_like(x_values)
    inside_lims_indices = (x_values > lims[0]) & (x_values < lims[1])
    expected_output[inside_lims_indices] = 1 + 2 * x_values[inside_lims_indices] + 3 * x_values[inside_lims_indices]**2

    # Get model output
    model_output = model_function(x_values, 1, 2, 3)

    # Assertions
    np.testing.assert_almost_equal(model_output, expected_output, decimal=5)

def test_combined_polynomial_gaussian_with_constraints():
    # Create a model factory
    model = ModelFactory()

    # Add four polynomials with different limits
    lims1 = [900, 1000]
    lims2 = [1000, 1100]
    lims3 = [1100, 1200]
    lims4 = [1200, 1300]

    model.add_polynome(1, 2, 3, lims=lims1)
    model.add_polynome(2, 3, 4, lims=lims2)
    model.add_polynome(3, 4, 5, lims=lims3)
    model.add_polynome(4, 5, 6, lims=lims4)

    # Add four Gaussians
    model.add_gaussian(I=10, x=1000, s=0.1)
    model.add_gaussian(I=5, x=0, s=0.2)
    model.add_gaussian(I=7, x=0, s=0.3)
    model.add_gaussian(I=3, x=0, s=0.4)

    # Apply constraints between Gaussians
    model.lock(
        param_1={'model_type': 'gaussian', 'element_index': 1, 'parameter': 'x'},
        param_2={'model_type': 'gaussian', 'element_index': 0, 'parameter': 'x'},
        lock_protocol={'operation': 'add', 'value': 100}
    )
    model.lock(
        param_1={'model_type': 'gaussian', 'element_index': 2, 'parameter': 'x'},
        param_2={'model_type': 'gaussian', 'element_index': 0, 'parameter': 'x'},
        lock_protocol={'operation': 'add', 'value': 200}
    )
    model.lock(
        param_1={'model_type': 'gaussian', 'element_index': 3, 'parameter': 'x'},
        param_2={'model_type': 'gaussian', 'element_index': 0, 'parameter': 'x'},
        lock_protocol={'operation': 'add', 'value': 300}
    )

    # Generate the model function
    callables = model.get_model_function()
    model_function = callables['function']

    # Define test inputs
    x_values = np.arange(800, 1400, 1).astype(float)

    # Expected output: 
    expected_output = np.zeros_like(x_values)
    
    # Calculate expected polynomial contributions
    inside_lims_indices_1 = (x_values > lims1[0]) & (x_values < lims1[1])
    expected_output[inside_lims_indices_1] += 1 + 2 * x_values[inside_lims_indices_1] + 3 * x_values[inside_lims_indices_1]**2
    
    inside_lims_indices_2 = (x_values > lims2[0]) & (x_values < lims2[1])
    expected_output[inside_lims_indices_2] += 2 + 3 * x_values[inside_lims_indices_2] + 4 * x_values[inside_lims_indices_2]**2

    inside_lims_indices_3 = (x_values > lims3[0]) & (x_values < lims3[1])
    expected_output[inside_lims_indices_3] += 3 + 4 * x_values[inside_lims_indices_3] + 5 * x_values[inside_lims_indices_3]**2

    inside_lims_indices_4 = (x_values > lims4[0]) & (x_values < lims4[1])
    expected_output[inside_lims_indices_4] += 4 + 5 * x_values[inside_lims_indices_4] + 6 * x_values[inside_lims_indices_4]**2
    
    # Calculate expected Gaussian contributions
    expected_output += (10 * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2)) +
                        5 * np.exp(-(x_values - 1100)**2 / (2 * 0.2**2)) +
                        7 * np.exp(-(x_values - 1200)**2 / (2 * 0.3**2)) +
                        3 * np.exp(-(x_values - 1300)**2 / (2 * 0.4**2)))

    # Get model output
    model_output = model_function(x_values, 
                                  10, 1000, 0.1,  # Gaussian 1
                                  5, 0.2,         # Gaussian 2 (x is constrained)
                                  7, 0.3,         # Gaussian 3 (x is constrained)
                                  3, 0.4,         # Gaussian 4 (x is constrained)
                                  1, 2, 3,        # Polynomial 1
                                  2, 3, 4,        # Polynomial 2
                                  3, 4, 5,        # Polynomial 3
                                  4, 5, 6)        # Polynomial 4

    # Assertions
    np.testing.assert_almost_equal(model_output, expected_output, decimal=5)

def test_single_polynomial_jacobian_with_lims():
    # Create a model factory
    model = ModelFactory()
    
    # Add a single polynomial with specific limits
    lims = [900, 1000]
    model.add_polynome(1, 2, 3, lims=lims)

    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_function = callables['function']
    model_jacobian = callables['jacobian']

    # Define test inputs, including values inside and outside the limits
    x_values = np.arange(800, 1100, 1)

    # Expected output: The polynomial is applied only within the limits [900, 1000]
    expected_output = np.zeros_like(x_values)
    inside_lims_indices = (x_values > lims[0]) & (x_values < lims[1])
    expected_output[inside_lims_indices] = 1 + 2 * x_values[inside_lims_indices] + 3 * x_values[inside_lims_indices]**2

    # Expected Polynomial Jacobians within limits
    expected_jacobian_B0 = np.zeros_like(x_values)
    expected_jacobian_B1 = np.zeros_like(x_values)
    expected_jacobian_B2 = np.zeros_like(x_values)

    expected_jacobian_B0[inside_lims_indices] = 1
    expected_jacobian_B1[inside_lims_indices] = x_values[inside_lims_indices]
    expected_jacobian_B2[inside_lims_indices] = x_values[inside_lims_indices]**2

    # Get model output and Jacobian output
    model_output = model_function(x_values, 1, 2, 3)
    model_jacobian_output = model_jacobian(x_values, 1, 2, 3)

    # Assertions for the output
    np.testing.assert_almost_equal(model_output, expected_output, decimal=5)

    # Assertions for the Jacobian
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_jacobian_B0, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_jacobian_B1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_jacobian_B2, decimal=5)

    #plot each component of the jacobian with the expected value
def test_polynomial_and_gaussian_jacobian_with_lims():
    # Create a model factory
    model = ModelFactory()
    
    # Add a polynomial with specific limits
    lims = [900, 1000]
    model.add_polynome(1, 2, 3, lims=lims)
    
    # Add a Gaussian component
    model.add_gaussian(I=10, x=950, s=50)

    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_function = callables['function']
    model_jacobian = callables['jacobian']

    # Define test inputs, including values inside and outside the polynomial limits
    x_values = np.arange(800, 1100, 1, dtype=float)

    # Expected output: The polynomial is applied only within the limits [900, 1000]
    # and the Gaussian is applied over the entire range
    expected_output = np.zeros_like(x_values)
    inside_lims_indices = (x_values > lims[0]) & (x_values < lims[1])
    expected_output[inside_lims_indices] = 1 + 2 * x_values[inside_lims_indices] + 3 * x_values[inside_lims_indices]**2
    expected_output += 10 * np.exp(-(x_values - 950)**2 / (2 * 50**2))

    # Expected Polynomial Jacobians within limits
    expected_jacobian_B0 = np.zeros_like(x_values)
    expected_jacobian_B1 = np.zeros_like(x_values)
    expected_jacobian_B2 = np.zeros_like(x_values)

    expected_jacobian_B0[inside_lims_indices] = 1
    expected_jacobian_B1[inside_lims_indices] = x_values[inside_lims_indices]
    expected_jacobian_B2[inside_lims_indices] = x_values[inside_lims_indices]**2

    # Expected Gaussian Jacobians
    gaussian_exp = np.exp(-(x_values - 950)**2 / (2 * 50**2))
    expected_jacobian_I = gaussian_exp
    expected_jacobian_x = 10 * (x_values - 950) / (50**2) * gaussian_exp
    expected_jacobian_s = 10 * ((x_values - 950)**2 / (50**3)) * gaussian_exp

    # Get model output and Jacobian output
    model_output = model_function(x_values, 10, 950, 50,1, 2, 3)
    model_jacobian_output = model_jacobian(x_values,  10, 950, 50,1, 2, 3)

    # Assertions for the output
    np.testing.assert_almost_equal(model_output, expected_output, decimal=5)

    # Assertions for the Jacobian
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_jacobian_I, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_jacobian_x, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_jacobian_s, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 3], expected_jacobian_B0, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 4], expected_jacobian_B1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 5], expected_jacobian_B2, decimal=5)

def test_single_polynomial_gaussian_jacobian_with_constraints():
    # Create a model factory
    model = ModelFactory()

    # Add a single polynomial with specific limits
    lims = [900, 1000]
    model.add_polynome(1, 2, 3, lims=lims)

    # Add three Gaussians
    model.add_gaussian(I=10, x=1000, s=0.1)
    model.add_gaussian(I=5, x=0, s=0.2)
    model.add_gaussian(I=7, x=0, s=0.3)

    # Apply constraints between Gaussians
    model.lock(
        param_1={'model_type': 'gaussian', 'element_index': 1, 'parameter': 'x'},
        param_2={'model_type': 'gaussian', 'element_index': 0, 'parameter': 'x'},
        lock_protocol={'operation': 'add', 'value': 100}
    )
    model.lock(
        param_1={'model_type': 'gaussian', 'element_index': 2, 'parameter': 'x'},
        param_2={'model_type': 'gaussian', 'element_index': 0, 'parameter': 'x'},
        lock_protocol={'operation': 'add', 'value': 200}
    )

    # Generate the model function and its Jacobian
    callables = model.get_model_function()
    model_function = callables['function']
    model_jacobian = callables['jacobian']

    # Define test inputs
    x_values = np.arange(800, 1100, 1,dtype=float)

    # Get model Jacobian output
    model_jacobian_output = model_jacobian(
        x_values, 
        10, 1000, 0.1,  # Gaussian 1
        5, 0.2,         # Gaussian 2 (x is constrained)
        7, 0.3,         # Gaussian 3 (x is constrained)
        1, 2, 3         # Polynomial
    )

    # Expected Gaussian Jacobians
    expected_jacobian_I1 = np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    
    # Derivative for the first Gaussian (x1)
    derivative_x1 = 10 * (x_values - 1000) / (0.1**2) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    # Derivative for the second Gaussian (x2 = x1 + 100)
    derivative_x2 = 5 * (x_values - (1100)) / (0.2**2) * np.exp(-(x_values - (1100))**2 / (2 * 0.2**2))
    # Derivative for the third Gaussian (x3 = x1 + 200)
    derivative_x3 = 7 * (x_values - (1200)) / (0.3**2) * np.exp(-(x_values - (1200))**2 / (2 * 0.3**2))
    # Combine all derivatives
    expected_jacobian_x1 = derivative_x1 + derivative_x2 + derivative_x3
        
    expected_jacobian_s1 = 10 * (x_values - 1000)**2 / (0.1**3) * np.exp(-(x_values - 1000)**2 / (2 * 0.1**2))
    
    expected_jacobian_I2 = np.exp(-(x_values - 1100)**2 / (2 * 0.2**2))
    expected_jacobian_s2 = 5 * (x_values - 1100)**2 / (0.2**3) * np.exp(-(x_values - 1100)**2 / (2 * 0.2**2))
    
    expected_jacobian_I3 = np.exp(-(x_values - 1200)**2 / (2 * 0.3**2))
    expected_jacobian_s3 = 7 * (x_values - 1200)**2 / (0.3**3) * np.exp(-(x_values - 1200)**2 / (2 * 0.3**2))

    # Expected Polynomial Jacobians within limits
    inside_lims_indices = (x_values > lims[0]) & (x_values < lims[1])
    expected_polynomial_jacobian_B0 = np.zeros_like(x_values)
    expected_polynomial_jacobian_B0[inside_lims_indices] = 1
    
    expected_polynomial_jacobian_B1 = np.zeros_like(x_values)
    expected_polynomial_jacobian_B1[inside_lims_indices] = x_values[inside_lims_indices]
    
    expected_polynomial_jacobian_B2 = np.zeros_like(x_values)
    expected_polynomial_jacobian_B2[inside_lims_indices] = x_values[inside_lims_indices]**2

    # Assertions for Gaussian 1
    np.testing.assert_almost_equal(model_jacobian_output[:, 0], expected_jacobian_I1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 1], expected_jacobian_x1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 2], expected_jacobian_s1, decimal=5)
    
    # Assertions for Gaussian 2 with constraint
    np.testing.assert_almost_equal(model_jacobian_output[:, 3], expected_jacobian_I2, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 4], expected_jacobian_s2, decimal=5)
    
    # Assertions for Gaussian 3 with constraint
    np.testing.assert_almost_equal(model_jacobian_output[:, 5], expected_jacobian_I3, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 6], expected_jacobian_s3, decimal=5)
    
    # Assertions for Polynomial
    np.testing.assert_almost_equal(model_jacobian_output[:, 7], expected_polynomial_jacobian_B0, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 8], expected_polynomial_jacobian_B1, decimal=5)
    np.testing.assert_almost_equal(model_jacobian_output[:, 9], expected_polynomial_jacobian_B2, decimal=5)
