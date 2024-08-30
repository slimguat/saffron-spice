from saffron import ModelFactory , ModelCodec

def test_codec():
    model = ModelFactory()
    model.add_gaussian(I=10, x=1000, s=0.1)
    model.add_polynome(1, 2, 3, lims=[700, 1300])

    # Encode the model
    codec = ModelCodec(version='latest')
    encoded_model = codec.serialize(model.functions)

    # Decode the model
    decoded_functions = codec.decode(encoded_model)

    # Assertions
    assert model.functions == decoded_functions, "Encoded and decoded functions do not match the original"
