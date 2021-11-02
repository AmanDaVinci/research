from abc import ABC, abstractmethod

class BasePredictor(ABC):
    """ Abstract base class for all Model Predictor classes """

    @property
    @abstractmethod
    def display_name(self):
        """ Human readable display name of the model """
        raise NotImplementedError

    @property
    @abstractmethod
    def qualified_name(self):
        """ Unique qualified name of the model """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_file(self):
        """ File having the trained parameters of the model """
        raise NotImplementedError

    @property
    @abstractmethod
    def input_schema(self):
        """ Data schema accepted by the predict() method """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_schema(self):
        """ Data schema returned by the predict() method """
        raise NotImplementedError

    @abstractmethod
    def __init__(self):
        """ Loads, deserializes and initializes the model """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, data):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, data):
        raise NotImplementedError

    def __call__(self, data):
        try:
            self.input_schema.validate(data)
        except Exception as e:
            raise PredictorSchemaValidationException(
                f"Invalid input data: {str(e)}"
            )
        model_input = self.preprocess(data)
        model_output = self.predict(model_input)
        output = self.postprocess(model_output)
        try:
            self.output_schema.validate(output)
        except Exception as e:
            raise PredictorSchemaValidationException(
                f"Invalid output data: {str(e)}"
            )
        return output
    
    def __str__(self):
        return self.qualified_name



class PredictorSchemaValidationException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)