from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def create_mlp(dim, regress=False):
    """
    create a multi-layer perceptron architeture with: dim-8-4
    :param dim:
    :param regress:
    :return:
    """
    #define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation='relu'))
    model.add(Dense(4, activation='relu'))

    # check to see if the regression node should be added
    # If we are performing regression, we add a Dense layer containing a
    # single neuron with a linear activation function (below).
    # Typically we use ReLU-based activations, but since we are performing
    # regression we need a linear activation.
    if regress:
        model.add(Dense(1, activation="linear"))

    return model
