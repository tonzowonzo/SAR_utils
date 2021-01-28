from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, BatchNormalization, Input, UpSampling2D, concatenate
from keras.layers import Lambda, GlobalAveragePooling2D
import keras.backend as K
from keras.models import Model

def logFunc(x):
    """
    Takes the log of an input and returns it.

    :param x:
    :return:
    """
    return K.log(x)


def expFunc(x):
    """
    Takes the inverse of a log and returns it.

    :param x:
    :return:
    """
    return K.exp(x)


def build_model():
    input_1 = Input((64, 64, 1))
    model = Conv2D(64, 3, padding="same", activation="relu")(input_1)

    # Main convolutional block.
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(model)
    model = BatchNormalization()(model)

    # Output layer.
    model = Conv2D(1, 3, activation="linear", padding="same")(model)
    model = Model(inputs=[input_1], outputs=[model])
    model.compile(optimizer="adam", metrics=["mse", "mae"], loss="mse")

    return model


model = build_model()
print(model.summary())

