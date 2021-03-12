from tensorflow import image
import tensorflow as tf
from keras.models import *
from keras.layers import *

img_size = 256
lambda_value = 0.0002


def custom_loss(layer):
    def total_variation_loss(y_actual, y_predicted):
        loss = tf.reduce_sum(image.total_variation(y_actual - y_predicted)) * lambda_value
        mse = tf.reduce_mean((y_actual - y_predicted) ** 2)
        loss += mse
        return loss
    return total_variation_loss


def dilation_net(pretrained_weights=None, input_size=(img_size, img_size, 1)):
    """
    Architecture from: https://github.com/zhixuhao/unet/blob/master/model.py
    :param pretrained_weights: Weights to add to the UNet.
    :param input_size: The size of the input image.
    :return:
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, dilation_rate=1, activation="relu", padding="same")(inputs)
    conv2 = Conv2D(64, 3, dilation_rate=2, activation="relu", padding="same")(conv1)
    conv3 = Conv2D(64, 3, dilation_rate=3, activation="relu", padding="same")(conv2)
    conv4 = Conv2D(64, 3, dilation_rate=4, activation="relu", padding="same")(conv3)
    conv5 = Conv2D(64, 3, dilation_rate=3, activation="relu", padding="same")(conv4)
    conv6 = Conv2D(64, 3, dilation_rate=2, activation="relu", padding="same")(conv5)
    conv7 = Conv2D(1, 1, dilation_rate=1, padding="same", activation="relu", kernel_initializer="ones")(conv6)
    # model_loss = Lambda(lambda x: x[0] / x[1])([inputs, conv7])
    model = Activation(activation="tanh")(conv7)
    model = Model(inputs=[inputs], outputs=[model])

    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    print(model.summary())
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def sar_drn(input_size=(256, 256, 1)):
    """
    An implementation of SARDRN by by Zhang et al (https://arxiv.org/abs/1709.02898).

    :param input_size: The image input size, the models in this repo are trained using just
    one channel, so using more i.e the VV and VH polarisations of Sentinel 1 is untested.
    :return: A keras model.
    """
    input_sar = Input(input_size)
    conv1 = Conv2D(64, 3, activation="relu", dilation_rate=1, padding="same")(input_sar)
    conv2 = Conv2D(64, 3, activation="relu", dilation_rate=2, padding="same")(conv1)
    conv3 = Conv2D(64, 3, activation="relu", dilation_rate=3, padding="same")(conv2)
    skip_1 = Add()([conv1, conv3])
    conv4 = Conv2D(64, 3, activation="relu", dilation_rate=4, padding="same")(skip_1)
    conv5 = Conv2D(64, 3, activation="relu", dilation_rate=3, padding="same")(conv4)
    conv6 = Conv2D(64, 3, activation="relu", dilation_rate=2, padding="same",
                   kernel_initializer="he_normal")(conv5)
    skip_2 = Add()([conv4, conv6])
    conv7 = Conv2D(1, 3, dilation_rate=1, padding="same")(skip_2)
    noise_residual = Subtract()([input_sar, conv7])
    model = Model(inputs=input_sar, outputs=noise_residual)
    model.compile(optimizer="adam", loss="mse", metrics=['mae', 'mse'])
    return model




if __name__ == '__main__':
    model = dilation_net()
    model.summary()
