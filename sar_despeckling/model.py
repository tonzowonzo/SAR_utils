from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Input, UpSampling2D, concatenate
from keras.models import Model


def build_model():
    input_1 = Input((64, 64, 1))
    model = Conv2D(128, (3, 3), padding="same")(input_1)
    conv1 = Conv2D(128, (3, 3), padding="same")(model)
    model = MaxPool2D((2, 2))(conv1)
    model = Dropout(0.4)(model)

    model = Conv2D(64, (3, 3), padding="same", activation="relu")(model)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(model)
    model = MaxPool2D((2, 2))(conv2)
    model = Dropout(0.4)(model)

    up1 = UpSampling2D((2, 2))(model)
    concat1 = concatenate([conv2, up1])
    model = Conv2D(64, (3, 3), padding="same", activation="relu")(concat1)
    model = Conv2D(64, (3, 3), padding="same", activation="relu")(model)

    up2 = UpSampling2D((2, 2))(model)
    concat2 = concatenate([conv1, up2])
    model = Conv2D(128, (3, 3), padding="same", activation="relu")(concat2)
    model = Conv2D(128, (3, 3), padding="same", activation="relu")(model)

    model = Conv2D(256, (3, 3), padding="same", activation="relu")(model)
    model = Conv2D(1, (1, 1))(model)
    model = Model(inputs=[input_1], outputs=[model])

    model.compile(optimizer="adam", metrics=["mse", "mae"], loss="mae")

    return model


model = build_model()
print(model.summary())

