from keras.models import load_model
from data_generator import generator
import cv2
gen = generator()


model = load_model("C:/Users/tim.iles/noise_model_noise2.h5")
path = "C:/Users/tim.iles/Documents/Projects/sar_speckle_filtering/tests/"

X, y = next(gen)
pred = model.predict(X)

pred = pred.reshape((32, 64, 64))
X = X.reshape((32, 64, 64))
y = y.reshape((32, 64, 64))


for i in range(32):
    cv2.imwrite(f"{path}/pred_{i}.tif", pred[i, :, :])
    cv2.imwrite(f"{path}/X_{i}.tif", X[i, :, :])
    cv2.imwrite(f"{path}/y_{i}.tif", y[i, :, :])

