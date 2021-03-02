from keras.models import load_model
from data_generator import generator
import cv2
gen = generator()


model = load_model("C:/Users/tim.iles/noise_model_noise4.h5")
path = "C:/Users/tim.iles/Documents/Projects/sar_speckle_filtering/data/s1/ROIs1158_spring/s1_106/"

X, y = next(gen)
pred = model.predict(X)

pred = pred.reshape((8, 128, 128))
X = X.reshape((8, 128, 128))
y = y.reshape((8, 128, 128))


for i in range(32):
    cv2.imwrite(f"{path}/pred_{i}.tif", pred[i, :, :])
    cv2.imwrite(f"{path}/X_{i}.tif", X[i, :, :])
    cv2.imwrite(f"{path}/y_{i}.tif", y[i, :, :])

