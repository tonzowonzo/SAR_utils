import rasterio
import numpy as np
from keras.models import load_model
import cv2

with rasterio.open("C:/Users/tim.iles/Documents/Projects/sar_speckle_filtering/data/iceye/37992.tif") as src:
    img = src.read()
    kwargs = src.profile

img = img.reshape(img.shape[1:])
img = (img / 255).astype(np.float32)
img = img[4000:7000, 4000:7000]
out_img = np.zeros(img.shape).astype(np.float32)

model = load_model("C:/Users/tim.iles/noise_model_noise2.h5")

for i in range(0, img.shape[0] - 64, 64):
    print(i)
    for j in range(0, img.shape[1]-64, 64):
        X = img[i:i+64, j:j+64]
        X = X.reshape((1, 64, 64, 1))
        pred = model.predict(X)
        pred = pred.reshape((64, 64))

        out_img[i:i+64, j:j+64] = pred


out_img = out_img.astype(np.float32)
kwargs["dtype"] = "float32"

with rasterio.open("C:/Users/tim.iles/Documents/Projects/sar_speckle_filtering/test_iceye.tif", "w", **kwargs) as dst:
    dst.write(out_img, 1)