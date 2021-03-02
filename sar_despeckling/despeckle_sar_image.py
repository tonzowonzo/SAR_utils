import rasterio
import numpy as np
from keras.models import load_model
import cv2
from keras.utils.generic_utils import get_custom_objects
from model import custom_loss


import os
path = "D:/sar/test3/"
model_cnn = load_model("C:/Users/tim.iles/noise_model_noise_synthetic_sv_loss.h5",
                       custom_objects={"total_variation_loss": custom_loss})

if path.endswith("test3/"):
    scale_factor = 45
else:
    scale_factor = 255
img_size = 256
border_size = 8
for i in [i for i in os.listdir(path) if i.endswith("tif")]:
    # Load in given image from image folder path.
    with rasterio.open(f"{path}{i}") as src:
        img = src.read()
        kwargs = src.profile

    # Take just the first band of the image if there's more than 1.
    img_name = i.split(".")[0]
    img = img[0, :, :]

    # Scale the image by a scale factor.
    img = (img / scale_factor).astype(np.float32)

    # Scale all values between 0 and 0.5 then scale them between 0 and 1.
    img[img > 0.5] = 0.5
    img /= 0.5
    out_img = np.zeros(img.shape).astype(np.float32)

    # Iterate over the image and fill it.
    for i in range(20, img.shape[0] - img_size, img_size - (border_size*2)):
        print(f"{i}/{img.shape[0]} complete.")
        for j in range(20, img.shape[1] - img_size, img_size - (border_size*2)):
            # Change j or i so that almost the whole image will be despeckled.
            if j + (img_size - (border_size*2)) > img.shape[1]:
                j = j - img_size
            if i + (img_size - (border_size*2)) > img.shape[0]:
                i = i - img_size
            # Reshape the inputs for input into the model.
            X = img[i:i+img_size, j:j+img_size]
            X = X.reshape((1, img_size, img_size, 1))
            pred = model_cnn.predict(X)
            pred = pred.reshape((img_size, img_size))
            pred = pred[border_size:-border_size, border_size:-border_size]

            # Place the predicted image into the correct area in the entire image.
            out_img[i+border_size:i+img_size-border_size, j+border_size:j+img_size-border_size] = pred

    # Prepare the image for saving.
    out_img = out_img.reshape(img.shape)
    kwargs["dtype"] = "float32"
    kwargs["driver"] = "GTiff"

    # Save the image to the output path.
    with rasterio.open(f"{path}{img_name}_filtered.tif", "w", **kwargs) as dst:
        out_img = (out_img * 255).astype(np.float32)
        dst.write(out_img, 1)

    with rasterio.open(f"{path}{img_name}_X.tif", "w", **kwargs) as dst:
        img = (img * 255).astype(np.float32)
        dst.write(img, 1)