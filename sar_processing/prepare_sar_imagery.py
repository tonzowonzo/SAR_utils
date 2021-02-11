import subprocess
import rasterio
import numpy as np


def change_to_utm_projection():
    pass


def rescale_data(image_path: str, out_path: str, out_bits=8):
    """
    Rescales the SAR data either to 16 bit uint or 8 bit uint. All values of backscatter above 1 are set to 1, and all
     values of backscatter below 0 are set to 0.

    :param image_path: The path to the image to scale.
    :type image_path: str
    :param out_path: The path for the output file.
    :type out_path: str
    :return: None, rescales the data inplace and saves it as a geotiff.
    """
    with rasterio.open(image_path) as src:
        img = src.read()
        kwargs = src.profile

    img[img > 0.5] = 0.5
    img[img < 0] = 0

    if out_bits == 8:
        img = (img * (255 * 2)).astype(np.uint8)
        kwargs["dtype"] = "uint8"

    elif out_bits == 16:
        img = (img * (65535 * 2)).astype(np.uint16)
        kwargs["dtype"] = "uint16"

    kwargs["driver"] = "GTiff"

    with rasterio.open(out_path, "w", **kwargs) as dst:
        dst.write(img)


def cut_s1_to_s2_grid(image_path: str):
    """

    :param image_path:
    :return:
    """
    pass


def create_timeseries_stack(image_paths_list: list):
    """


    :param image_paths_list:
    :return:
    """
    pass


rescale_data("D:/sar/20190305T053954.data/Sigma0_VH.img",
             "D:/sar/20190305T053954_scaled.tif")

