from snappy import ProductIO, HashMap, GPF
import os


def apply_orbit_file(product):
    parameters = HashMap()
    parameters.put("Apply-Orbit-File", True)
    operator_name = "Apply-Orbit-File"
    target_product = GPF.createProduct(operator_name, parameters, product)
    return target_product


def thermal_noise_removal(product, remove_thermal_noise=True):
    parameters = HashMap()
    parameters.put("removeThermalNoise", remove_thermal_noise)

    operator_name = "ThermalNoiseRemoval"
    target_product = GPF.createProduct(operator_name, parameters, product)
    return target_product


def border_noise_remove(product, border_margin_limit=500, thresold=0.5):
    parameters = HashMap()
    parameters.put("borderMarginLimit", border_margin_limit)
    parameters.put("Threshold", thresold)

    operator_name = "Remove-GRD-Border-Noise"
    target_product = GPF.createProduct(operator_name, parameters, product)
    return target_product


def calibration(product, output_type="sigma0", polarization="both"):
    """
    Passes the snap product from border_noise_removal and calibrates it to either sigma0, beta0 or gamma0 depending on
    the users selection.

    :param product: The product from ProductIO.
    :param output_type: The type of calibration to undertake, choose from ["sigma0", "beta0", "gamma0"].
    :param polarization: The polarization of the images being passed to calibration. Choose from:
    ["both", "vv", "vh"].
    :return: The calibrated product.
    """
    parameters = HashMap()
    polarization = polarization.lower()
    output_type = output_type.lower()

    if polarization not in ["both", "vv", "vh"]:
        raise ValueError("The polarization chosen is not supported, choose from 'both', 'vv' or 'vh'")

    if output_type not in ["sigma0", "beta0", "gamma0"]:
        raise ValueError("The output type chosen isn't possible, choose from 'sigma0', 'beta0' or 'gamma0'")

    # Choose calibration to undertake.
    if output_type == "sigma0":
        parameters.put("outputSigmaBand", True)
    elif output_type == "gamma0":
        parameters.put("outputGammaBand", True)
    else:
        parameters.put("outputBetaBand", True)

    # Choose polarizations to use.
    if polarization == "both":
        parameters.put("sourceBands", "Intensity_VH,Intensity_VV")
    elif polarization == "vh":
        parameters.put("sourceBands", "Intensity_VH")
    elif polarization == "vv":
        parameters.put("sourceBands", "Intensity_VV")

    operator_name = "Calibration"
    target_product = GPF.createProduct(operator_name, parameters, product)
    return target_product


def terrain_correction(product):
    parameters = HashMap()
    parameters.put("demName", "ACE30")
    parameters.put("imgResamplingMethod", "BICUBIC_INTERPOLATION")
    parameters.put("saveProjectedLocalIncidenceAngle", True)
    parameters.put('saveSelectedSourceBand', True)

    operator_name = "Terrain-Correction"
    target_product = GPF.createProduct(operator_name, parameters, product)
    return target_product


def speckle_filtering(product, filter_type="lee", filter_size=5):
    """
    Attempts to filter the speckle from the image, without too much loss of edges.

    :param product: The input product processed to calibration or terrain correction level.
    :param filter_type: Choose from lee, lee_sigma, refined_lee, median and cnn. It should be noted that cnn is a tool
    external from the SNAP toolbox and therefore if CNN is chosen here, we move into numpy arrays.
    :param filter_size: An odd number is required, the base size is 5 but it's recommended to change this for some
    settings.
    :return:
    """
    if filter_size % 2 == 0:
        raise ValueError("The filter size must be an odd value.")
    if filter_type not in ["lee", "lee_sigma", "refined_lee", "median", "cnn"]:
        raise ValueError("The filter type chosen is not valid in this pipeline, choose from 'lee', 'lee_sigma',"
                         " 'refine_lee', 'median', 'cnn'")

    parameters = HashMap()
    parameters.put("filterSizeX", filter_size)
    parameters.put("filterSizeY", filter_size)

    # Apply the chosen filter.
    if filter_type == "lee":
        parameters.put("filter", "Lee")
    elif filter_type == "lee_sigma":
        parameters.put("filter", "LeeSigma")
    elif filter_type == "refine_lee":
        parameters.put("filter", "RefineLee")
    elif filter_type == "median":
        parameters.put("filter", "Median")
    else:
        print("This is not ready yet.")

    operator_name = "Speckle-Filter"
    target_product = GPF.createProduct(operator_name, parameters, product)
    return target_product


def run_sar_pipeline(path_to_product: str, out_path: str, filter_image=False, remove_thermal=True,
                     border_margin_limit=500, threshold=0.5, output_type="sigma0", polarization="both",
                     filter_type="lee", filter_size=5):
    """
    Runs the sar pipeline to process a single input.

    :param path_to_product:
    :param out_path:
    :param filter_image:
    :param remove_thermal:
    :param border_margin_limit:
    :param threshold:
    :param output_type:
    :param polarization:
    :param filter_type:
    :param filter_size:
    :return:
    """
    # Load
    s1_img = ProductIO.readProduct(path_to_product)

    # Process the image.
    # Apply orbit file.
    s1_img = apply_orbit_file(s1_img)
    # Remove thermal noise.
    s1_img = thermal_noise_removal(s1_img, remove_thermal)
    # Remove border noise.
    s1_img = border_noise_remove(s1_img, border_margin_limit, threshold)
    # Calibrate.
    s1_img = calibration(s1_img, output_type, polarization)
    # Terrain correction.
    s1_img = terrain_correction(s1_img)

    # Filtering.
    if filter_image:
        speckle_filtering(s1_img, filter_type, filter_size)

    # Write the imagery.
    ProductIO.writeProduct(s1_img, out_path, "BEAM-DIMAP")


run_sar_pipeline("D:/sar/s1_denmark/S1B_IW_GRDH_1SDV_20190305T053954_20190305T054019_015215_01C76B_406C.SAFE",
                 "D:/sar/20190305T053954.dim")

if __name__ == '__main__':
    for product in os.listdir("D:/sar/s1_denmark"):
        out_name = f"{out_name.split('_')[4]}.dim"
        out_path = f"D:/sar/s1_denmark_processed/{out_name}"
        run_sar_pipeline(f"D:/sar/s1_denmark/{product}",
                         out_path)