# SAR_utils
A selection of functions for working with SAR data.

### sar_despeckling:

A collection of scripts to despeckle images, CNN based despeckling. Despeckling is based on the paper by Puyang Wang (https://arxiv.org/pdf/1706.00552.pdf) but also makes use of dilations from a paper by Zhang et al. (https://arxiv.org/ftp/arxiv/papers/1709/1709.02898.pdf). But is modified to not do residual learning and instead just learns to denoise without a skip connection. The data used to train was created using the paper by Dahbi et al. (https://ieee-dataport.org/open-access/virtual-sar-synthetic-dataset-deep-learning-based-speckle-noise-reduction-algorithms) which allows synthetic SAR speckle to be created on non-SAR imagery. Several open datasets were then used to create a "clean" and "speckled" dataset, including a dataset supplied by Dahbi et al. If using a GPU with cuda, the approx. time to despeckle a 6000*6000 image is 30 seconds.

At some point I will add true implementations of papers that copy their architecture exactly, but for now it's a slightly modified combination of the two.

#### Testing examples
I will add some testing examples for both Sentinel-1, ICEYE and Capella as soon as possible.

#### requirements.txt
For now the requirements.txt is wrong, I will fix this as soon as I have a nice environment locally. More models will also be added soon, with the skip connections included for the residual noise calculations.

#### The three models so far are:
1. noise_model_noise_synthetic.h5 - Based on the "dilation_net" function in model.py and trained with mse loss. This is based on a hybrid approach from Zhang et al. and Wang et al.
2. noise_model_noise_synthetic_sv_loss.h5 - The same as the above but sv is used in conjunction with mse loss.
3. noise_model_synthetic_mse_loss_sar_drn.h5 - A recreation of the model from the Zhang et al. paper, uses skip connections and calculates the residual noise to despeckle the image. It uses mse as a loss function for training.

#### Future implementations:
1. https://www.mdpi.com/2072-4292/11/13/1532/htm - A U-Net based residual network quite different to the implementations above by Latarri et al.

Artificial speckle: The functions to create the artificial noise will be adapted to NumPy code at some point and added to this repo. Right now they aren't here but I will release them soon. With the artificial speckle function users will be able to add their own noise to aerial/satellite imagery to train for any resolution. Right now these denoisers work best with high res sar, but feasibly they would work better on lower res such as Sentinel 1 if trained on Sentinel 2 images with artificial speckle added. 

#### Examples: ICEYE:
##### noise_model_noise_synthetic.h5
Speckled:
<img src="https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_speckled.PNG?raw=true">
Filtered:
<img src="https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_filtered.PNG?raw=true">

Speckled:
<img src="https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_speckled2.PNG?raw=true">
Filtered:
<img src="https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_filtered2.PNG?raw=true">
##### noise_model_synthetic_mse_loss_sar_drn.h5
Speckled:
<img src="https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/sar_example_drn_speckled.PNG?raw=true">
Filtered:
<img src="https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/sar_example_drn_despeckled.PNG?raw=true">

##### Despeckling with this model works better under several circumstances: 
1. The image hasn't been log scaled.
2. The image is scaled between 0 and 255 where 255 was initially a backscatter of 0.5.

### sar_processing:

A pipeline for end-to-end Sentinel 1 processing based on snappy.
