# SAR_utils
A selection of functions for working with SAR data.

### sar_despeckling:

A collection of scripts to despeckle images, CNN based despeckling. Despeckling is based on the paper by Puyang Wang (https://arxiv.org/pdf/1706.00552.pdf). But is modified to not do residual learning and instead just learns to denoise without a skip connection. The data used to train was created using the paper by Dahbi et al. (https://ieee-dataport.org/open-access/virtual-sar-synthetic-dataset-deep-learning-based-speckle-noise-reduction-algorithms) which allows synthetic SAR speckle to be created on non-SAR imagery. Several open datasets were then used to create a "clean" and "speckled" dataset, including a dataset supplied by Dahbi et al. If using a GPU with cuda, the approx. time to despeckle a 6000*6000 image is 30 seconds.

Examples: ICEYE:
Speckled:
<img src=https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_speckled.PNG?raw=true>
Filtered:
<img src=https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_filtered.PNG?raw=true>

Speckled:
<img src=https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_speckled2.PNG?raw=true>
Filtered:
<img src=https://github.com/tonzowonzo/SAR_utils/blob/main/examples/iceye/iceye_filtered2.PNG?raw=true>


##### Despeckling with this model works better under several circumstances: 
1. The image hasn't been log scaled.
2. The image is scaled between 0 and 255 where 255 was initially a backscatter of 0.5.

### sar_processing:

A pipeline for end-to-end Sentinel 1 processing based on snappy.
