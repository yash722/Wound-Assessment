# Wound Segmentation Prediction
This repository contains a script and notebook for performing wound segmentation on images using a pre-trained Deeplabv3 model loaded from a Google Cloud Storage (GCS) bucket. The script segments the image using overlapping patches and overlays the segmentation mask on the original image. The Binary Mask and the Overlay Image are presented. 

## Requirements

Ensure you have the following dependencies installed:

- `gcsfs`
- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `matplotlib`

## Notebooks
The two notebooks presented show how the model was trained and tested.
