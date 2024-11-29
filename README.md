# HiWet-DBNet
HiWet-DBNet: A Multi-Source Feature Fusion Network for Wetland Mapping

1. Environment Requirements
Python version: 3.9
PyTorch version: 2.0.0
Other required libraries: Refer to requirements.txt or install dependencies as needed.

2. Data Requirements
Sentinel-1 Radar Imagery
Specifications:
Temporal continuity: The imagery should cover multiple time periods to capture dynamic changes.
Feature bands: Includes VV, VH, VV/VH, and the Normalized Difference Polarization Index (NDPI).
Spatial resolution: 10 meters.
Sentinel-2 Optical Imagery
Specifications:
Feature bands: A total of 12 bands, including B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, the Normalized Difference Vegetation Index (NDVI), and the Normalized Difference Water Index (NDWI).
Spatial resolution: 10 meters.
Sample Data
Generation: Samples are cropped from imagery based on predefined sample points, with each sample being a .tif file of size 15×15 pixels.
Class labels: Seven classes are included: Mudflat, Zizania, Triarrhena-Phragmites, Sand, Submerged, Carex-Phalaris, Water.
File naming: Sample files are named according to their corresponding class label for ease of classification and management.

3. File Descriptions
(1) Training-Related Files
Train_.py: Loads sample data and trains the model.
Train_config.py: Contains parameter settings for the training process, such as learning rate and optimizer configurations.
(2) Testing-Related Files
Decoder_.py: Loads target imagery and performs model inference.
Decoder_config.py: Contains parameter settings for the testing process, such as input and output paths.
(3) Other Utility Files
focal_loss.py: Defines the loss function used to compute branch-specific losses during training.
user_dataset.py: A custom dataset class for loading .tif sample data, supporting data augmentation and batch processing.

Notes
Ensure that input data formats and naming conventions adhere to the requirements outlined above.
It is recommended to use a virtual environment for installing dependencies and ensure that data paths are correctly configured in the respective files.
For more details, refer to the code comments or contact the project contributors.
