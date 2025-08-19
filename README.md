<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

#  nnUNet Sample Application

This app used pre-trained nnUNet models to do both interactive and automated segmentation over 3D heart images. 

### How To Use the App

```bash
# Required packages
monai == 1.5.0
nnunetv2 == 2.0

# Download the App (skip this if you have already downloaded the app or using github repository (dev mode))

# Start MONAI Label Server with the model
monailabel start_server --app nnunet_heart --studies workspace/images --conf models nnunet/all

# Tutorial using the app as a 3D slicer extension
example using radiology app: 
https://github.com/Project-MONAI/tutorials/blob/main/monailabel/monailabel_HelloWorld_radiology_3dslicer.ipynb

# Pre-trained model preparation
1. Create a directory in the home directory: nnunet_heart/model
2. Prepare the model:
    The example model link:
    For a typical nnUNet, the model directory will be nnUNetv2/nnUNet_results/Dataset101_modality/NetTrainer__nnUNetPlans__3d_fullres
3. Set up the model directory by either using log: - model dir
    or add the pathway to lib/config/nnunet.py
4. Other parameters: 1. use folds. 2. model configs (2D or 3D). 3. target spacing (1,1,1). 4. ROI

# Post-transforms
Similar with monailabel workflow, restore the labels in the /data/label/final

