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

### Supported Viewers
The nnUNet Sample Application supports the following viewers:

- [3D Slicer](../../plugins/slicer/)
- [OHIF](../../plugins/ohif/)

For more information on each of the viewers, see the [plugin extension folder](../../plugins/) for the given viewer.

### Pretrained Models
The following are the models which are currently added into the App:

| Name | Description |
|------|-------------|
| CT heart nnUNet 3D full-res model |

### How To Use the App
The following commands are examples of how to start the nnUNet Sample Application.  Make sure when you're running the command that you use the correct app and studies path for your system.


```bash
# Download the App (skip this if you have already downloaded the app or using github repository (dev mode))

# Start MONAI Label Server with the model
monailabel start_server --app nnunet_heart --studies workspace/images --conf models nnunet/all

