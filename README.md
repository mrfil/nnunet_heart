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

# MONAI Label – nnUNet Heart Segmentation App

This repository provides a MONAI Label app that integrates customized **nnUNet v2** models into the MONAI Label framework. It enables interactive and batch segmentation of medical data directly from **3D Slicer** or via the MONAI Label REST API.

---

##  Repository Structure

    └── main.py # Entry point for starting MONAI Label app

    └── lib # Entry point for starting MONAI Label app

        └── config

            └── nnunet.py # TaskConfig binding nnUNet into MONAI Label

        └── infer

            └── nnunet_segmentation.py # nnUNet wrapper + MONAI InferTask
        ...

    └── model/ # Directory for storing trained nnUNet models

        └── nnunet/...

---

##  Requirements
- Python 3.9+  
- [MONAI Label](https://github.com/Project-MONAI/MONAILabel) ≥ 0.7  
- [nnUNet v2](https://github.com/MIC-DKFZ/nnUNet) (installed and trained models available)  
- 3D Slicer (with [MONAI Label extension](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer))  

---

##  Running the App

### 1. Start the MONAI Label Server
Example:

```bash

monailabel start_server --app nnunet_heart --studies ~/Dataset --conf models nnunet
```

This starts the MONAI Label server with your nnUNet heart model.

### 2. Pre-trained model preparation

- Create a directory in the home directory: nnunet_heart/model

- Prepare the model:

    - The example model link:
    
    - For a typical nnUNet, the model directory will be nnUNetv2/nnUNet_results/Dataset101_modality/NetTrainer__nnUNetPlans__3d_fullres

### 3. Connect via 3D Slicer

Monailabel extension tutorial in 3D slicer: [Quickstart](https://docs.monai.io/projects/label/en/latest/quickstart.html)

Open 3D Slicer.

Install and enable the MONAI Label extension.

Connect to your running server (http://127.0.0.1:8000).

Click Next Sample 

Use the Auto-Segmentation → nnunet_model option and click Run to generate the segmentation.

### 4. Features

- Auto-Segmentation with nnUNet v2 (nnunet_segmentation.py handles preprocessing, inference, and postprocessing).

- Custom Configs: specify target spacing, spatial ROI, folds, and model path in nnunet.py.
  
- Segmented labels are saved back into the studies folder under Dataset/labels/final.

- Extendable:
    - Add more models by writing new TaskConfig classes in lib/configs.

    - Batch Inference: run segmentation on all studies in the datastore.
  
    - The app can be extended with active learning strategies and scoring methods if needed.
  




