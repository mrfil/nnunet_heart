# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile
import logging
from typing import Callable, Sequence

import numpy as np
import nibabel as nib
from pathlib import Path

import torch
import torch.nn as nn

from monai.inferers import Inferer, SimpleInferer
from monai.data.meta_tensor import MetaTensor
from monai.apps.nnunet.nnunetv2_runner import nnUNetV2Runner
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ToMetaTensord,
)
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored
from lib.utils.nnunet_utils import prepare_nnunet_input, cleanup_temp_files

logger = logging.getLogger(__name__)

class NNUNetWrapper(nn.Module):
    """
    Wrap nnUNet as a PyTorch module usable by MONAI Label.
    Uses nnUNetV2Runner.predict under the hood.
    """

    def __init__(
        self,
        model_path,
        model_config: str = "3d_fullres",
        use_folds=(0, 1, 2, 3, 4),
        input_spacing=(1.0, 1.0, 1.0), 
    ):
        super().__init__()
        self.model_path = model_path
        self.model_config = model_config
        self.use_folds = use_folds
        self.input_spacing = tuple(float(s) for s in input_spacing)
        self._runner = None
        self.temp_dirs = []

        self.nnunet_config = {
            "datalist": "dummy_datalist.json",
            "dataroot": "/tmp",
            "modality": "CT",
            "dataset_name_or_id": "999",
        }

    def _initialize_runner(self):
        if self._runner is None:
            self._runner = nnUNetV2Runner(
                input_config = self.nnunet_config,
                trainer_class_name = "nnUNetTrainer",
                work_dir = tempfile.mkdtemp(prefix="nnunet_work_"),
                export_validation_probabilities = False,
            )
            logger.info("nnUNetV2Runner initialized")

    def forward(self, x: torch.Tensor):
        """
        x: tensor [B, C, H, W, D]
        """
        self._initialize_runner()

        temp_input_dir = tempfile.mkdtemp(prefix="nnunet_patch_input_")
        temp_output_dir = tempfile.mkdtemp(prefix="nnunet_patch_output_")
        self.temp_dirs.extend([temp_input_dir, temp_output_dir])

        try:
            # convert to numpy [C, H, W, D] and drop batch
            x_in = x
            x_np = x.detach().cpu().numpy()
            if x_np.ndim == 5: x_np = x_np[0]        # [C,H,W,D]
            x_np = x_np[0] if x_np.shape[0] > 1 else x_np[0]  # [H,W,D]

            # get affine from MetaTensor if present
            aff = None
            if isinstance(x_in, MetaTensor) and getattr(x_in, "affine", None) is not None:
                aff = x_in.affine.detach().cpu().numpy()
            if aff is None:
                aff = np.eye(4)  # fallback

            # save temp nifti
            temp_nifti_path = os.path.join(temp_input_dir, "patch.nii.gz")
            nii_img = nib.Nifti1Image(x_np, affine=aff)
            nib.save(nii_img, temp_nifti_path)

            # rename to case_ijk_0000.nii.gz for nnU-Net
            prepared_dir = prepare_nnunet_input(temp_nifti_path, temp_input_dir + "_prepared")

            # ---- run nnU-Net prediction ----
            self._runner.predict(
                list_of_lists_or_source_folder=prepared_dir,
                output_folder=temp_output_dir,
                model_training_output_dir=self.model_path,
                use_folds=self.use_folds,
                save_probabilities=False,
                overwrite=True,
                verbose=False,
                gpu_id=0,
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_gpu=True,
            )

            # load result; force discrete labels
            outs = list(Path(temp_output_dir).glob("*.nii*"))
            if not outs:
                raise RuntimeError("nnUNet prediction failed - no output files")

            result_nii = nib.load(str(outs[0]))
            result_data = result_nii.get_fdata().astype(np.uint8)

            # convert back to tensor format expected by MONAI
            result_tensor = torch.from_numpy(result_data)  # uint8

            # add batch dimension back -> [1, C, H, W, D]
            if result_tensor.ndim == 3:
                result_tensor = result_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D]
            elif result_tensor.ndim == 4:
                result_tensor = result_tensor.unsqueeze(0)  # [1,C,H,W,D]

            return result_tensor.to(x.device)

        except Exception as e:
            logger.error(f"nnUNet prediction failed: {e}")
            return torch.zeros_like(x)  # safe fallback

        finally:
            cleanup_temp_files([temp_input_dir, temp_output_dir])


class NNUNetSegmentation(BasicInferTask):
    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained nnUNet model for volumetric (3D) Segmentation",
        model_config="3d_fullres",
        use_folds=(0, 1, 2, 3, 4),
        spatial_size=(96, 96, 96),
        **kwargs,
    ):
        self.model_dir = path

        nnunet_network = NNUNetWrapper(
            model_path=self.model_dir,
            model_config=model_config,
            use_folds=use_folds,
            input_spacing=target_spacing,
        )

        super().__init__(
            path="",
            network=nnunet_network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )

        self.target_spacing = target_spacing
        self.model_config = model_config
        self.use_folds = use_folds
        self.roi_size = spatial_size
        logger.info(f"nnUNet model initialized with ROI size: {self.roi_size}")

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            ToMetaTensord(keys="image"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
        ]

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Restored(keys="pred", ref_image="image"),
        ]

    def run_invert_transforms(self, data, pre_transforms, inverse_transforms):
        # Skip global auto-invert; we explicitly restore with Restored()
        return data
