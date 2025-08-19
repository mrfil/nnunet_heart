import logging
import os
from typing import Any, Dict, Optional

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from lib.infers.nnunet_segmentation import NNUNetSegmentation 

logger = logging.getLogger(__name__)

class NNUNet(TaskConfig):
    """
    TaskConfig that wires a pre-trained nnUNet (v2) model directory
    into a MONAI Label inferer named `nnunet_model`.
    """

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any):
        super().init(name, model_dir, conf, planner)

        # REQUIRED: Path to nnUNet training output directory that contains fold_0..fold_4
        self.nnunet_model_path = conf.get(
            "nnunet_model_path",
            "/home/jriwei2/apps/nnunet_heart/model/nnunet/Dataset102_HeartCT/nnUNetTrainer__nnUNetPlans__3d_fullres",
        )
        # Optional: use_folds; default to all 5 folds
        self.use_folds = tuple(
            int(f) for f in conf.get("use_folds", "0,1,2,3,4").split(",") if f.strip() != ""
        )
        # For test
        self.use_folds = (0, 1)
        # Optional metadata for logs / UI
        self.model_config = conf.get("nnunet_config", "3d_fullres")

        # Labels
        self.labels = {
            "background": 0,
            "heart": 1,
        }

        # Planner hints if you want to override (used only for logging here)
        self.target_spacing = tuple(
            float(x) for x in conf.get("target_spacing", "1.0,1.0,1.0").split(",")
        )
        self.spatial_size = tuple(
            int(x) for x in conf.get("spatial_size", "96,96,96").split(",")
        )

        # Sanity/Logging
        if not os.path.isdir(self.nnunet_model_path):
            logger.warning(f"nnUNet model dir not found: {self.nnunet_model_path}")
        else:
            logger.info(f" Using nnUNet dir: {self.nnunet_model_path}")

        logger.info("nnUNet Configuration Summary:")
        logger.info(f"  - Model Dir:     {self.nnunet_model_path}")
        logger.info(f"  - Use Folds:     {self.use_folds}")
        logger.info(f"  - Model Config:  {self.model_config}")
        logger.info(f"  - TargetSpacing: {self.target_spacing}")
        logger.info(f"  - ROI Size:      {self.spatial_size}")

    def infer(self) -> Dict[str, InferTask]:
        """
        Create the entry point of MONAI Label.
        """
        task: InferTask = NNUNetSegmentation(
            path=self.nnunet_model_path,
            target_spacing=self.target_spacing,
            labels=self.labels,
            model_config=self.model_config,
            use_folds=self.use_folds,
            spatial_size=self.spatial_size,
            description=f"nnUNet {self.model_config} segmentation",
        )
        return {"nnunet_model": task}

    # Keep these as stubs (will implement in the future)
    def trainer(self) -> Optional[TrainTask]:
        return None

    def strategy(self) -> Optional[Strategy]:
        return None

    def scoring_method(self) -> Optional[ScoringMethod]:
        return None
