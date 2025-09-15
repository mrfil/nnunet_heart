import os, json
from pathlib import Path
from typing import Optional, Any

from monailabel.interfaces.tasks.train import TrainTask
from monai.apps.nnunet.nnunetv2_runner import nnUNetV2Runner
from lib.utils.nnunet_utils import _strip_nifti_suffix

from lib.utils.nnunet_utils import (
    make_tempdir,
    ensure_dir,
    resolve_datastore,
    build_minimal_dataset_from_studies,
)

class NNUNetTrainTask(TrainTask):
    def __init__(self, conf=None):
        super().__init__(conf or {})
        self.conf = conf or {}
        self.description = "Train nnUNet v2 on labeled datastore cases"

        self.dataset_name_or_id = str(self.conf.get("nnunet_dataset_id", "102"))
        self.modality = self.conf.get("nnunet_modality", "CT")
        self.plans = self.conf.get("nnunet_plans", "nnUNetPlans")
        self.trainer_class = self.conf.get("nnunet_trainer", "nnUNetTrainer")
        self.nnunet_config_name = self.conf.get("nnunet_config", "3d_fullres")
        self.output_dir = self.conf.get("nnunet_output_dir")
        self.use_folds = [int(x) for x in str(self.conf.get("use_folds", "0")).split(",") if x != ""]

        self._tmp_root: Optional[Path] = None
        self._work_dir: Optional[Path] = None

    def config(self):
        return {
            "nnunet_dataset_id": self.dataset_name_or_id,
            "nnunet_modality": self.modality,
            "nnunet_plans": self.plans,
            "nnunet_trainer": self.trainer_class,
            "nnunet_config": self.nnunet_config_name,
            "use_folds": ",".join(map(str, self.use_folds)) if self.use_folds else "0",
            "nnunet_output_dir": self.output_dir or "(temporary work dir)",
        }

    def __call__(self, request: dict, datastore: Optional[Any] = None):
        ds = resolve_datastore(self, request, datastore)
        if hasattr(self, "init") and self.init(request) is False:
            return {"message": "init() returned False; training not started"}
        try:
            return self.run(request, ds)
        finally:
            if hasattr(self, "finalize"):
                self.finalize()

    def init(self, request):
        return True

    def run(self, request: dict, datastore: Any = None):
        # Build minimal Dataset under temp directory
        self._tmp_root = make_tempdir(prefix="nnunet_ds_")

        studies_root = Path(self.conf.get("studies") or request.get("studies", "")).expanduser().resolve()
        if not studies_root.exists():
            raise RuntimeError(f"studies path not found: {studies_root}")

        use_labels = (request.get("use_labels") or "final").lower()
        debug = bool(request.get("debug", False))

        dataset_info = build_minimal_dataset_from_studies(
            studies_root=studies_root,
            dataset_id=self.dataset_name_or_id,
            modality=self.modality,
            tmp_root=self._tmp_root,
            use_labels=use_labels,
            debug=debug,
        )

        data_root = Path(dataset_info["data_root"])
        dataset_name = data_root.name
        tmp_root = data_root.parent

        # Define nnU-Net directories
        nnunet_raw = str(tmp_root)
        nnunet_preprocessed = str(make_tempdir("nnunet_preprocessed_"))
        nnunet_results = str(make_tempdir("nnunet_results_"))

        # Set environment variables for nnUNet
        os.environ["nnUNet_raw"] = nnunet_raw
        os.environ["nnUNet_preprocessed"] = nnunet_preprocessed  
        os.environ["nnUNet_results"] = nnunet_results

        # Create nnUNet runner
        runner = nnUNetV2Runner(
            input_config={
                "datalist": str(data_root / "dataset.json"),
                "dataroot": nnunet_raw,
                "modality": self.modality,
                "dataset_name_or_id": self.dataset_name_or_id,
                "nnunet_raw": nnunet_raw,
                "nnunet_preprocessed": nnunet_preprocessed,
                "nnunet_results": nnunet_results,
            },
            trainer_class_name=self.trainer_class,
            work_dir=make_tempdir("nnunet_work_"),
            export_validation_probabilities=False,
        )

        # Plan and preprocess
        runner.plan_and_process(
            verify_dataset_integrity=bool(request.get("verify_dataset_integrity", False)),
            overwrite_plans_name=self.plans,
            c=(self.nnunet_config_name,),
            n_proc=(8,),
            verbose=debug,
        )

        # Find preprocessed data directory
        pp_root = Path(nnunet_preprocessed)
        dataset_name = Path(dataset_info["data_root"]).name
        ds_num = dataset_name.replace("Dataset", "")

        # Search for dataset directory
        candidate_dirs = [pp_root / dataset_name, pp_root / ds_num]
        ds_dir_for_gt = None
        plans_dir = None

        for cand in candidate_dirs:
            if cand.exists():
                hit = next((p for p in cand.rglob(f"{self.plans}_{self.nnunet_config_name}") if p.is_dir()), None)
                if hit:
                    plans_dir = hit
                    ds_dir_for_gt = cand
                    break

        if plans_dir is None:
            # Last resort: search entire preprocessed root
            hit = next((p for p in pp_root.rglob(f"{self.plans}_{self.nnunet_config_name}") if p.is_dir()), None)
            if hit:
                plans_dir = hit
                ds_dir_for_gt = hit.parent

        if plans_dir is None:
            raise RuntimeError(f"Could not locate plans directory '{self.plans}_{self.nnunet_config_name}' under {pp_root}")

        gt_dir = ds_dir_for_gt / "gt_segmentations"

        # Collect preprocessed case IDs
        def _preproc_id(p: Path) -> str:
            name = p.name
            # Skip segmentation files
            if name.endswith("_seg.b2nd"):
                return None
            if name.endswith(".b2nd"):
                name = name[:-len(".b2nd")]
            elif name.endswith(".npz"):
                name = name[:-len(".npz")]
            if name.endswith("_0000"):
                name = name[:-5]
            return name

        preproc_files = list(plans_dir.rglob("*.npz")) + list(plans_dir.rglob("*.b2nd"))
        data_ids = sorted({_preproc_id(p) for p in preproc_files if _preproc_id(p) is not None})

        # Get GT case IDs
        gt_files = list(gt_dir.glob("*.nii.gz")) + list(gt_dir.glob("*.nii"))
        gt_ids = sorted({_strip_nifti_suffix(p.name) for p in gt_files})

        # Find matching IDs
        avail_ids = [i for i in data_ids if i in set(gt_ids)]

        if not avail_ids:
            raise RuntimeError(
                f"No matching case IDs found between preprocessed data ({len(data_ids)} cases) "
                f"and ground truth labels ({len(gt_files)} cases)"
            )

        # Create train/validation splits
        splits_file = ds_dir_for_gt / "splits_final.json"
        if len(avail_ids) < 5 or not splits_file.exists():
            val = avail_ids[:1] if len(avail_ids) > 1 else []
            train = avail_ids[1:] if len(avail_ids) > 1 else avail_ids
            splits = [{"train": train, "val": val}]
            splits_file.write_text(json.dumps(splits, indent=2))

        # Verify splits reference valid data
        missing = [i for i in (train + val) if i not in set(data_ids)]
        if missing:
            raise RuntimeError(f"Split references missing preprocessed files: {missing}")

        # Train specified folds
        folds = self.use_folds or [0]
        for f in folds:
            runner.train_single_model(
                config=self.nnunet_config_name,
                fold=int(f),
                gpu_id=0,
                p=self.plans,
            )

        # Return training summary
        return {
            "dataset": dataset_name,
            "num_cases": dataset_info["num_cases"],
            "preprocessed_dir": str(Path(nnunet_preprocessed) / dataset_name),
            "results_dir": nnunet_results,
            "folds_trained": list(map(int, folds)),
            "config": self.nnunet_config_name,
            "trainer": self.trainer_class,
            "plans": self.plans,
        }

    def finalize(self):
        pass