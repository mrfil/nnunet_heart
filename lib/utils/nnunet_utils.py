"""
Utility functions for nnUNet integration with MONAI Label
"""

import re, os, json, shutil, tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

def _strip_nifti_suffix(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else (name[:-4] if name.endswith(".nii") else name)

def _find_first(root: Path, rel_patterns: list[str]) -> Optional[Path]:
    """
    Glob relative patterns from a given root directory.
    """
    for pat in rel_patterns:
        matches = sorted(root.glob(pat))
        if matches:
            return matches[0]
    return None

def _find_image_for_id(studies_root: Path, iid: str) -> Optional[Path]:
    """
    Search for an image file matching the given id under studies_root.
    Checks both studies_root and studies_root/images.
    """
    roots = [studies_root, studies_root / "images"]
    for base in roots:
        candidates = [
            f"{iid}.nii.gz",
            f"{iid}.nii",
            f"{iid}*.nii*",
        ]
        hit = _find_first(base, candidates)
        if hit and hit.exists():
            return hit
    return None

def _normalize_id(iid: str) -> str:
    # keep alnum, dash, underscore; replace everything else with underscore
    return re.sub(r'[^A-Za-z0-9_-]+', '_', iid)

def build_minimal_dataset_from_studies(
    studies_root: Path,
    dataset_id: str,
    modality: str,
    tmp_root: Path,
    use_labels: str = "final",  # "final" or "original"
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Build Dataset{ID} for nnU-Net using ONLY filesystem paths:
      images:   found under studies_root (and optional studies_root/images)
      labels:   under studies_root/labels/<use_labels>  (exclusive choice)
    """
    if use_labels not in {"final", "original"}:
        raise ValueError("use_labels must be 'final' or 'original'")

    label_dir = studies_root / "labels" / use_labels
    if not label_dir.exists():
        raise RuntimeError(f"Label folder not found: {label_dir}")

    data_root = tmp_root / f"Dataset{dataset_id}"
    imagesTr = ensure_dir(data_root / "imagesTr")
    labelsTr = ensure_dir(data_root / "labelsTr")

    kept = 0
    label_files = sorted(label_dir.glob("*.nii*"))

    if debug:
        print(f"[DBG] use_labels={use_labels}  studies_root={studies_root}")
        print(f"[DBG] found {len(label_files)} label(s) in {label_dir}")

    for lbl in label_files:
        raw_iid = _strip_nifti_suffix(lbl.name)
        img_p = _find_image_for_id(studies_root, raw_iid)
        if debug:
            print(f"[DBG] raw_iid={raw_iid}: label={lbl}  image_match={img_p}")

        if not img_p:
            # also try swapping '.' and '_' as a fallback
            alt1 = raw_iid.replace('.', '_')
            alt2 = raw_iid.replace('_', '.')
            for cand in (alt1, alt2):
                img_p = _find_image_for_id(studies_root, cand)
                if img_p:
                    if debug: print(f"[DBG] recovered image via alt id: {cand} -> {img_p}")
                    break
        if not img_p:
            continue

        iid = _normalize_id(raw_iid)

        dst_img = imagesTr / f"{iid}_0000.nii.gz"
        dst_lbl = labelsTr / f"{iid}.nii.gz"
        symlink_or_copy(img_p, dst_img)
        symlink_or_copy(lbl, dst_lbl)
        kept += 1        

    if kept == 0:
        raise RuntimeError(
            f"No image/label pairs were formed. "
            f"Checked labels in {label_dir} and images under {studies_root} (and images/)."
        )

    dataset_json = {
        "name": f"Dataset{dataset_id}",
        "channel_names": {"0": modality},          
        "labels": {"background": 0, "heart": 1},   
        "numTraining": kept,
        "file_ending": ".nii.gz",
        "training": [
            {"image": f"./imagesTr/{p.stem}_0000.nii.gz", "label": f"./labelsTr/{p.name}"}
            for p in sorted(labelsTr.glob("*.nii*"))
        ],
        "test": [],
    }
    (data_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    return {
        "data_root": data_root,
        "imagesTr": imagesTr,
        "labelsTr": labelsTr,
        "num_cases": kept,
    }

# ---------- filesystem helpers ----------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def cleanup_temp_files(temp_dirs: List[str]) -> None:
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {temp_dir}: {e}")

def make_tempdir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))

# ---------- datastore helpers ----------

def resolve_datastore(self_obj, request: dict, datastore_obj: Optional[Any]) -> Any:
    cand = datastore_obj if datastore_obj is not None else getattr(self_obj, "datastore", None)
    if callable(cand):
        try:
            return cand()
        except TypeError:
            return cand
    if cand is not None:
        return cand
    ds = request.get("datastore")
    if ds is None:
        raise RuntimeError("Datastore not available to training task")
    return ds

def select_label_tag(all_tags: Iterable[str], requested_tag: Optional[str]) -> Optional[str]:
    if requested_tag:
        return requested_tag if requested_tag in all_tags else None
    tags = list(all_tags) or []
    if not tags:
        return None
    if "final" in tags:
        return "final"
    if "original" in tags:
        return "original"
    return sorted(tags)[-1]

# ---------- dataset building ----------

def iter_labeled_pairs(datastore, requested_tag: Optional[str]) -> Iterable[Tuple[str, str, str]]:
    """
    Yields (case_id, image_path, label_path) for all labeled cases that have usable files.
    """
    labeled_ids = datastore.get_labeled_images()
    for iid in labeled_ids:
        img_uri = datastore.get_image_uri(iid)
        try:
            tags = list(datastore.get_label_tags(iid)) or []
        except Exception:
            tags = []
        tag = select_label_tag(tags, requested_tag)
        if not tag:
            continue
        lbl_uri = datastore.get_label_uri(iid, tag)
        if not img_uri or not lbl_uri:
            continue
        yield iid, img_uri, lbl_uri

# ---------- manual-paths dataset building (no datastore) ----------

def prepare_nnunet_input(input_file_path: str, output_dir: str, case_name: str = "case_001") -> str:
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    ensure_dir(output_dir)
    if input_file_path.endswith(".nii.gz"):
        extension = ".nii.gz"
    elif input_file_path.endswith(".nii"):
        extension = ".nii"
    else:
        raise ValueError(f"Unsupported file format: {input_file_path}")
    nnunet_filename = f"{case_name}_0000{extension}"
    output_file_path = os.path.join(output_dir, nnunet_filename)
    shutil.copy2(input_file_path, output_file_path)
    print(f"Prepared nnUNet input: {input_file_path} → {output_file_path}")
    return output_dir
