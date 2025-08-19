"""
Utility functions for nnUNet integration with MONAI Label
"""

import os
import shutil
import glob
from typing import List


def prepare_nnunet_input(input_file_path: str, output_dir: str, case_name: str = "case_001") -> str:
    """
    Prepare a single input file for nnUNet processing by renaming it to nnUNet format
    
    Args:
        input_file_path: Path to the input NIfTI file
        output_dir: Directory to place the prepared file
        case_name: Name for the case (default: case_001)
    
    Returns:
        Path to the directory containing the prepared file
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine file extension
    if input_file_path.endswith('.nii.gz'):
        extension = '.nii.gz'
    elif input_file_path.endswith('.nii'):
        extension = '.nii'
    else:
        raise ValueError(f"Unsupported file format: {input_file_path}")
    
    # Create nnUNet-style filename
    nnunet_filename = f"{case_name}_0000{extension}"
    output_file_path = os.path.join(output_dir, nnunet_filename)
    
    # Copy file with new name
    shutil.copy2(input_file_path, output_file_path)
    
    print(f"Prepared nnUNet input: {input_file_path} → {output_file_path}")
    
    return output_dir


def cleanup_temp_files(temp_dirs: List[str]) -> None:
    """
    Clean up temporary directories
    
    Args:
        temp_dirs: List of temporary directory paths to clean up
    """
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {temp_dir}: {e}")
