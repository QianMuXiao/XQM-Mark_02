# MG-RDFD

This repository contains the research implementation of **Memory-Guided Random-Direction Feature Disentanglement (MG-RDFD)** for multi-phase contrast-enhanced MRI translation.

The code includes the main encoder-decoder model, random-direction content/style sampling modules, segmentation-guided memory bank, paired MRI dataset loader, and training pipeline used in our study. Due to dataset access requirements, environment differences, and hardware-dependent training settings, this repository is intended as a reference implementation rather than a one-command reproduction package.

## Code Structure

```text
.
├── Models_v2.py              # Autoencoder-based translation backbone
├── Sample.py                 # Random-direction content/style sampling modules
├── Memory_pair_v7.py          # Segmentation-guided memory bank
├── mri_dataset_v4.py          # Paired multi-phase MRI slice dataset
├── loss_fn.py                 # Training losses
├── train_v5.py                # Distributed training script
└── lesion_patient_list.txt    # Patient-level lesion category list used for splitting
```

## Environment

The code was developed with PyTorch and common medical image processing libraries. A typical environment should include:

```text
python
torch
torchvision
monai-generative
SimpleITK
scipy
numpy
tqdm
tensorboardX
pytorch-msssim
lpips
xformers
```

The attention blocks can use xFormers/FlashAttention when available. If xFormers is not compatible with the local GPU, CUDA, or PyTorch version, disable the flash-attention path in the model configuration before training or inference.

## Dataset

This work uses the LLD-MMRI dataset for multi-phase liver lesion MRI analysis. The dataset is not redistributed in this repository. Please refer to the official dataset release/challenge page for access and usage terms:

- LLD-MMRI dataset: https://bit.ly/3IyYlgN

After preprocessing the original 3D volumes into paired 2D slices, the dataset loader expects the following structure:

```text
LLD-MMRI/
├── lesion_patient_list.txt
│
└── 2d_mri_body_dataset_mutil_phase_corp_body_v1/
    ├── pre/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── c_a/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── c_v/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── delay/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── totalseg/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── tumor/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    └── body/
        ├── MR-391135_10.nii.gz
        ├── MR-391135_11.nii.gz
        └── ...
```

Each slice is stored as a 2D `.nii.gz` file named as `{patient_id}_{slice_index}.nii.gz`. The same filename should exist in all required phase and mask folders so that the loader can pair slices by `(patient_id, slice_index)`.

The `data_path` argument should point to:

```text
LLD-MMRI/2d_mri_body_dataset_mutil_phase_corp_body_v1
```

The `lesion_patient_file` argument should point to:

```text
LLD-MMRI/lesion_patient_list.txt
```

The dataset class performs lesion-stratified patient-level train/validation splitting internally. Separate `train/` and `val/` folders are therefore not required.

## Training Notes

Before launching training, set the task-specific paths and phases in `train_v5.py`, including:

```python
phase_1 = "pre"
phase_2 = "c_a"  # or "c_v"
data_path = "/path/to/LLD-MMRI/2d_mri_body_dataset_mutil_phase_corp_body_v1"
lesion_patient_file = "/path/to/LLD-MMRI/lesion_patient_list.txt"
```

The script uses PyTorch distributed training utilities. For a single GPU run, a typical launch pattern is:

```bash
torchrun --nproc_per_node=1 train_v5.py
```

Please adjust batch size, learning rate, number of workers, save paths, and attention settings according to your hardware and software environment.

## Citation

If you use this code, please cite the corresponding MG-RDFD paper. Citation information will be updated after the proceedings metadata is available.
