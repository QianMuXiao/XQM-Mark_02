import os
import random
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage


def resize_and_normalize_array(input_array, target_shape, order=1, normalize=True):
    if len(input_array.shape) != len(target_shape):
        raise ValueError("输入数组的维度和目标形状的维度不匹配。")

    scale_factors = [target_shape[i] / input_array.shape[i] for i in range(len(target_shape))]
    resized_array = scipy.ndimage.zoom(input_array, scale_factors, order=order)

    if normalize:
        min_val = np.min(resized_array)
        max_val = np.max(resized_array)
        if max_val > min_val:
            normalized_resized_array = np.interp(resized_array, (min_val, max_val), (-1, 1))
        else:
            normalized_resized_array = np.zeros_like(resized_array)
    else:
        normalized_resized_array = resized_array

    return normalized_resized_array


def _zoom_to_shape(arr: np.ndarray, out_shape: Tuple[int, int], order: int):
    if arr.shape == out_shape:
        return arr
    scale_factors = [out_shape[0] / arr.shape[0], out_shape[1] / arr.shape[1]]
    return scipy.ndimage.zoom(arr, scale_factors, order=order)


def _crop_square_and_resize(arr: np.ndarray, y0: int, x0: int, side: int, out_shape: Tuple[int, int], order: int):
    crop = arr[y0:y0 + side, x0:x0 + side]
    return _zoom_to_shape(crop, out_shape, order=order)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))


def _sample_square_roi(
    h: int,
    w: int,
    scale_range: Tuple[float, float],
    rng: np.random.Generator,
    center_bias: bool = True,
    center_std: float = 0.18,
):
    """
    返回 (y0, x0, side)，其中 side 是正方形边长像素数。
    center_bias=True 时，ROI 中心服从以图像中心为均值的截断高斯；
    center_std 是相对标准差（相对于 min(h,w)），越小越偏向中心。
    """
    s_min, s_max = float(scale_range[0]), float(scale_range[1])
    if s_min > s_max:
        s_min, s_max = s_max, s_min

    side = int(round(rng.uniform(s_min, s_max) * min(h, w)))
    side = _clamp_int(side, 1, min(h, w))

    # 采样中心点
    if center_bias:
        std_pix = max(1.0, float(center_std) * float(min(h, w)))
        cy = rng.normal(loc=(h - 1) / 2.0, scale=std_pix)
        cx = rng.normal(loc=(w - 1) / 2.0, scale=std_pix)
        cy = float(np.clip(cy, 0.0, h - 1.0))
        cx = float(np.clip(cx, 0.0, w - 1.0))
    else:
        cy = rng.uniform(0.0, h - 1.0)
        cx = rng.uniform(0.0, w - 1.0)

    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))
    y0 = _clamp_int(y0, 0, h - side) if h > side else 0
    x0 = _clamp_int(x0, 0, w - side) if w > side else 0
    return y0, x0, side


class PairMRIDataset(Dataset):
    def __init__(
        self,
        data_path,
        lesion_patient_file,
        split: str = "train",
        val_ratio: float = 0.2,
        image_size: Tuple[int, int] = (256, 256),
        random_seed: int = 42,
        phase_1: str = "pre",
        phase_2: str = "delay",
        # ---- v4: ROI augmentation ----
        roi_aug_prob: float = 0.0,
        roi_scale_range: Tuple[float, float] = (0.6, 1.0),
        roi_aug_on_val: bool = False,
        roi_center_bias: bool = True,
        roi_center_std: float = 0.18,
        roi_clip: bool = True,
        # ---- v4: optional full-res mask output (for eval/vis) ----
        return_full_res_mask: bool = False,
    ):
        """
        v4 增强：随机正方形ROI裁剪→resize回原尺寸；支持中心偏置。

        Args:
            roi_aug_prob: 增强概率（0 关闭，1 总是启用）
            roi_scale_range: ROI 边长占 min(H,W) 的比例范围，例如 (0.6,1.0)
            roi_aug_on_val: 验证集是否也做增强（默认 False）
            roi_center_bias: 是否对 ROI 中心做“偏向图像中心”的采样
            roi_center_std: 中心采样 std（相对 min(H,W)），越小越集中
            roi_clip: ROI resize 后是否 clip 到 [-1,1]
            return_full_res_mask: 若为 True，则 __getitem__ 额外返回 final_mask_full（与 image_size 同分辨率）
        """
        self.data_path = data_path
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.image_size = image_size
        self.split = split

        self.random_seed = int(random_seed)
        self.epoch = 0  # 可选：外部每个 epoch 调一次 set_epoch 增加随机性

        # ROI aug config
        self.roi_aug_prob = float(roi_aug_prob)
        self.roi_scale_range = roi_scale_range
        self.roi_aug_on_val = bool(roi_aug_on_val)
        self.roi_center_bias = bool(roi_center_bias)
        self.roi_center_std = float(roi_center_std)
        self.roi_clip = bool(roi_clip)

        # full-res mask output
        self.return_full_res_mask = bool(return_full_res_mask)

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.lesion_patient_dict = self.load_lesion_patient_ids(lesion_patient_file)
        self.patient_ids = self.split_patient_ids(val_ratio)

        self.phase_1_images = []
        self.phase_2_images = []
        self.mask_images = []
        self.tumor_images = []
        self.body_images = []

        for p_id in self.patient_ids:
            phase_1_dir = os.path.join(data_path, self.phase_1)
            phase_2_dir = os.path.join(data_path, self.phase_2)
            mask_dir = os.path.join(data_path, "totalseg")
            tumor_dir = os.path.join(data_path, "tumor")
            body_dir = os.path.join(data_path, "body")

            phase_1_files = [f for f in os.listdir(phase_1_dir) if f.startswith(p_id) and f.endswith(".nii.gz")]
            phase_2_files = [f for f in os.listdir(phase_2_dir) if f.startswith(p_id) and f.endswith(".nii.gz")]
            mask_files = [f for f in os.listdir(mask_dir) if f.startswith(p_id) and f.endswith(".nii.gz")]
            tumor_files = [f for f in os.listdir(tumor_dir) if f.startswith(p_id) and f.endswith(".nii.gz")]
            body_files = [f for f in os.listdir(body_dir) if f.startswith(p_id) and f.endswith(".nii.gz")]

            self.phase_1_images.extend([os.path.join(phase_1_dir, f) for f in phase_1_files])
            self.phase_2_images.extend([os.path.join(phase_2_dir, f) for f in phase_2_files])
            self.mask_images.extend([os.path.join(mask_dir, f) for f in mask_files])
            self.tumor_images.extend([os.path.join(tumor_dir, f) for f in tumor_files])
            self.body_images.extend([os.path.join(body_dir, f) for f in body_files])

        assert len(self.phase_1_images) == len(self.phase_2_images), "两相图像数量不匹配"
        assert len(self.phase_1_images) == len(self.mask_images), "图像和掩码数量不匹配"

        self.image_pairs = []

        phase_1_dict = {}
        for path in self.phase_1_images:
            filename = os.path.basename(path)
            p_id, slice_idx = self.parse_filename(filename)
            phase_1_dict[(p_id, slice_idx)] = path

        phase_2_dict = {}
        for path in self.phase_2_images:
            filename = os.path.basename(path)
            p_id, slice_idx = self.parse_filename(filename)
            phase_2_dict[(p_id, slice_idx)] = path

        mask_dict = {}
        for path in self.mask_images:
            filename = os.path.basename(path)
            p_id, slice_idx = self.parse_filename(filename)
            mask_dict[(p_id, slice_idx)] = path

        tumor_dict = {}
        for path in self.tumor_images:
            filename = os.path.basename(path)
            p_id, slice_idx = self.parse_filename(filename)
            tumor_dict[(p_id, slice_idx)] = path

        body_dict = {}
        for path in self.body_images:
            filename = os.path.basename(path)
            p_id, slice_idx = self.parse_filename(filename)
            body_dict[(p_id, slice_idx)] = path

        # v4: common_keys 需包含 body_dict，避免 body_path 缺失导致后续 ReadImage 报错
        common_keys = (
            set(phase_1_dict.keys())
            & set(phase_2_dict.keys())
            & set(mask_dict.keys())
            & set(tumor_dict.keys())
            & set(body_dict.keys())
        )

        for key in common_keys:
            self.image_pairs.append((phase_1_dict[key], phase_2_dict[key], mask_dict[key], tumor_dict[key], body_dict[key]))

        random.shuffle(self.image_pairs)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def load_lesion_patient_ids(self, file_path):
        lesion_patient_dict = {}
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_lesion_class = None
        for line in lines:
            line = line.strip()
            if line.startswith("病灶分类"):
                current_lesion_class = line.split()[1].replace(":", "")
                lesion_patient_dict[current_lesion_class] = []
            elif line.startswith("病人 ID:"):
                patient_id = line.split(":")[1].strip()
                lesion_patient_dict[current_lesion_class].append(patient_id)

        return lesion_patient_dict

    def split_patient_ids(self, val_ratio):
        selected_patient_ids = []
        for lesion_class, patient_ids in self.lesion_patient_dict.items():
            patient_ids = sorted(list(set(patient_ids)))
            total_patients = len(patient_ids)
            val_count = max(1, int(total_patients * val_ratio))

            random.shuffle(patient_ids)

            if self.split == "train":
                selected_ids = patient_ids[val_count:]
            else:
                selected_ids = patient_ids[:val_count]

            selected_patient_ids.extend(selected_ids)

        return selected_patient_ids

    def parse_filename(self, filename):
        name = filename.replace(".nii.gz", "")
        parts = name.split("_")
        patient_id = parts[0]
        slice_idx = parts[1]
        return patient_id, slice_idx

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        phase_1_path, phase_2_path, mask_path, tumor_path, body_path = self.image_pairs[idx]

        phase_1_img = sitk.ReadImage(phase_1_path)
        phase_2_img = sitk.ReadImage(phase_2_path)
        mask_img = sitk.ReadImage(mask_path)
        tumor_img = sitk.ReadImage(tumor_path)
        body_img = sitk.ReadImage(body_path)

        phase_1_array = sitk.GetArrayFromImage(phase_1_img)
        phase_2_array = sitk.GetArrayFromImage(phase_2_img)

        mask_raw = sitk.GetArrayFromImage(mask_img)
        tumor_raw = sitk.GetArrayFromImage(tumor_img)
        body_raw = sitk.GetArrayFromImage(body_img)

        mask_full_array = None
        tumor_full_array = None
        body_full_array = None

        if self.image_size:
            # v4: feature_size 必须是 int tuple
            feature_size = (int(self.image_size[0] // 4), int(self.image_size[1] // 4))

            phase_1_array = resize_and_normalize_array(phase_1_array, self.image_size, order=1)
            phase_2_array = resize_and_normalize_array(phase_2_array, self.image_size, order=1)

            # masks for feature-size (H/4, W/4)
            mask_small_array = resize_and_normalize_array(mask_raw, feature_size, order=0, normalize=False)
            tumor_small_array = resize_and_normalize_array(tumor_raw, feature_size, order=0, normalize=False)
            body_small_array = resize_and_normalize_array(body_raw, feature_size, order=0, normalize=False)

            # masks for full-size (H, W) - optional for eval/vis
            if self.return_full_res_mask:
                mask_full_array = resize_and_normalize_array(mask_raw, self.image_size, order=0, normalize=False)
                tumor_full_array = resize_and_normalize_array(tumor_raw, self.image_size, order=0, normalize=False)
                body_full_array = resize_and_normalize_array(body_raw, self.image_size, order=0, normalize=False)
        else:
            # no resizing
            mask_small_array = mask_raw
            tumor_small_array = tumor_raw
            body_small_array = body_raw
            if self.return_full_res_mask:
                mask_full_array = mask_raw
                tumor_full_array = tumor_raw
                body_full_array = body_raw

        # ----------------------------
        # v4: ROI augment (same ROI for A/B and masks)
        # ----------------------------
        do_roi = (
            (self.split == "train" or self.roi_aug_on_val)
            and (self.roi_aug_prob > 0.0)
            and (random.random() < self.roi_aug_prob)
        )
        if do_roi:
            # 用 per-sample RNG：避免多 worker 时全局 random 状态互相影响
            # 这里混合 epoch 与 idx，保证你愿意时可通过 set_epoch 让增强随 epoch 变化
            seed = (self.random_seed * 1000003 + self.epoch * 9176 + idx) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)

            H, W = phase_1_array.shape
            y0, x0, side = _sample_square_roi(
                H,
                W,
                scale_range=self.roi_scale_range,
                rng=rng,
                center_bias=self.roi_center_bias,
                center_std=self.roi_center_std,
            )

            phase_1_array = _crop_square_and_resize(phase_1_array, y0, x0, side, (H, W), order=1)
            phase_2_array = _crop_square_and_resize(phase_2_array, y0, x0, side, (H, W), order=1)

            # feature-size masks ROI（按分辨率比例映射 ROI）
            h2, w2 = mask_small_array.shape
            y0m = int(round(y0 * h2 / H))
            x0m = int(round(x0 * w2 / W))
            sidem = int(round(side * h2 / H))
            sidem = _clamp_int(sidem, 1, min(h2, w2))
            y0m = _clamp_int(y0m, 0, h2 - sidem) if h2 > sidem else 0
            x0m = _clamp_int(x0m, 0, w2 - sidem) if w2 > sidem else 0

            mask_small_array = _crop_square_and_resize(mask_small_array, y0m, x0m, sidem, (h2, w2), order=0)
            tumor_small_array = _crop_square_and_resize(tumor_small_array, y0m, x0m, sidem, (h2, w2), order=0)
            body_small_array = _crop_square_and_resize(body_small_array, y0m, x0m, sidem, (h2, w2), order=0)

            # full-res masks ROI（如果开启了 full mask 输出）
            if self.return_full_res_mask:
                mask_full_array = _crop_square_and_resize(mask_full_array, y0, x0, side, (H, W), order=0)
                tumor_full_array = _crop_square_and_resize(tumor_full_array, y0, x0, side, (H, W), order=0)
                body_full_array = _crop_square_and_resize(body_full_array, y0, x0, side, (H, W), order=0)

            if self.roi_clip:
                phase_1_array = np.clip(phase_1_array, -1, 1)
                phase_2_array = np.clip(phase_2_array, -1, 1)

        phase_1_image = torch.tensor(phase_1_array, dtype=torch.float32).unsqueeze(0)
        phase_2_image = torch.tensor(phase_2_array, dtype=torch.float32).unsqueeze(0)

        # order=0 resize 后可能是 float，但值仍是最近邻；这里直接转 int16
        mask_image = torch.tensor(mask_small_array, dtype=torch.int16).unsqueeze(0)
        tumor_image = torch.tensor(tumor_small_array, dtype=torch.int16).unsqueeze(0)
        body_image = torch.tensor(body_small_array, dtype=torch.int16).unsqueeze(0)

        final_mask = torch.concat([mask_image, tumor_image, body_image], dim=0)

        if self.return_full_res_mask:
            mask_full_image = torch.tensor(mask_full_array, dtype=torch.int16).unsqueeze(0)
            tumor_full_image = torch.tensor(tumor_full_array, dtype=torch.int16).unsqueeze(0)
            body_full_image = torch.tensor(body_full_array, dtype=torch.int16).unsqueeze(0)
            final_mask_full = torch.concat([mask_full_image, tumor_full_image, body_full_image], dim=0)
            return phase_1_image, phase_2_image, final_mask, final_mask_full, phase_1_path, phase_2_path

        return phase_1_image, phase_2_image, final_mask, phase_1_path, phase_2_path


if __name__ == "__main__":
    data_path = "D:/LLD_MMRI_Dataset/2d_mri_body_dataset_mutil_phase_corp_body_v1"
    lesion_patient_file = "D:/LLD_MMRI_Dataset/lesion_patient_list.txt"

    dataset = PairMRIDataset(
        data_path=data_path,
        lesion_patient_file=lesion_patient_file,
        split="train",
        val_ratio=0.2,
        image_size=(160, 160),
        random_seed=42,
        phase_1="pre",
        phase_2="delay",
        roi_aug_prob=0.8,
        roi_scale_range=(0.6, 1.0),
        roi_center_bias=True,
        roi_center_std=0.18,
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for batch in tqdm(dataloader):
        phase_1_images, phase_2_images, final_mask, phase_1_paths, phase_2_paths = batch
        print("Phase 1 Images Shape:", phase_1_images.shape)
        print("Phase 2 Images Shape:", phase_2_images.shape)
        print("Final Mask Shape:", final_mask.shape)
        break