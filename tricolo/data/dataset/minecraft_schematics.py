import hashlib
import json
import os
import re
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


METADATA_COLUMNS = [
    "url",
    "title",
    "subtitle",
    "img",
    "user",
    "date",
    "bigImgs",
    "description",
    "tags",
    "diamondCount",
    "views",
    "downloads",
    "comments",
    "favorites",
    "downloadLink",
    "finalDownloadLink",
    "thirdPartyDownloadLink",
    "youtubeId",
]


def _cfg_get(container, key, default):
    return container[key] if key in container else default


def _resolve_path(path, root):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def _safe_name(value):
    return re.sub(r"[^\w\-]", "_", str(value))[:40]


class MinecraftSchematics(Dataset):
    """TriCoLo dataset adapter for Minecraft schematic Parquet rows.

    Each row is one retrieval identity. The positive pair is formed by metadata
    text, rendered views, and voxel data from the same row/model_id.
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.split = split
        self.project_root_path = Path(cfg.project_root_path)
        self.parquet_path = _resolve_path(self.data_cfg.parquet_path, self.project_root_path)
        self.renders_path = _resolve_path(self.data_cfg.renders_path, self.project_root_path)
        self.exp_data_root_path = _resolve_path(self.data_cfg.exp_data_root_path, self.project_root_path)
        self.exp_data_root_path.mkdir(parents=True, exist_ok=True)

        self.raw_voxel_size = int(_cfg_get(self.data_cfg, "raw_voxel_size", 32))
        self.voxel_size = int(self.data_cfg.voxel_size)
        if self.voxel_size != self.raw_voxel_size:
            raise ValueError(
                "MinecraftSchematics currently expects data.voxel_size to match "
                f"raw_voxel_size ({self.raw_voxel_size}); got {self.voxel_size}."
            )

        self.num_views = int(self.data_cfg.num_views)
        self.available_views = int(_cfg_get(self.data_cfg, "available_views", 12))
        self.image_size = int(self.data_cfg.image_size)
        self.view_pattern = _cfg_get(self.data_cfg, "view_pattern", "view_{view_idx:02d}.jpg")
        self.allow_missing_images = bool(_cfg_get(self.data_cfg, "allow_missing_images", False))
        self.drop_empty_voxels = bool(_cfg_get(self.data_cfg, "drop_empty_voxels", True))
        self.description_max_chars = int(_cfg_get(self.data_cfg, "description_max_chars", 512))
        self.include_description = bool(_cfg_get(self.data_cfg, "include_description", True))
        self.include_tags = bool(_cfg_get(self.data_cfg, "include_tags", True))
        self.max_text_len = int(_cfg_get(self.data_cfg, "max_text_len", 77))
        self.voxel_feature_mode = _cfg_get(self.data_cfg, "voxel_feature_mode", "block_hash")
        self.use_clip_tokens = cfg.model.text_encoder == "CLIPTextEncoder"

        self.normalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.view_indices = self._select_view_indices()

        metadata = pd.read_parquet(self.parquet_path, columns=METADATA_COLUMNS)
        metadata = metadata.reset_index(drop=True)
        metadata["dataset_index"] = np.arange(len(metadata), dtype=np.int32)
        metadata["model_id"] = metadata["dataset_index"].map(lambda idx: f"{idx + 1:06d}")
        metadata["caption"] = metadata.apply(self._build_caption, axis=1)

        self._ensure_voxel_cache(len(metadata))
        nonair = np.load(self._nonair_cache_path(), mmap_mode="r")
        if self.drop_empty_voxels:
            metadata = metadata[nonair[metadata["dataset_index"].to_numpy()] > 0]
            metadata = metadata.reset_index(drop=True)

        split_indices = self._split_indices(metadata)
        self.language_data = metadata.iloc[split_indices].reset_index(drop=True)
        self.voxel_data = np.load(self._voxel_cache_path(), mmap_mode="r")
        self.clip_embeddings = self._load_clip_embeddings()

    def __len__(self):
        return len(self.language_data)

    def __getitem__(self, idx):
        item = self.language_data.iloc[idx]
        caption = item["caption"]
        model_id = item["model_id"]

        if self.use_clip_tokens:
            tokens = clip.tokenize(caption, truncate=True)[0]
        else:
            tokens = torch.from_numpy(self._hash_tokens(caption))

        data_dict = {
            "model_id": model_id,
            "category": item["subtitle"],
            "tokens": tokens,
            "images": self._load_images(item),
        }

        locs, feats = self._load_sparse_voxels(int(item["dataset_index"]))
        data_dict["locs"] = torch.from_numpy(locs)
        data_dict["feats"] = torch.from_numpy(feats)

        clip_item = self.clip_embeddings.get(model_id) if self.clip_embeddings is not None else None
        data_dict["clip_embeddings_img"] = clip_item["img"].to(torch.float32) if clip_item is not None else None
        data_dict["clip_embeddings_text"] = clip_item["text"].to(torch.float32) if clip_item is not None else None
        return data_dict

    def _select_view_indices(self):
        if "view_indices" in self.data_cfg and self.data_cfg.view_indices is not None:
            return [int(index) for index in self.data_cfg.view_indices]
        return np.round(np.linspace(0, self.available_views - 1, self.num_views)).astype(int).tolist()

    def _build_caption(self, row):
        title = self._clean_text(row.get("title", ""))
        subtitle = self._clean_text(row.get("subtitle", ""))
        parts = [title, subtitle]

        if self.include_tags:
            tags = self._parse_tags(row.get("tags", ""))
            if tags:
                parts.append("Tags: " + ", ".join(tags))

        if self.include_description:
            description = self._clean_text(row.get("description", ""))
            if self.description_max_chars > 0:
                description = description[: self.description_max_chars]
            if description:
                parts.append(description)

        return ". ".join(part for part in parts if part)

    @staticmethod
    def _clean_text(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    @staticmethod
    def _parse_tags(value):
        try:
            tags = json.loads(value) if value else []
            return [str(tag) for tag in tags] if isinstance(tags, list) else []
        except Exception:
            return []

    def _hash_tokens(self, text):
        vocab_size = int(self.data_cfg.vocab_size)
        token_ids = np.zeros(self.max_text_len, dtype=np.int64)
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())[: self.max_text_len]
        for index, token in enumerate(tokens):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            token_ids[index] = int(digest, 16) % (vocab_size - 1) + 1
        return token_ids

    def _split_indices(self, metadata):
        train_ratio = float(_cfg_get(self.data_cfg, "train_ratio", 0.8))
        val_ratio = float(_cfg_get(self.data_cfg, "val_ratio", 0.1))
        seed = int(_cfg_get(self.data_cfg, "split_seed", 123))
        rng = np.random.default_rng(seed)
        selected = []

        for _, group in metadata.groupby("subtitle", sort=False):
            indices = group.index.to_numpy()
            rng.shuffle(indices)
            n_total = len(indices)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            if self.split == "train":
                split_part = indices[:n_train]
            elif self.split == "val":
                split_part = indices[n_train:n_train + n_val]
            elif self.split == "test":
                split_part = indices[n_train + n_val:]
            else:
                raise ValueError(f"Unknown split: {self.split}")
            selected.extend(split_part.tolist())

        return np.array(sorted(selected), dtype=np.int64)

    def _voxel_cache_path(self):
        dtype_name = _cfg_get(self.data_cfg, "voxel_cache_dtype", "uint16")
        return self.exp_data_root_path / f"voxel{self.raw_voxel_size}_{dtype_name}.npy"

    def _nonair_cache_path(self):
        return self.exp_data_root_path / f"voxel{self.raw_voxel_size}_nonair.npy"

    def _ensure_voxel_cache(self, num_rows):
        voxel_cache_path = self._voxel_cache_path()
        nonair_cache_path = self._nonair_cache_path()
        expected_shape = (num_rows, self.raw_voxel_size ** 3)
        if voxel_cache_path.exists() and nonair_cache_path.exists():
            cached = np.load(voxel_cache_path, mmap_mode="r")
            if cached.shape == expected_shape:
                return

        dtype = np.dtype(_cfg_get(self.data_cfg, "voxel_cache_dtype", "uint16"))
        batch_size = int(_cfg_get(self.data_cfg, "voxel_cache_batch_size", 128))
        parquet = pq.ParquetFile(self.parquet_path)
        voxels = np.lib.format.open_memmap(voxel_cache_path, mode="w+", dtype=dtype, shape=expected_shape)
        nonair = np.lib.format.open_memmap(nonair_cache_path, mode="w+", dtype=np.int32, shape=(num_rows,))

        offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["voxel_data"]):
            values = batch.column(0).values.to_numpy(zero_copy_only=False)
            values = values.reshape(batch.num_rows, self.raw_voxel_size ** 3).astype(dtype, copy=False)
            voxels[offset:offset + batch.num_rows] = values
            nonair[offset:offset + batch.num_rows] = np.count_nonzero(values, axis=1)
            offset += batch.num_rows

        voxels.flush()
        nonair.flush()

    def _load_clip_embeddings(self):
        clip_path = self.exp_data_root_path / f"clip_embeddings_{self.split}.pth"
        if clip_path.exists() and (
            self.cfg.model.text_encoder == "CLIPTextEncoder" or self.cfg.model.image_encoder == "CLIPImageEncoder"
        ):
            return torch.load(clip_path, map_location="cpu")
        return None

    def _render_folder(self, row):
        dataset_index = int(row["dataset_index"])
        title = row["title"]
        folder_name = f"{dataset_index + 1:06d}_{_safe_name(title)}"
        folder = self.renders_path / folder_name
        if folder.exists():
            return folder

        legacy_folder = self.renders_path / f"{dataset_index + 1:05d}_{_safe_name(title)}"
        if legacy_folder.exists():
            return legacy_folder

        return folder

    def _load_images(self, row):
        folder = self._render_folder(row)
        images = []
        for view_idx in self.view_indices:
            image_path = folder / self.view_pattern.format(view_idx=view_idx)
            if not image_path.exists():
                if self.allow_missing_images:
                    images.append(torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32))
                    continue
                raise FileNotFoundError(f"Missing render image: {image_path}")

            with Image.open(image_path) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
                tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).to(torch.float32) / 255.0
                images.append(self.normalize(tensor))

        return torch.stack(images, dim=0)

    def _load_sparse_voxels(self, dataset_index):
        voxel = np.asarray(self.voxel_data[dataset_index]).reshape(
            self.raw_voxel_size, self.raw_voxel_size, self.raw_voxel_size
        )
        coords = np.stack(np.nonzero(voxel), axis=1).astype(np.int32)
        if len(coords) == 0:
            return np.zeros((1, 3), dtype=np.int32), np.zeros((1, 3), dtype=np.float32)

        block_ids = voxel[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
        if self.voxel_feature_mode == "occupancy":
            feats = np.ones((len(coords), 3), dtype=np.float32)
        elif self.voxel_feature_mode == "block_id":
            max_block_state_id = float(_cfg_get(self.data_cfg, "max_block_state_id", 20000))
            normalized = np.clip(block_ids / max_block_state_id, 0.0, 1.0)
            feats = np.stack([normalized, np.ones_like(normalized), np.log1p(block_ids) / np.log1p(max_block_state_id)], axis=1)
        elif self.voxel_feature_mode == "block_hash":
            ids = block_ids.astype(np.int64)
            feats = np.stack([
                (ids % 251) / 250.0,
                ((ids * 37) % 251) / 250.0,
                ((ids * 97) % 251) / 250.0,
            ], axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unknown voxel_feature_mode: {self.voxel_feature_mode}")

        return coords, feats.astype(np.float32)
