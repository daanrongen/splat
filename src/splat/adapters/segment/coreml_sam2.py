"""Automatic mask segmentation via Apple's official CoreML SAM2.1 conversion
(`apple/coreml-sam2.1-*`). Apple ships only the raw image-encoder /
prompt-encoder / mask-decoder `.mlpackage` files — no automatic
mask-generation orchestration — so this module ports the SAM AMG algorithm
(grid-of-points prompting, score filtering, box NMS) to plain NumPy driving
those three CoreML models directly via coremltools.

The prompt-encoder's CoreML conversion has a fixed input shape requiring
exactly 2 points per call (confirmed against a working reference:
github.com/mikeesto/sam2-coreml-python); each grid point is duplicated to
satisfy that shape.
"""

from pathlib import Path

import coremltools as ct
import numpy as np
from huggingface_hub import snapshot_download

from splat.adapters.formats.image import read_rgb, resize
from splat.domain.image_space import Sticker
from splat.domain.value_objects import ModelLicense
from splat.paths import model_cache_dir

_INPUT_SIZE = (1024, 1024)  # (W, H), fixed by Apple's conversion


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        ious = _box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thresh]
    return keep


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _build_point_grid(points_per_side: int, width: int, height: int) -> np.ndarray:
    xs = (np.arange(points_per_side) + 0.5) / points_per_side * width
    ys = (np.arange(points_per_side) + 0.5) / points_per_side * height
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)


class CoreMLSam2Backend:
    name = "coreml-sam2"

    def __init__(self, *, hf_repo_id: str, license: ModelLicense, device: str = "auto") -> None:
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._compute_unit = (
            ct.ComputeUnit.CPU_ONLY if device == "cpu" else ct.ComputeUnit.CPU_AND_GPU
        )
        self._image_encoder = None
        self._prompt_encoder = None
        self._mask_decoder = None

    def _weights_dir(self) -> Path:
        target = model_cache_dir() / "coreml-sam2" / self._hf_repo_id.replace("/", "--")
        # local_dir= materializes real files (reusing already-downloaded blobs via
        # hardlink, no re-fetch) instead of the default symlinked cache — the
        # CoreML compiler cannot stage a symlinked weight.bin during compilation.
        if not any(target.glob("*.mlpackage")):
            snapshot_download(
                repo_id=self._hf_repo_id, local_dir=target, allow_patterns=["*.mlpackage/**"]
            )
        return target

    def _load(self) -> None:
        if self._image_encoder is not None:
            return
        packages = sorted((self._weights_dir()).glob("*.mlpackage"))
        by_role = {}
        for package in packages:
            if "ImageEncoder" in package.name:
                by_role["image_encoder"] = package
            elif "PromptEncoder" in package.name:
                by_role["prompt_encoder"] = package
            elif "MaskDecoder" in package.name:
                by_role["mask_decoder"] = package

        self._image_encoder = ct.models.MLModel(
            str(by_role["image_encoder"]), compute_units=self._compute_unit
        )
        self._prompt_encoder = ct.models.MLModel(
            str(by_role["prompt_encoder"]), compute_units=self._compute_unit
        )
        self._mask_decoder = ct.models.MLModel(
            str(by_role["mask_decoder"]), compute_units=self._compute_unit
        )

    def segment(
        self,
        image_path: Path,
        *,
        max_stickers: int | None = None,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.85,
        box_nms_thresh: float = 0.7,
        **params,
    ) -> list[Sticker]:
        self._load()

        original = read_rgb(image_path)
        original_size = (original.shape[1], original.shape[0])
        resized = resize(original, _INPUT_SIZE)

        embeddings = self._image_encoder.predict({"image": resized})

        grid = _build_point_grid(points_per_side, original_size[0], original_size[1])
        scale = np.array([_INPUT_SIZE[0] / original_size[0], _INPUT_SIZE[1] / original_size[1]])

        masks: list[np.ndarray] = []
        scores: list[float] = []
        boxes: list[list[float]] = []

        for point in grid:
            scaled_point = point * scale
            coords = np.stack([scaled_point, scaled_point])[None, :, :].astype(np.float32)
            labels = np.array([[1, 1]], dtype=np.int32)

            prompt_out = self._prompt_encoder.predict({"points": coords, "labels": labels})
            decoder_out = self._mask_decoder.predict(
                {
                    "image_embedding": embeddings["image_embedding"],
                    "sparse_embedding": prompt_out["sparse_embeddings"],
                    "dense_embedding": prompt_out["dense_embeddings"],
                    "feats_s0": embeddings["feats_s0"],
                    "feats_s1": embeddings["feats_s1"],
                }
            )

            best_idx = int(np.argmax(decoder_out["scores"][0]))
            score = float(decoder_out["scores"][0, best_idx])
            if score < pred_iou_thresh:
                continue

            low_res = decoder_out["low_res_masks"][0, best_idx]
            binary_mask = (
                resize(low_res.astype(np.float32), original_size, interpolation="linear") > 0
            )
            bbox = _mask_to_bbox(binary_mask)
            if bbox is None:
                continue

            x0, y0, w, h = bbox
            masks.append(binary_mask)
            scores.append(score)
            boxes.append([x0, y0, x0 + w, y0 + h])

        if not masks:
            return []

        keep = _nms(np.array(boxes), np.array(scores), box_nms_thresh)
        keep.sort(key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        if max_stickers is not None:
            keep = keep[:max_stickers]

        stickers = []
        for i in keep:
            mask = masks[i]
            x0, y0, x1, y1 = (int(v) for v in boxes[i])
            w, h = x1 - x0, y1 - y0
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = original[y0:y1, x0:x1]
            rgba[:, :, 3] = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
            stickers.append(
                Sticker(
                    rgba=rgba, bbox=(x0, y0, w, h), score=float(scores[i]), area=int(mask.sum())
                )
            )
        return stickers
