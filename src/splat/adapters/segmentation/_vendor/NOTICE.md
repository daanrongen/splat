`mlx_sam/` is vendored, unmodified, from
[ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)'s
`segment_anything/segment_anything/` package (MIT License, Copyright © 2023-2024
Apple Inc.), because it has no packaging metadata and isn't published to PyPI.
See `src/splat/adapters/segmentation/mlx_sam.py` for the adapter that wraps it
against this project's `SegmentationBackend` port, and for the weight
conversion logic (adapted from the upstream `convert.py`) that turns Meta's
original `facebook/sam-vit-base` checkpoint into MLX format.
