`mlx_stable_diffusion/` is vendored from
[ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)'s
`stable_diffusion/stable_diffusion/` package (MIT License, Copyright © 2023-2024
Apple Inc.), because it has no packaging metadata and isn't published to PyPI.
See `src/splat/adapters/diffusion/mlx_stable_diffusion.py` for the adapter
that wraps it against this project's `DiffusionBackend` port.

The only change reads the model files from a local directory (`model_io.py`'s `_FILES`) instead of downloading them from the Hugging Face Hub.
