"""Reader/writer for the widely-used 3D Gaussian Splatting .ply convention
(as written by the original INRIA codebase and read by SuperSplat,
PlayCanvas, gsplat, and most viewers):

    x, y, z, nx, ny, nz,
    f_dc_0, f_dc_1, f_dc_2,
    f_rest_0 .. f_rest_{3*K-1}   (channel-major: all of channel 0's K
                                  coefficients, then channel 1's, then 2's)
    opacity, scale_0, scale_1, scale_2,
    rot_0, rot_1, rot_2, rot_3    (quaternion w, x, y, z)

scale/opacity are stored raw (log-scale, logit-opacity) — the same
parameterization the aggregate defaults to.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement, PlyParseError

from splat.domain.errors import InvalidGaussianCloud, UnsupportedSHDegree
from splat.domain.gaussians import (
    MAX_SH_DEGREE,
    GaussianCloud,
    GaussianCloudMetadata,
    sh_rest_count,
)
from splat.domain.value_objects import ModelLicense


def _degree_from_rest_count(count: int) -> int:
    for degree in range(MAX_SH_DEGREE + 1):
        if sh_rest_count(degree) == count:
            return degree
    raise UnsupportedSHDegree(f"Unrecognized f_rest coefficient count per channel: {count}")


class PlyReader:
    name = "ply"

    def read(self, path: Path) -> GaussianCloud:
        try:
            ply = PlyData.read(str(path))
            vertex = ply["vertex"]
        except (PlyParseError, KeyError) as exc:
            raise InvalidGaussianCloud(f"{path.name} is not a Gaussian splat PLY: {exc}") from exc
        names = vertex.data.dtype.names
        if "f_dc_0" not in names:
            raise InvalidGaussianCloud(f"{path.name} is not a Gaussian splat PLY (no f_dc_0).")
        comments = dict(c.split(" ", 1) for c in ply.comments if " " in c)
        n = vertex["x"].shape[0]

        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(
            np.float32
        )

        rest_names = sorted(
            (name for name in names if name.startswith("f_rest_")),
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        if rest_names:
            num_rest_per_channel = len(rest_names) // 3
            flat = np.stack([vertex[name] for name in rest_names], axis=1).astype(np.float32)
            # stored channel-major (c, k); domain wants coefficient-major (k, c)
            sh_rest = flat.reshape(n, 3, num_rest_per_channel).transpose(0, 2, 1).copy()
            sh_degree = _degree_from_rest_count(num_rest_per_channel)
        else:
            sh_rest = None
            sh_degree = 0

        opacities = np.asarray(vertex["opacity"], dtype=np.float32)
        scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(
            np.float32
        )
        rotations = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
        ).astype(np.float32)

        return GaussianCloud(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
            sh_degree=sh_degree,
            scale_activation="log",
            opacity_activation="logit",
            metadata=GaussianCloudMetadata(
                source_format="ply",
                source_model=comments.get("source_model"),
                license=ModelLicense(**json.loads(comments["license"]))
                if "license" in comments
                else None,
                up_axis=comments.get("up_axis", "y"),
                coordinate_convention=comments.get("coordinate_convention", "opengl"),
                capture_camera_position=json.loads(comments["capture_camera_position"])
                if "capture_camera_position" in comments
                else None,
                capture_camera_rotation=json.loads(comments["capture_camera_rotation"])
                if "capture_camera_rotation" in comments
                else None,
                capture_camera_intrinsics=json.loads(comments["capture_camera_intrinsics"])
                if "capture_camera_intrinsics" in comments
                else None,
                capture_camera_count=int(comments["capture_camera_count"])
                if "capture_camera_count" in comments
                else None,
                source_cameras=json.loads(comments["source_cameras"])
                if "source_cameras" in comments
                else None,
            ),
        )


class PlyWriter:
    name = "ply"

    def write(self, cloud: GaussianCloud, path: Path) -> None:
        n = cloud.point_count
        num_rest_per_channel = cloud.sh_rest.shape[1] if cloud.sh_rest is not None else 0

        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            *[(f"f_rest_{i}", "f4") for i in range(num_rest_per_channel * 3)],
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
        vertices = np.zeros(n, dtype=dtype)
        vertices["x"], vertices["y"], vertices["z"] = (
            cloud.means[:, 0],
            cloud.means[:, 1],
            cloud.means[:, 2],
        )
        vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = (
            cloud.sh_dc[:, 0],
            cloud.sh_dc[:, 1],
            cloud.sh_dc[:, 2],
        )
        if cloud.sh_rest is not None:
            # domain is coefficient-major (k, c); format wants channel-major (c, k)
            flat = cloud.sh_rest.transpose(0, 2, 1).reshape(n, -1)
            for i in range(flat.shape[1]):
                vertices[f"f_rest_{i}"] = flat[:, i]

        vertices["opacity"] = cloud.to_logit_opacities()
        log_scales = cloud.to_log_scales()
        vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = (
            log_scales[:, 0],
            log_scales[:, 1],
            log_scales[:, 2],
        )
        vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = (
            cloud.rotations[:, 0],
            cloud.rotations[:, 1],
            cloud.rotations[:, 2],
            cloud.rotations[:, 3],
        )

        element = PlyElement.describe(vertices, "vertex")
        comments = [
            f"up_axis {cloud.metadata.up_axis}",
            f"coordinate_convention {cloud.metadata.coordinate_convention}",
        ]
        if cloud.metadata.source_model is not None:
            comments.append(f"source_model {cloud.metadata.source_model}")
        if cloud.metadata.license is not None:
            comments.append(f"license {json.dumps(asdict(cloud.metadata.license))}")
        if cloud.metadata.capture_camera_position is not None:
            comments.append(
                f"capture_camera_position {json.dumps(cloud.metadata.capture_camera_position)}"
            )
        if cloud.metadata.capture_camera_rotation is not None:
            comments.append(
                f"capture_camera_rotation {json.dumps(cloud.metadata.capture_camera_rotation)}"
            )
        if cloud.metadata.capture_camera_intrinsics is not None:
            comments.append(
                f"capture_camera_intrinsics {json.dumps(cloud.metadata.capture_camera_intrinsics)}"
            )
        if cloud.metadata.capture_camera_count is not None:
            comments.append(f"capture_camera_count {cloud.metadata.capture_camera_count}")
        if cloud.metadata.source_cameras is not None:
            comments.append(f"source_cameras {json.dumps(cloud.metadata.source_cameras)}")
        PlyData([element], text=False, comments=comments).write(str(path))

    def supports(self, cloud: GaussianCloud) -> list[str]:
        return []  # .ply carries full SH degree and float precision losslessly
