from pathlib import Path

import numpy as np
import pytest

from splat.adapters.formats.ply import PlyWriter
from splat.adapters.gaussian.mlx3d_capture import MLX3DCaptureBackend
from splat.domain.errors import SplatDomainError
from splat.domain.value_objects import MIT


class _FakeCamera:
    def __init__(
        self, position, rotation, fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480
    ):
        self.camera_center = np.array(position, dtype=np.float32)
        self.R = np.array(rotation, dtype=np.float32)
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height


class _FakeColmapDataset:
    def __init__(self, cameras):
        self.cameras = cameras


def _write_input_image(path: Path) -> None:
    from tests.image_helpers import write_sample_png

    write_sample_png(path, (2, 2))


def _backend() -> MLX3DCaptureBackend:
    return MLX3DCaptureBackend(license=MIT)


def _mock_run_capture(mocker, tmp_path: Path, synthetic_cloud, summary_extra: dict):
    splat_path = tmp_path / "splat.ply"
    PlyWriter().write(synthetic_cloud, splat_path)
    summary = {"splat": str(splat_path), **summary_extra}
    mocker.patch("mlx3d.capture.run_capture", return_value=summary)
    return summary


def test_reconstruct_attaches_camera_pose_from_refined_sparse(mocker, tmp_path, synthetic_cloud):
    image = tmp_path / "a.png"
    _write_input_image(image)
    refined_sparse = tmp_path / "capture" / "refined" / "sparse" / "0"
    _mock_run_capture(
        mocker, tmp_path, synthetic_cloud, {"train": {"refined_sparse": str(refined_sparse)}}
    )

    cam = _FakeCamera(position=[1.0, 2.0, 3.0], rotation=np.eye(3).tolist())
    load_colmap = mocker.patch(
        "mlx3d.datasets.colmap.load_colmap", return_value=_FakeColmapDataset([cam])
    )

    cloud = _backend().reconstruct([image])

    load_colmap.assert_called_once_with(str(tmp_path / "capture" / "refined"), load_images=False)
    assert cloud.metadata.capture_camera_position == [1.0, 2.0, 3.0]
    assert cloud.metadata.capture_camera_rotation == np.eye(3).tolist()
    assert cloud.metadata.capture_camera_intrinsics == [800.0, 800.0, 320.0, 240.0, 640.0, 480.0]
    assert cloud.metadata.capture_camera_count == 1


def test_reconstruct_uses_primary_sparse_when_no_refinement(mocker, tmp_path, synthetic_cloud):
    image = tmp_path / "a.png"
    _write_input_image(image)
    _mock_run_capture(mocker, tmp_path, synthetic_cloud, {"train": {}})

    load_colmap = mocker.patch(
        "mlx3d.datasets.colmap.load_colmap",
        return_value=_FakeColmapDataset(
            [_FakeCamera(position=[0, 0, 0], rotation=np.eye(3).tolist())]
        ),
    )

    cloud = _backend().reconstruct([image])

    # output_dir is inside the backend's own tempfile.TemporaryDirectory(), not tmp_path -
    # only its basename ("capture", set by the adapter itself) is knowable from here.
    called_root, called_kwargs = load_colmap.call_args
    assert Path(called_root[0]).name == "capture"
    assert called_kwargs == {"load_images": False}
    assert cloud.metadata.capture_camera_position == [0.0, 0.0, 0.0]


def test_reconstruct_leaves_pose_none_when_sparse_model_missing(mocker, tmp_path, synthetic_cloud):
    image = tmp_path / "a.png"
    _write_input_image(image)
    _mock_run_capture(mocker, tmp_path, synthetic_cloud, {"train": {}})
    mocker.patch("mlx3d.datasets.colmap.load_colmap", side_effect=FileNotFoundError)

    cloud = _backend().reconstruct([image])

    assert cloud.metadata.capture_camera_position is None
    assert cloud.metadata.capture_camera_count is None


def test_reconstruct_leaves_pose_none_when_no_cameras_registered(mocker, tmp_path, synthetic_cloud):
    image = tmp_path / "a.png"
    _write_input_image(image)
    _mock_run_capture(mocker, tmp_path, synthetic_cloud, {"train": {}})
    mocker.patch("mlx3d.datasets.colmap.load_colmap", return_value=_FakeColmapDataset([]))

    cloud = _backend().reconstruct([image])

    assert cloud.metadata.capture_camera_position is None


def test_reconstruct_raises_a_domain_error_when_capture_fails(mocker, tmp_path):
    image = tmp_path / "a.png"
    _write_input_image(image)
    mocker.patch("mlx3d.capture.run_capture", side_effect=RuntimeError("no views registered"))

    with pytest.raises(SplatDomainError, match="no views registered"):
        _backend().reconstruct([image])


def test_stage_inputs_evens_out_exposure_and_keeps_exif(tmp_path):
    from PIL import Image

    from splat.adapters.gaussian.mlx3d_capture import _stage_inputs

    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 24
    dark, bright = tmp_path / "dark.jpg", tmp_path / "bright.jpg"
    Image.new("RGB", (8, 8), (60, 60, 60)).save(dark, exif=exif.tobytes())
    Image.new("RGB", (8, 8), (180, 180, 180)).save(bright, exif=exif.tobytes())
    staged = tmp_path / "staged"
    staged.mkdir()

    _stage_inputs([dark, bright], staged, normalize_color=True)

    means = [np.asarray(Image.open(p)).mean() for p in sorted(staged.iterdir())]
    assert abs(means[0] - means[1]) < 5
    with Image.open(staged / "000.jpg") as image:
        assert image.getexif().get_ifd(0x8769).get(0xA405) == 24
