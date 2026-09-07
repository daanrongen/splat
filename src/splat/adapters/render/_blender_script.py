# Runs inside Blender's own bundled Python (`blender --background --python
# <this file> -- --input ... --output ...`), not splat's venv - only
# bpy/numpy/stdlib are available here, no `splat` imports.
#
# Reads the .npz dump written by BlenderBackend and renders each Gaussian with
# the image formation model 3DGS actually uses: emissive, unlit, alpha
# composited, with an `exp(-0.5 * r^2)` falloff. See `_make_material` for the
# alpha derivation and issue #82 for what this replaces (lit opaque IcoSpheres
# under a sun lamp, which turned every colour into a lighting response).
#
# Each Gaussian is a camera-facing quad, not a mesh ellipsoid: the falloff is
# evaluated per-ray in the shader from the Gaussian's own inverse-covariance
# basis, so the proxy geometry only has to cover the kernel's screen footprint.
# Two triangles per Gaussian instead of an 80-face IcoSphere is a 40x cut in
# geometry for an image that is strictly closer to a real rasterizer's.
#
# Two Blender quirks this graph works around, found by prototyping interactively
# over blender-mcp before wiring this up as a standalone script:
#   - a point-domain attribute must be explicitly re-stored onto the INSTANCE
#     domain (Store Named Attribute) before Realize Instances - it does not
#     survive automatically just by existing on the pre-instancing points.
#   - a material assigned via `pointcloud.materials.append(...)` is not picked
#     up by geometry-nodes-generated output - it needs an explicit Set Material
#     node inside the tree itself.

import argparse
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

_REPORT_PREFIX = "splat| "

# A ray crosses many kernels before it accumulates full opacity, and Cycles
# treats each one as a transparent bounce. The default 8 clips the stack and
# shows through to the background.
_TRANSPARENT_BOUNCES = 256

_NAMED_COLOURS = {"black": "000000", "white": "ffffff", "grey": "808080"}

# Per-Gaussian attributes that have to survive Realize Instances, as
# (name, geometry-nodes data type, shader Attribute output socket).
_INSTANCE_ATTRIBUTES = (
    ("Col", "FLOAT_COLOR"),
    ("pt_opacity", "FLOAT"),
    ("pt_center", "FLOAT_VECTOR"),
    ("pt_axis0", "FLOAT_VECTOR"),
    ("pt_axis1", "FLOAT_VECTOR"),
    ("pt_axis2", "FLOAT_VECTOR"),
)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=["cycles", "eevee"], default="cycles")
    # Not `--background`: Blender's own top-level flag already uses that name,
    # and two of them in one argv reads as a bug even though argparse only sees
    # what follows the `--` separator.
    parser.add_argument("--background-color", default="black")
    return parser.parse_args(argv)


def _build_point_cloud(means, colors, opacities, axes, radii):
    n = len(means)
    pc = bpy.data.pointclouds.new("GaussianCloud")
    pc.resize(n)
    pc.attributes["position"].data.foreach_set("vector", means.flatten().tolist())

    # The instance origin is the Gaussian centre, but after Realize Instances
    # the shader's Position is a quad corner, so the centre has to travel as
    # its own attribute for the falloff to have something to measure from.
    pc.attributes.new("pt_center", "FLOAT_VECTOR", "POINT").data.foreach_set(
        "vector", means.flatten().tolist()
    )
    pc.attributes.new("pt_radius", "FLOAT", "POINT").data.foreach_set("value", radii.tolist())
    pc.attributes.new("pt_opacity", "FLOAT", "POINT").data.foreach_set("value", opacities.tolist())
    for index in range(3):
        pc.attributes.new(f"pt_axis{index}", "FLOAT_VECTOR", "POINT").data.foreach_set(
            "vector", axes[:, index, :].flatten().tolist()
        )

    col_attr = pc.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([colors, np.ones((n, 1), dtype=np.float32)], axis=1)
    col_attr.data.foreach_set("color", rgba.flatten().tolist())
    return pc


def _plug(node_tree, value, target):
    """Accepts either an output socket to link or a literal to set."""
    if hasattr(value, "is_output"):
        node_tree.links.new(value, target)
    else:
        target.default_value = value


def _vec(node_tree, operation, a, b=None, out="Vector"):
    node = node_tree.nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    _plug(node_tree, a, node.inputs[0])
    if b is not None:
        _plug(node_tree, b, node.inputs[1])
    return node.outputs[out]


def _num(node_tree, operation, a, b=None):
    node = node_tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    _plug(node_tree, a, node.inputs[0])
    if b is not None:
        _plug(node_tree, b, node.inputs[1])
    return node.outputs["Value"]


def _attribute(node_tree, name, out="Vector"):
    node = node_tree.nodes.new("ShaderNodeAttribute")
    node.attribute_name = name
    return node.outputs[out]


def _make_material():
    """Emissive alpha compositing with the 3DGS falloff evaluated per ray.

    A Gaussian's contribution at a pixel is `opacity * exp(-0.5 * m^2)` where
    `m` is the Mahalanobis distance between the Gaussian's centre and the view
    ray - the ray-Gaussian form of the kernel, rather than EWA splatting's
    projected-2D-covariance approximation. Both agree at the centre and differ
    only in how the tail is skewed by perspective.

    `pt_axis0..2` are the rows of `S^-1 R^T`, so `u = (axis_i . x)` puts the
    world offset `x = P - centre` into the space where the Gaussian is the unit
    isotropic sphere. There, closest approach is Pythagoras:

        m^2 = |u|^2 - (u . dhat)^2

    with `dhat` the view direction mapped into the same space and normalized.
    Emission (not a BSDF) and a Transparent BSDF mixed by that alpha means the
    rendered colour is the Gaussian's own colour, with no lighting response.
    """
    mat = bpy.data.materials.new("GaussianMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    geometry = nt.nodes.new("ShaderNodeNewGeometry")
    axes = [_attribute(nt, f"pt_axis{index}") for index in range(3)]
    offset = _vec(nt, "SUBTRACT", geometry.outputs["Position"], _attribute(nt, "pt_center"))

    def _to_local(vector):
        combine = nt.nodes.new("ShaderNodeCombineXYZ")
        for index, axis in enumerate(axes):
            nt.links.new(_vec(nt, "DOT_PRODUCT", axis, vector, out="Value"), combine.inputs[index])
        return combine.outputs["Vector"]

    u = _to_local(offset)
    direction = _vec(nt, "NORMALIZE", _to_local(geometry.outputs["Incoming"]))
    along = _vec(nt, "DOT_PRODUCT", u, direction, out="Value")
    m2 = _num(
        nt,
        "SUBTRACT",
        _vec(nt, "DOT_PRODUCT", u, u, out="Value"),
        _num(nt, "MULTIPLY", along, along),
    )
    alpha = _num(
        nt,
        "MULTIPLY",
        _attribute(nt, "pt_opacity", out="Fac"),
        _num(nt, "EXPONENT", _num(nt, "MULTIPLY", m2, -0.5)),
    )

    emission = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(_attribute(nt, "Col", out="Color"), emission.inputs["Color"])
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(alpha, mix.inputs["Fac"])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"  # EEVEE preview only; Cycles ignores it
    return mat


def _build_billboard_instancer(mat, rotation):
    """One camera-facing quad per Gaussian, sized to cover its 3-sigma support.

    `rotation` is the camera's own orientation: for a still frame every
    billboard shares it, so the quads span the view plane without needing a
    per-instance align-to-camera step.
    """
    ng = bpy.data.node_groups.new("GaussianBillboards", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    group_in = ng.nodes.new("NodeGroupInput")
    group_out = ng.nodes.new("NodeGroupOutput")

    quad = ng.nodes.new("GeometryNodeMeshGrid")
    quad.inputs["Size X"].default_value = 2.0
    quad.inputs["Size Y"].default_value = 2.0
    quad.inputs["Vertices X"].default_value = 2
    quad.inputs["Vertices Y"].default_value = 2

    named_radius = ng.nodes.new("GeometryNodeInputNamedAttribute")
    named_radius.data_type = "FLOAT"
    named_radius.inputs["Name"].default_value = "pt_radius"

    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(group_in.outputs["Geometry"], inst.inputs["Points"])
    ng.links.new(quad.outputs["Mesh"], inst.inputs["Instance"])
    ng.links.new(named_radius.outputs["Attribute"], inst.inputs["Scale"])
    inst.inputs["Rotation"].default_value = rotation

    geometry = inst.outputs["Instances"]
    for name, data_type in _INSTANCE_ATTRIBUTES:
        named = ng.nodes.new("GeometryNodeInputNamedAttribute")
        named.data_type = data_type
        named.inputs["Name"].default_value = name
        store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type = data_type
        store.domain = "INSTANCE"
        store.inputs["Name"].default_value = name
        ng.links.new(geometry, store.inputs["Geometry"])
        ng.links.new(named.outputs["Attribute"], store.inputs["Value"])
        geometry = store.outputs["Geometry"]

    realize = ng.nodes.new("GeometryNodeRealizeInstances")
    ng.links.new(geometry, realize.inputs["Geometry"])

    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    ng.links.new(realize.outputs["Geometry"], set_mat.inputs["Geometry"])
    ng.links.new(set_mat.outputs["Geometry"], group_out.inputs["Geometry"])
    return ng


def _frame_camera_auto_fit(means: np.ndarray):
    median_center = np.median(means, axis=0)
    radii = np.linalg.norm(means - median_center, axis=1)
    # A high percentile, not the median: normalize_gaussian_cloud pins the
    # median radius to 1.0 while real clouds reach several times that, so a
    # median-derived distance puts the camera inside the cloud (issue #83).
    extent = float(np.percentile(radii, 95)) or 1.0
    center = Vector(median_center.tolist())

    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    distance = extent * 2.5
    cam_obj.location = center + Vector((distance * 0.7, -distance * 0.9, distance * 0.5))
    direction = (center - cam_obj.location).normalized()
    euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_obj.rotation_euler = euler
    bpy.context.scene.camera = cam_obj
    return euler


def _frame_camera_from_capture(position, rotation, intrinsics):
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = Vector(position.tolist())
    cam_obj.rotation_mode = "QUATERNION"
    quaternion = Matrix(rotation.tolist()).to_quaternion()
    cam_obj.rotation_quaternion = quaternion
    if intrinsics is not None:
        fx, _fy, _cx, _cy, width, _height = intrinsics.tolist()
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.lens_unit = "FOV"
        cam_data.angle = 2.0 * np.arctan(width / (2.0 * fx))
    bpy.context.scene.camera = cam_obj
    return quaternion.to_euler()


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _set_background(scene, spec: str) -> None:
    """`transparent` or a hex colour. Emissive Gaussians over the grey that
    --factory-startup ships read as fog; black is the honest default and
    transparent is what you want for compositing."""
    if spec == "transparent":
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        return
    text = _NAMED_COLOURS.get(spec, spec).lstrip("#")
    if len(text) != 6:
        raise SystemExit(
            f"--background-color expects transparent, a hex colour, or one of "
            f"{', '.join(_NAMED_COLOURS)}; got {spec!r}"
        )
    rgb = [_srgb_to_linear(int(text[i : i + 2], 16) / 255.0) for i in (0, 2, 4)]
    world = bpy.data.worlds.new("SplatWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*rgb, 1.0)
    scene.world = world


def _enable_gpu_compute(scene) -> str:
    """Cycles defaults to CPU-only unless a compute backend is both selected in
    preferences AND enabled per-device. Returns a human label for whatever it
    ended up on: a silent CPU fallback turns a 90-second render into a very
    long one, so the caller reports it rather than leaving it invisible."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpu_devices = [d for d in prefs.devices if d.type == backend]
        if gpu_devices:
            for device in prefs.devices:
                device.use = device.type == backend
            scene.cycles.device = "GPU"
            return f"{gpu_devices[0].name} [{backend}]"
    scene.cycles.device = "CPU"
    return "CPU (no GPU compute backend available)"


def _report(message: str) -> None:
    """Prefixed so the adapter can pick our lines out of Blender's own noise."""
    print(f"{_REPORT_PREFIX}{message}", flush=True)


def _install_progress_reporter() -> None:
    """Blender renders nothing to stdout in background mode, so a multi-minute
    Cycles render is indistinguishable from a hang. The `render_stats` handler
    does fire here, and carries both `Sample N/M` and the first-run
    "Loading render kernels" phase that accounts for most of the surprise."""

    def on_stats(*args) -> None:
        stats = next((arg for arg in args if isinstance(arg, str)), None)
        if stats:
            _report(stats)

    bpy.app.handlers.render_stats.append(on_stats)


def main() -> None:
    args = _parse_args()
    data = np.load(args.input)
    means, colors, opacities = data["means"], data["colors"], data["opacities"]
    axes, radii = data["axes"], data["radii"]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # The camera is framed first: every billboard is oriented by its rotation.
    if "camera_position" in data and "camera_rotation" in data:
        camera_euler = _frame_camera_from_capture(
            data["camera_position"], data["camera_rotation"], data.get("camera_intrinsics")
        )
    else:
        camera_euler = _frame_camera_auto_fit(means)

    mat = _make_material()
    ng = _build_billboard_instancer(mat, camera_euler)
    pc = _build_point_cloud(means, colors, opacities, axes, radii)
    obj = bpy.data.objects.new("GaussianCloud", pc)
    bpy.context.collection.objects.link(obj)
    modifier = obj.modifiers.new("Billboards", "NODES")
    modifier.node_group = ng

    scene = bpy.context.scene
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.filepath = args.output
    # Gaussian colours are data, not a photograph: a film view transform (AgX
    # by default) desaturates and darkens every one of them.
    scene.view_settings.view_transform = "Standard"
    _set_background(scene, args.background_color)
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.transparent_max_bounces = _TRANSPARENT_BOUNCES
        device = _enable_gpu_compute(scene)
    else:
        scene.render.engine = "BLENDER_EEVEE"
        device = "EEVEE"
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = args.samples

    _report(f"device {device}")
    _report(f"{len(means):,} gaussians at {args.width}x{args.height}, {args.samples} samples")
    _install_progress_reporter()
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
