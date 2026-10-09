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

# A three-quarter view: the default only applies when nothing else framed the
# shot, and a dead-on front view of a single-view reconstruction shows nothing
# of its geometry.
_DEFAULT_AZIMUTH = 25.0
_DEFAULT_ELEVATION = 20.0
_FIT_MARGIN = 1.1  # room for the splats' own extent around their means

# Clouds are handed to Blender in the OpenGL convention (up is +Y), not
# Blender's own world convention (up is +Z). See `_look_at_euler`.
_UP = Vector((0.0, 1.0, 0.0))

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
    parser.add_argument("--azimuth", type=float, default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--look-at", default=None)
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


def _look_at_euler(position: Vector, target: Vector):
    """Orientation for a camera at `position` aimed at `target`, with +Y up.

    `Vector.to_track_quat` looks like the tool for this but resolves its up
    hint against Blender's world +Z, and a Y-up cloud framed that way comes out
    upside down. Building the basis explicitly is both correct and the only
    place in this script that needs to know which axis is up. Camera local axes
    are +X right, +Y up, -Z forward, so the columns go (right, up, backward).
    """
    backward = (position - target).normalized()
    right = _UP.cross(backward)
    if right.length < 1e-6:  # aimed straight along the up axis; any roll will do
        right = Vector((1.0, 0.0, 0.0))
    right.normalize()
    up = backward.cross(right)
    return Matrix((right, up, backward)).transposed().to_euler()


def _bounding_sphere(means: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of the cloud's bulk, trimmed of extreme floaters."""
    lo, hi = np.percentile(means, [1, 99], axis=0)
    margin = hi - lo
    bulk = means[((means >= lo - margin) & (means <= hi + margin)).all(axis=1)]
    center = (bulk.min(axis=0) + bulk.max(axis=0)) / 2
    return center, float(np.linalg.norm(bulk - center, axis=1).max()) or 1.0


def _fit_distance(radius: float, tan_half_fov: float, zoom: float) -> float:
    """Distance at which a sphere of `radius` fills a cone of the given half angle."""
    return radius / np.sin(np.arctan(tan_half_fov)) * _FIT_MARGIN / zoom


def _tan_half_fov(cam_data, width: int, height: int) -> float:
    """Half-angle tangent along the narrower image side."""
    fov_side = {"HORIZONTAL": width, "VERTICAL": height}.get(
        cam_data.sensor_fit, max(width, height)
    )
    return float(np.tan(cam_data.angle / 2)) * min(width, height) / fov_side


def _frame_camera_orbit(means: np.ndarray, args):
    """Spherical framing around the cloud's bounding sphere.

    Clouds arrive in the OpenGL convention, so up is +Y and azimuth 0 sits on
    +Z looking down -Z - the direction a default camera already points. The
    distance fits the bulk's bounding sphere to the field of view, so thin
    protrusions stay in frame at any scale (issues #83, #195).
    """
    center, radius = _bounding_sphere(means)
    if args.look_at is not None:
        center = np.array([float(part) for part in args.look_at.split(",")], dtype=np.float64)
        radius = float(np.linalg.norm(means - center, axis=1).max()) or 1.0

    cam_data = bpy.data.cameras.new("RenderCamera")
    if args.fov is not None:
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.lens_unit = "FOV"
        cam_data.angle = np.radians(args.fov)
    tan_half_fov = _tan_half_fov(cam_data, args.width, args.height)
    zoom = 1.0 if args.zoom is None else args.zoom
    distance = _fit_distance(radius, tan_half_fov, zoom) if args.distance is None else args.distance

    azimuth = np.radians(_DEFAULT_AZIMUTH if args.azimuth is None else args.azimuth)
    elevation = np.radians(_DEFAULT_ELEVATION if args.elevation is None else args.elevation)
    offset = distance * np.array(
        [
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
            np.cos(azimuth) * np.cos(elevation),
        ]
    )

    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    target = Vector(center.tolist())
    cam_obj.location = target + Vector(offset.tolist())
    euler = _look_at_euler(cam_obj.location, target)
    cam_obj.rotation_euler = euler
    bpy.context.scene.camera = cam_obj
    _report(
        f"camera orbit azimuth {np.degrees(azimuth):.1f} elevation "
        f"{np.degrees(elevation):.1f} distance {distance:.3f} "
        f"centre {center[0]:.3f},{center[1]:.3f},{center[2]:.3f}"
    )
    return euler


def _requested_orbit(args) -> bool:
    return any(
        value is not None
        for value in (
            args.azimuth,
            args.elevation,
            args.distance,
            args.zoom,
            args.fov,
            args.look_at,
        )
    )


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
    _report("camera capture pose")
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
    # An explicit viewpoint wins over the capture pose; without either, the
    # orbit defaults are a three-quarter view of the whole cloud.
    has_capture = "camera_position" in data and "camera_rotation" in data
    if has_capture and not _requested_orbit(args):
        camera_euler = _frame_camera_from_capture(
            data["camera_position"], data["camera_rotation"], data.get("camera_intrinsics")
        )
    else:
        camera_euler = _frame_camera_orbit(means, args)

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
