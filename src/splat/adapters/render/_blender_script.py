# Runs inside Blender's own bundled Python (`blender --background --python
# <this file> -- --input ... --output ...`), not splat's venv - only
# bpy/numpy/stdlib are available here, no `splat` imports.
#
# Reads the .npz dump written by BlenderBackend (means/colors/scales/rotations/
# opacities, plus an optional camera_position/camera_rotation/camera_intrinsics -
# already converted to Blender's axis convention on the splat-venv side) and
# renders each Gaussian as a true oriented, anisotropically-scaled, alpha-blended
# ellipsoid via geometry nodes, replacing v1's flat uniform-radius point splats
# (see issue #50 for that scope cut, #58/#60 for this one).
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


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=["cycles", "eevee"], default="cycles")
    return parser.parse_args(argv)


def _build_point_cloud(means, colors, scales, rotations, opacities):
    n = len(means)
    pc = bpy.data.pointclouds.new("GaussianCloud")
    pc.resize(n)
    pc.attributes["position"].data.foreach_set("vector", means.flatten().tolist())

    pc.attributes.new("pt_scale", "FLOAT_VECTOR", "POINT").data.foreach_set(
        "vector", scales.flatten().tolist()
    )
    pc.attributes.new("pt_rotation", "QUATERNION", "POINT").data.foreach_set(
        "value", rotations.flatten().tolist()
    )
    pc.attributes.new("pt_opacity", "FLOAT", "POINT").data.foreach_set("value", opacities.tolist())

    col_attr = pc.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([colors, np.ones((n, 1), dtype=np.float32)], axis=1)
    col_attr.data.foreach_set("color", rgba.flatten().tolist())
    return pc


def _make_material():
    mat = bpy.data.materials.new("GaussianMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.5
    attr_col = nt.nodes.new("ShaderNodeAttribute")
    attr_col.attribute_name = "Col"
    attr_op = nt.nodes.new("ShaderNodeAttribute")
    attr_op.attribute_name = "pt_opacity"
    nt.links.new(attr_col.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(attr_op.outputs["Fac"], bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.blend_method = "HASHED"  # EEVEE preview only; Cycles alpha is unaffected by this setting
    return mat


def _build_ellipsoid_instancer(mat, subdivisions=1):
    ng = bpy.data.node_groups.new("EllipsoidInstancer", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    group_in = ng.nodes.new("NodeGroupInput")
    group_out = ng.nodes.new("NodeGroupOutput")

    ico = ng.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = subdivisions

    named_scale = ng.nodes.new("GeometryNodeInputNamedAttribute")
    named_scale.data_type = "FLOAT_VECTOR"
    named_scale.inputs["Name"].default_value = "pt_scale"

    named_rot = ng.nodes.new("GeometryNodeInputNamedAttribute")
    named_rot.data_type = "QUATERNION"
    named_rot.inputs["Name"].default_value = "pt_rotation"

    named_col = ng.nodes.new("GeometryNodeInputNamedAttribute")
    named_col.data_type = "FLOAT_COLOR"
    named_col.inputs["Name"].default_value = "Col"

    named_op = ng.nodes.new("GeometryNodeInputNamedAttribute")
    named_op.data_type = "FLOAT"
    named_op.inputs["Name"].default_value = "pt_opacity"

    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(group_in.outputs["Geometry"], inst.inputs["Points"])
    ng.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    ng.links.new(named_scale.outputs["Attribute"], inst.inputs["Scale"])
    ng.links.new(named_rot.outputs["Attribute"], inst.inputs["Rotation"])

    store_col = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store_col.data_type = "FLOAT_COLOR"
    store_col.domain = "INSTANCE"
    store_col.inputs["Name"].default_value = "Col"
    ng.links.new(inst.outputs["Instances"], store_col.inputs["Geometry"])
    ng.links.new(named_col.outputs["Attribute"], store_col.inputs["Value"])

    store_op = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store_op.data_type = "FLOAT"
    store_op.domain = "INSTANCE"
    store_op.inputs["Name"].default_value = "pt_opacity"
    ng.links.new(store_col.outputs["Geometry"], store_op.inputs["Geometry"])
    ng.links.new(named_op.outputs["Attribute"], store_op.inputs["Value"])

    realize = ng.nodes.new("GeometryNodeRealizeInstances")
    ng.links.new(store_op.outputs["Geometry"], realize.inputs["Geometry"])

    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    ng.links.new(realize.outputs["Geometry"], set_mat.inputs["Geometry"])
    ng.links.new(set_mat.outputs["Geometry"], group_out.inputs["Geometry"])
    return ng


def _frame_camera_auto_fit(means: np.ndarray):
    median_center = np.median(means, axis=0)
    radii = np.linalg.norm(means - median_center, axis=1)
    median_radius = float(np.median(radii)) or 1.0
    center = Vector(median_center.tolist())

    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    distance = median_radius * 4.0
    cam_obj.location = center + Vector((distance * 0.7, -distance * 0.9, distance * 0.5))
    direction = (center - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam_obj


def _frame_camera_from_capture(position, rotation, intrinsics):
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = Vector(position.tolist())
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.rotation_quaternion = Matrix(rotation.tolist()).to_quaternion()
    if intrinsics is not None:
        fx, _fy, _cx, _cy, width, _height = intrinsics.tolist()
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.lens_unit = "FOV"
        cam_data.angle = 2.0 * np.arctan(width / (2.0 * fx))
    bpy.context.scene.camera = cam_obj


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
    means, colors = data["means"], data["colors"]
    scales, rotations, opacities = data["scales"], data["rotations"], data["opacities"]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    mat = _make_material()
    ng = _build_ellipsoid_instancer(mat)
    pc = _build_point_cloud(means, colors, scales, rotations, opacities)
    obj = bpy.data.objects.new("GaussianCloud", pc)
    bpy.context.collection.objects.link(obj)
    modifier = obj.modifiers.new("Ellipsoids", "NODES")
    modifier.node_group = ng

    if "camera_position" in data and "camera_rotation" in data:
        _frame_camera_from_capture(
            data["camera_position"], data["camera_rotation"], data.get("camera_intrinsics")
        )
    else:
        _frame_camera_auto_fit(means)

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 2.5
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (0.9, 0.0, 0.6)
    bpy.context.collection.objects.link(sun)

    scene = bpy.context.scene
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.filepath = args.output
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
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
