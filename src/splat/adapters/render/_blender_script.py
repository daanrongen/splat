# Runs inside Blender's own bundled Python (`blender --background --python
# <this file> -- --input ... --output ...`), not splat's venv - only
# bpy/numpy/stdlib are available here, no `splat` imports.
#
# Reads the .npz dump written by BlenderRenderBackend (means/colors/radius)
# and renders it as a native Blender point cloud colored by the flattened SH
# DC term, framed by a robust (median) center/radius - the same technique
# validated interactively over blender-mcp before this was wired up as a
# standalone script.

import argparse
import sys

import bpy
import numpy as np
from mathutils import Vector


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args(argv)


def _build_point_cloud(means: np.ndarray, colors: np.ndarray, radius: float):
    pc = bpy.data.pointclouds.new("GaussianCloud")
    n = len(means)
    pc.resize(n)
    pc.attributes["position"].data.foreach_set("vector", means.flatten().tolist())
    pc.attributes.new("radius", "FLOAT", "POINT")
    pc.attributes["radius"].data.foreach_set("value", np.full(n, radius, dtype=np.float32).tolist())

    col_attr = pc.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([colors, np.ones((n, 1), dtype=np.float32)], axis=1)
    col_attr.data.foreach_set("color", rgba.flatten().tolist())

    mat = bpy.data.materials.new("GaussianColor")
    mat.use_nodes = True  # deprecated for removal in Blender 6.0; harmless until then
    node_tree = mat.node_tree
    node_tree.nodes.clear()
    attr = node_tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    emit = node_tree.nodes.new("ShaderNodeEmission")
    output = node_tree.nodes.new("ShaderNodeOutputMaterial")
    node_tree.links.new(attr.outputs["Color"], emit.inputs["Color"])
    node_tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
    pc.materials.append(mat)

    obj = bpy.data.objects.new("GaussianCloud", pc)
    bpy.context.collection.objects.link(obj)
    return obj


def _frame_camera(means: np.ndarray) -> float:
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
    return median_radius


def main() -> None:
    args = _parse_args()
    data = np.load(args.input)
    means, colors = data["means"], data["colors"]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    median_radius = _frame_camera(means)
    _build_point_cloud(means, colors, median_radius * 0.03)

    scene = bpy.context.scene
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.filepath = args.output
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples

    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
