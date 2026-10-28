# SPDX-FileCopyrightText: 2026 hyun2xyz
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build an animated SOMA character (and the scene's greybox cubes) from a MotionBricks recording.

The recording comes from recorder/mb_record.py in this repository. It holds the SOMA mesh in
skeleton space with its skin weights and inverse bind transforms, the local joint transforms of
every frame, and the scene's solid cubes. The armature's rest pose is the bind pose, so Blender's
armature deform reproduces MotionBricks' skinning (joint world x inverse bind x point) exactly.
"""

from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

FORMAT = "motionbricks-blender-npz"
FORMAT_VERSION = 1


@dataclass
class ImportOptions:
    start_frame: int = 1
    import_cubes: bool = True
    add_floor: bool = True
    add_camera: bool = True
    add_light: bool = True
    set_scene_range: bool = True


@dataclass
class ImportResult:
    collection: bpy.types.Collection
    armature: bpy.types.Object
    frames: int
    fps: int
    joints: int


def _xf_to_mat(x) -> Matrix:
    """px py pz qx qy qz qw -> 4x4 matrix."""
    m = Quaternion((float(x[6]), float(x[3]), float(x[4]), float(x[5]))).to_matrix().to_4x4()
    m.translation = Vector((float(x[0]), float(x[1]), float(x[2])))
    return m


def _forward_kinematics(local: np.ndarray, parents: np.ndarray) -> list[Matrix]:
    world = []
    for j, parent in enumerate(parents):
        m = _xf_to_mat(local[j])
        world.append(m if parent < 0 else world[parent] @ m)
    return world


def _material(name: str, color, roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = (*color, 1.0)
    return mat


def _checker_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("MB_greybox_floor")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.9
    coord = nodes.new("ShaderNodeTexCoord")
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 1.0
    checker.inputs["Color1"].default_value = (0.32, 0.32, 0.34, 1.0)
    checker.inputs["Color2"].default_value = (0.26, 0.26, 0.28, 1.0)
    links.new(coord.outputs["Object"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    mat.diffuse_color = (0.3, 0.3, 0.32, 1.0)
    return mat


def _box_mesh(name: str) -> bpy.types.Mesh:
    """Unit cube centred on the origin."""
    h = 0.5
    verts = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h), (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _link(collection: bpy.types.Collection, obj: bpy.types.Object) -> bpy.types.Object:
    collection.objects.link(obj)
    return obj


def load_recording(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        recording = {key: data[key] for key in data.files}
    found = str(recording["format"]) if "format" in recording else FORMAT
    version = int(recording["format_version"]) if "format_version" in recording else FORMAT_VERSION
    if found != FORMAT or version > FORMAT_VERSION:
        raise ValueError(f"{path.name} is not a supported MotionBricks recording ({found} v{version}).")
    if "points" not in recording:
        raise ValueError(f"{path.name} has no SOMA mesh. Only SOMA recordings are supported.")
    return recording


def import_recording(context, path: Path, options: ImportOptions) -> ImportResult:
    rec = load_recording(path)
    names = [str(n) for n in rec["joint_names"]]
    parents = rec["parents"].astype(int)
    joints = len(names)
    if any(parents[j] >= j for j in range(joints)):
        raise ValueError("Joint parents must come before their children.")
    to_runtime = _xf_to_mat(rec["model_to_runtime"])
    rest_target = [to_runtime @ _xf_to_mat(x).inverted() for x in rec["inverse_bind"]]
    poses = rec["poses"]
    frames = poses.shape[0]
    fps = int(rec["fps"])
    first = options.start_frame

    scene = context.scene
    collection = bpy.data.collections.new(f"MotionBricks {path.stem}")
    scene.collection.children.link(collection)

    # Armature: bone rest matrices = bind pose in MotionBricks world space (Z up, metres).
    arm_data = bpy.data.armatures.new(f"MB_{path.stem}_rig")
    arm_data.display_type = "STICK"
    armature = _link(collection, bpy.data.objects.new(f"MB_{path.stem}_rig", arm_data))
    armature.show_in_front = True
    previous_active = context.view_layer.objects.active
    for obj in context.selected_objects:
        obj.select_set(False)
    context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    heads = [m.translation.copy() for m in rest_target]
    children = [[c for c in range(joints) if parents[c] == j] for j in range(joints)]
    edit_bones = []
    for j in range(joints):
        bone = arm_data.edit_bones.new(names[j])
        lengths = [(heads[c] - heads[j]).length for c in children[j]]
        length = max(0.02, min([x for x in lengths if x > 1e-4], default=0.06))
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, length, 0.0)
        bone.matrix = rest_target[j]
        edit_bones.append(bone)
    for j in range(joints):
        if parents[j] >= 0:
            edit_bones[j].parent = edit_bones[parents[j]]
    bpy.ops.object.mode_set(mode="OBJECT")
    rest = [arm_data.bones[n].matrix_local.copy() for n in names]

    # Skinned body. Rest vertices = model_to_runtime x skeleton-space points.
    points = rec["points"].astype(np.float64)
    rotation = np.array(to_runtime.to_3x3())
    verts = points @ rotation.T + np.array(to_runtime.translation)
    mesh = bpy.data.meshes.new(f"MB_{path.stem}_body")
    mesh.from_pydata(verts.tolist(), [], rec["faces"].astype(int).tolist())
    mesh.update()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    body = _link(collection, bpy.data.objects.new(f"MB_{path.stem}_body", mesh))
    body.parent = armature
    groups = [body.vertex_groups.new(name=n) for n in names]
    indices, weights = rec["joint_indices"], rec["joint_weights"]
    for vertex in range(indices.shape[0]):
        for k in range(indices.shape[1]):
            joint, weight = int(indices[vertex, k]), float(weights[vertex, k])
            if joint >= 0 and weight > 0.0:
                groups[joint].add([vertex], weight, "ADD")
    modifier = body.modifiers.new("Armature", "ARMATURE")
    modifier.object = armature
    body.data.materials.append(_material("MB_body", (0.82, 0.82, 0.80), 0.6))

    # Keys. Pose matrix P = A x rest_target^-1 x rest, so deform = A x inverse bind.
    pose_bones = [armature.pose.bones[n] for n in names]
    for pose_bone in pose_bones:
        pose_bone.rotation_mode = "QUATERNION"
    rel_rest_inv = [
        (rest[j] if parents[j] < 0 else rest[parents[j]].inverted() @ rest[j]).inverted() for j in range(joints)
    ]
    rest_target_inv = [m.inverted() for m in rest_target]
    previous = [None] * joints
    roots = []
    for frame in range(frames):
        world = _forward_kinematics(poses[frame], parents)
        posed = [world[j] @ rest_target_inv[j] @ rest[j] for j in range(joints)]
        roots.append(world[0].translation.copy())
        for j, pose_bone in enumerate(pose_bones):
            local = posed[j] if parents[j] < 0 else posed[parents[j]].inverted() @ posed[j]
            location, quaternion, _ = (rel_rest_inv[j] @ local).decompose()
            if previous[j] is not None and quaternion.dot(previous[j]) < 0.0:
                quaternion.negate()
            previous[j] = quaternion
            pose_bone.location = location
            pose_bone.rotation_quaternion = quaternion
            pose_bone.keyframe_insert("location", frame=first + frame, group=names[j])
            pose_bone.keyframe_insert("rotation_quaternion", frame=first + frame, group=names[j])
    armature.animation_data.action.name = f"MB_{path.stem}"

    if options.import_cubes and "cube_xf" in rec:
        for name, xf, size, color in zip(rec["cube_names"], rec["cube_xf"], rec["cube_size"], rec["cube_color"]):
            cube = _link(collection, bpy.data.objects.new(f"MB_{name}", _box_mesh(f"MB_{name}")))
            cube.matrix_world = _xf_to_mat(xf) @ Matrix.Diagonal((*[float(s) for s in size], 1.0))
            cube.data.materials.append(_material(f"MB_{name}", [float(c) for c in color], 0.7))

    if options.add_floor:
        s = 100.0
        floor_mesh = bpy.data.meshes.new("MB_floor")
        floor_mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
        floor_mesh.update()
        floor = _link(collection, bpy.data.objects.new("MB_floor", floor_mesh))
        floor.data.materials.append(_checker_material())

    if options.add_light:
        sun = _link(collection, bpy.data.objects.new("MB_sun", bpy.data.lights.new("MB_sun", "SUN")))
        sun.data.energy = 3.0
        sun.data.angle = 0.15
        sun.rotation_euler = (0.9, 0.2, 0.6)

    if options.add_camera:
        camera = _link(collection, bpy.data.objects.new("MB_follow_camera", bpy.data.cameras.new("MB_follow_camera")))
        camera.data.lens = 35
        target = Vector(roots[0])
        for frame in range(frames):
            target = target.lerp(Vector(roots[frame]), 0.08)
            look = Vector((target.x, target.y, 0.9))
            camera.location = look + Vector((-4.2, -1.6, 1.4))
            camera.rotation_euler = (look - camera.location).to_track_quat("-Z", "Y").to_euler()
            camera.keyframe_insert("location", frame=first + frame)
            camera.keyframe_insert("rotation_euler", frame=first + frame)
        scene.camera = camera

    if options.set_scene_range:
        scene.render.fps = fps
        scene.render.fps_base = 1.0
        scene.frame_start = first
        scene.frame_end = first + frames - 1
        scene.frame_set(first)

    for obj in context.selected_objects:
        obj.select_set(False)
    armature.select_set(True)
    context.view_layer.objects.active = armature if armature else previous_active
    return ImportResult(collection, armature, frames, fps, joints)
