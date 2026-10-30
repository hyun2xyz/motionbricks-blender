# SPDX-FileCopyrightText: 2026 hyun2xyz
# SPDX-License-Identifier: Apache-2.0

"""Build a ready-to-open Blender demo scene from one or more MotionBricks .npz files.

Needs the MotionBricks Importer add-on installed and enabled in Blender (see AGENTS.md):
    blender --background --python tools/make_demo.py -- --out demo.blend my_course.npz [more.npz ...]
Options:
    --render demo_preview.mp4   also render a quick EEVEE preview video (960x540)
    --spacing 2.0               offset each extra character along -Y by this many metres
The first file brings the floor, sun, greybox cubes and follow camera; later files add characters only.
Then open the scene with:  blender demo.blend   (press Space to play)
"""

import argparse
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("npz", nargs="+", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--render", type=Path)
ap.add_argument("--spacing", type=float, default=0.0)
args = ap.parse_args(argv)

if not hasattr(bpy.ops.import_scene, "motionbricks"):
    sys.exit("The MotionBricks Importer add-on is not enabled. Install the release zip first (AGENTS.md, step 3): "
             "blender --command extension install-file -r user_default -e motionbricks_importer-<version>.zip")

bpy.ops.wm.read_homefile(use_empty=True)  # keeps the preferences, so the add-on stays enabled
scene = bpy.context.scene
for i, path in enumerate(args.npz):
    path = path.resolve()
    first = i == 0
    before = set(bpy.data.objects)
    bpy.ops.import_scene.motionbricks(filepath=str(path), import_cubes=first, add_floor=first, add_light=first,
                                      add_camera=first, set_scene_range=first)
    if args.spacing and not first:
        for obj in set(bpy.data.objects) - before:
            if obj.parent is None:
                obj.location.y -= args.spacing * i
    print(f"imported {path.name}")

engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in engines else "BLENDER_EEVEE_NEXT"  # 4.2-4.5: _NEXT
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
args.out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()))
print(f"saved {args.out} (frames {scene.frame_start}-{scene.frame_end} at {scene.render.fps} fps)")

if args.render:
    scene.render.resolution_x, scene.render.resolution_y = 960, 540
    if hasattr(scene.render.image_settings, "media_type"):  # Blender 5.x
        scene.render.image_settings.media_type = "VIDEO"
    else:
        scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.filepath = str(args.render.resolve())
    bpy.ops.render.render(animation=True)
    print(f"rendered {args.render}")
