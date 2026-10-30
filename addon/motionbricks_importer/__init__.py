# SPDX-FileCopyrightText: 2026 hyun2xyz
# SPDX-License-Identifier: GPL-3.0-or-later

"""MotionBricks Importer (unofficial): bring NVIDIA MotionBricks motion recordings into Blender.

File > Import > MotionBricks (.npz), or the MotionBricks tab in the 3D Viewport sidebar (N).
Make the .npz with walker/mb_walk.py (or recorder/mb_record.py) from
https://github.com/hyun2xyz/motionbricks-blender.
Not affiliated with or endorsed by NVIDIA.
"""

import time
from pathlib import Path

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import importer


class IMPORT_SCENE_OT_motionbricks(bpy.types.Operator, ImportHelper):
    """Import a MotionBricks recording (.npz) as an animated SOMA character"""

    bl_idname = "import_scene.motionbricks"
    bl_label = "Import MotionBricks"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".npz"
    filter_glob: StringProperty(default="*.npz", options={"HIDDEN"})

    start_frame: IntProperty(name="Start Frame", default=1, min=0, description="Frame of the first recorded pose")
    import_cubes: BoolProperty(name="Greybox Cubes", default=True, description="Add the scene's solid cubes")
    add_floor: BoolProperty(name="Checker Floor", default=True, description="Add a 200 m greybox checker floor")
    add_light: BoolProperty(name="Sun Light", default=True, description="Add a soft sun light")
    add_camera: BoolProperty(name="Follow Camera", default=True, description="Add a camera that follows the character")
    set_scene_range: BoolProperty(
        name="Set Frame Range and FPS", default=True, description="Match the scene's frame range and FPS to the recording"
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "start_frame")
        layout.prop(self, "set_scene_range")
        column = layout.column(heading="Add")
        column.prop(self, "import_cubes")
        column.prop(self, "add_floor")
        column.prop(self, "add_light")
        column.prop(self, "add_camera")

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        path = Path(self.filepath)
        options = importer.ImportOptions(
            start_frame=self.start_frame,
            import_cubes=self.import_cubes,
            add_floor=self.add_floor,
            add_camera=self.add_camera,
            add_light=self.add_light,
            set_scene_range=self.set_scene_range,
        )
        started = time.perf_counter()
        try:
            result = importer.import_recording(context, path, options)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            self.report({"ERROR"}, f"MotionBricks import failed: {error}")
            return {"CANCELLED"}
        seconds = time.perf_counter() - started
        context.scene.motionbricks_last_import = (
            f"{path.name}: {result.frames} frames at {result.fps} fps, {result.joints} joints"
        )
        self.report({"INFO"}, f"Imported {context.scene.motionbricks_last_import} in {seconds:.1f} s")
        return {"FINISHED"}


class VIEW3D_PT_motionbricks(bpy.types.Panel):
    bl_label = "MotionBricks"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MotionBricks"

    def draw(self, context):
        layout = self.layout
        layout.operator(IMPORT_SCENE_OT_motionbricks.bl_idname, text="Import Motion (.npz)", icon="IMPORT")
        last = context.scene.motionbricks_last_import
        if last:
            box = layout.box()
            box.label(text="Last import", icon="ARMATURE_DATA")
            box.label(text=last)
        row = layout.row(align=True)
        row.operator("screen.animation_play", text="Play", icon="PLAY")
        if context.scene.camera:
            row.operator("view3d.view_camera", text="Camera View", icon="VIEW_CAMERA")
        layout.separator()
        layout.label(text="Make .npz files with mb_walk.py", icon="INFO")


def menu_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_motionbricks.bl_idname, text="MotionBricks (.npz)")


classes = (IMPORT_SCENE_OT_motionbricks, VIEW3D_PT_motionbricks)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.motionbricks_last_import = StringProperty(name="Last MotionBricks import", default="")
    bpy.types.TOPBAR_MT_file_import.append(menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    del bpy.types.Scene.motionbricks_last_import
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
