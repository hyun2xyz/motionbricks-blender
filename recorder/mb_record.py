# SPDX-FileCopyrightText: 2026 hyun2xyz
# SPDX-License-Identifier: Apache-2.0

"""Record a MotionBricks Scene Preview session to an .npz that the Blender add-on imports.

Run it from the root of your NVlabs/motionbricks checkout (v0.2.0), with the MotionBricks
environment (uv). Two modes:

  Play (default): the normal Scene Preview window opens. Walk with W A S D, run with Space,
  walk into objects to interact. Close the window to save.
      uv run python path/to/mb_record.py --scene assets/scenes/soma_gym.scene.json --out my_walk.npz

  Script: no window. Keys follow a timeline and every frame advances exactly 1/fps seconds,
  so the same command always records the same motion.
      uv run python path/to/mb_record.py --scene ... --out ... --script "0:,1:w,6:w+space,9:w" --seconds 14

Only SOMA (human) scenes are supported. The file holds the SOMA mesh from your MotionBricks
checkout, the joint transforms of every frame and the scene's solid cubes.
Not affiliated with or endorsed by NVIDIA.
"""

import argparse
import datetime
import json
import sys
import time as _time
import types
from pathlib import Path

# The MotionBricks checkout goes first on the import path, so check that this is one (a stray numpy.py in some
# other folder would otherwise be imported).
if not all((Path.cwd() / name).is_dir() for name in ("motionbricks", "app")):
    raise SystemExit("Run this from the root of your motionbricks checkout with 'uv run python ...'.")
sys.path.insert(0, str(Path.cwd()))

try:
    import numpy as np
    import warp as wp

    import newton.examples
    import newton.viewer

    import app.scene_preview as sp_mod
    from app.runtime.common.rig import model_to_runtime
    from app.utils.cli import add_model_precision_argument, configure_logging, create_gl_parser
    from app.utils.model_loading import require_tensorrt
    from motionbricks import __version__ as motionbricks_version
    from motionbricks.io import utils as io_utils
    from motionbricks.model.contract import ModelRig
except ImportError as error:  # pragma: no cover - guidance for a wrong working directory
    raise SystemExit(
        f"{error}\nRun this from the root of your motionbricks checkout with 'uv run python ...'."
    ) from error

FORMAT = "motionbricks-blender-npz"
FORMAT_VERSION = 1


def parse_script(text: str) -> list[tuple[float, set[str]]]:
    """'0:,1:w,6:w+space' -> [(0.0, set()), (1.0, {'w'}), (6.0, {'w', 'space'})]."""
    steps = []
    for item in text.split(","):
        at, _, keys = item.partition(":")
        steps.append((float(at), {k for k in keys.split("+") if k}))
    return sorted(steps, key=lambda step: step[0])


def keys_at(steps, t: float) -> set[str]:
    current = set()
    for at, keys in steps:
        if t >= at:
            current = keys
    return current


def slerp(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    """Per-joint slerp of [J, 4] xyzw quaternions."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(theta)
    small = sin < 1e-6
    a = np.where(small, 1.0 - w, np.sin((1.0 - w) * theta) / np.where(small, 1.0, sin))
    b = np.where(small, w, np.sin(w * theta) / np.where(small, 1.0, sin))
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(times: list[float], poses: list[np.ndarray], fps: int) -> np.ndarray:
    """Resample poses shown at irregular times to a steady fps."""
    times = np.asarray(times)
    out = []
    t = times[0]
    i = 0
    while t <= times[-1] + 1e-9:
        while i + 1 < len(times) and times[i + 1] < t:
            i += 1
        j = min(i + 1, len(times) - 1)
        span = times[j] - times[i]
        w = 0.0 if span <= 0 else float(np.clip((t - times[i]) / span, 0.0, 1.0))
        a, b = poses[i], poses[j]
        pose = np.empty_like(a)
        pose[:, :3] = (1.0 - w) * a[:, :3] + w * b[:, :3]
        pose[:, 3:] = slerp(a[:, 3:], b[:, 3:], w)
        out.append(pose)
        t += 1.0 / fps
    return np.stack(out).astype(np.float32)


def transforms_to_np(transforms) -> np.ndarray:
    return np.array([[*t.p, *t.q] for t in transforms], dtype=np.float32)


def main() -> None:
    parser = create_gl_parser()
    add_model_precision_argument(parser)
    parser.add_argument("--scene", type=Path, required=True, help="SOMA scene document (.scene.json).")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz for the Blender add-on.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate of the saved motion.")
    parser.add_argument("--script", default=None, help="Key timeline for a windowless recording, e.g. '0:,1:w,6:w+space'.")
    parser.add_argument("--seconds", type=float, default=14.0, help="Length of a --script recording.")
    parser.add_argument("--show-poses", action="store_true", help="Draw context and target poses in the window.")
    args, _ = parser.parse_known_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists. Choose a new file name.")
    out.parent.mkdir(parents=True, exist_ok=True)

    human = io_utils.get_soma_human_model()
    skeleton = human.skeleton
    override = human.override_poses["RelaxedHands"].numpy().reshape(skeleton.num_joints, 7)
    fingers = human.override_pose_masks["Fingers"].numpy() > 0.5
    times: list[float] = []
    poses: list[np.ndarray] = []

    if args.script is None:
        viewer, newton_args = newton.examples.init(parser)
        configure_logging(newton_args)
    else:
        clock = types.SimpleNamespace(t=0.0)
        sp_mod.time = types.SimpleNamespace(perf_counter=lambda: clock.t, time=_time.time, sleep=_time.sleep)
        wp.init()
        viewer = newton.viewer.ViewerGL(width=640, height=360, headless=True)
        pressed: set[str] = set()
        viewer.is_key_down = lambda key: key in pressed
    require_tensorrt()
    preview = sp_mod.ScenePreview(viewer, args.scene, fp16=args.model_precision == "fp16")
    if preview._scene_session.definition.rig is not ModelRig.SOMA:
        raise SystemExit("Only SOMA (human) scenes can be recorded for Blender.")
    if args.show_poses:
        preview.show_context_poses = True
        preview.show_target_poses = True

    original_step = preview.step

    def recording_step() -> None:
        original_step()
        pose = preview._scene_session.character.current_pose.numpy().reshape(skeleton.num_joints, 7).copy()
        pose[fingers] = override[fingers]
        times.append(float(preview.time))
        poses.append(pose)

    preview.step = recording_step
    session = preview._scene_session

    if args.script is None:
        print("Recording. Play with W A S D (Space runs). Close the window to save.")
        preview.run(newton_args.num_frames if newton_args.test else None)
        motion = resample(times, poses, args.fps) if len(poses) > 1 else np.stack(poses)
    else:
        steps = parse_script(args.script)
        for frame in range(int(args.seconds * args.fps)):
            clock.t = (frame + 1) / args.fps
            pressed.clear()
            pressed.update(keys_at(steps, frame / args.fps))
            preview._load_pending_scene()
            with wp.ScopedStream(preview._scene_session.character.engine.stream):
                preview.step()
                preview.render()
        preview.close()
        viewer.close()
        motion = np.stack(poses)

    cubes = session.definition.solid_cubes
    mesh = human.skinned_meshes[0]
    m2r = model_to_runtime(ModelRig.SOMA)
    np.savez_compressed(
        out,
        format=np.array(FORMAT),
        format_version=np.int32(FORMAT_VERSION),
        source=np.array(json.dumps({
            "motionbricks": motionbricks_version,
            "scene": args.scene.name,  # the name only: a full path would carry the user's folder names
            "mode": "script" if args.script else "play",
            "script": args.script,
            "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        })),
        fps=np.int32(args.fps),
        joint_names=np.array(skeleton.joint_names),
        parents=skeleton.parent_indices.numpy(),
        inverse_bind=human.inverse_bind_transforms.numpy().reshape(-1, 7),
        model_to_runtime=np.array([*m2r.p, *m2r.q], dtype=np.float32),
        points=mesh.points.numpy(),
        faces=mesh.indices.numpy().reshape(-1, 3),
        joint_indices=mesh.joint_indices.numpy().reshape(mesh.num_points, -1),
        joint_weights=mesh.joint_weights.numpy().reshape(mesh.num_points, -1),
        poses=motion,
        cube_names=np.array([c.name for c in cubes]) if cubes else np.array([], dtype="<U1"),
        cube_xf=transforms_to_np([c.transform for c in cubes]) if cubes else np.zeros((0, 7), np.float32),
        cube_size=np.array([[c.width, c.depth, c.height] for c in cubes], np.float32).reshape(-1, 3),
        cube_color=np.array([list(c.color) for c in cubes], np.float32).reshape(-1, 3),
    )
    print(f"Saved {motion.shape[0]} frames at {args.fps} fps to {out}")


if __name__ == "__main__":
    main()
