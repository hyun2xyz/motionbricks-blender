# SPDX-FileCopyrightText: 2026 hyun2xyz
# SPDX-License-Identifier: Apache-2.0

"""Walk a MotionBricks character along a course, with no motion data at all, and save an .npz
that the Blender add-on imports.

Run it from the root of your NVlabs/motionbricks checkout (v0.2.0), with the MotionBricks
environment (uv):
    uv run python path/to/mb_walk.py --course my_course.json --out my_course.npz

Course file (JSON, metres and degrees; yaw 0 faces -Y, 90 faces +X):
    {"start": [x, y, yaw], "waypoints": [[x, y], ...], "end_yaw": yaw or null, "speed": 1.2,
     "cubes": [{"name": "Box_A", "center": [x, y, z], "size": [w, d, h], "color": [r, g, b]}]}
The cubes are scenery for Blender only; the character does not react to them, so place the
waypoints around them.

MotionBricks needs target poses: "be in this pose, here". The official demos take them from
NVIDIA's sample motions, which are licensed for internal evaluation only. Here every target
comes from the SOMA bind pose that ships with the MotionBricks code (Apache-2.0), or from poses
the model itself generated in this run:
  1. "stand": the bind T-pose with the arms lowered by four rotations.
  2. One generation from stand to a stand 1.3 m ahead gives the first, rough steps. The four
     frames with the highest root speed become the walk template.
  3. The template walks a 10 m straight line; a new template is cut from the steady part of
     that walk. This repeats (--rounds) and the walk settles down.
  4. The course is walked with the same loop as MotionBricks' docs/examples/locomotion_loop.py
     (GeneratedMotionPlayer, place_target_poses, adjust_target_pose_speed), steering toward each
     waypoint, and ends on a stand target at the last one.
Walking, turning and stopping only. Not affiliated with or endorsed by NVIDIA.
"""

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

# The MotionBricks checkout goes first on the import path, so check that this is one (a stray numpy.py in some
# other folder would otherwise be imported).
if not all((Path.cwd() / name).is_dir() for name in ("motionbricks", "app")):
    sys.exit("Run this from the root of your motionbricks checkout: uv run python path/to/mb_walk.py ...")
sys.path.insert(0, str(Path.cwd()))

try:
    import numpy as np
    import warp as wp
    from scipy.spatial.transform import Rotation as R

    from app.runtime.common.rig import model_to_runtime
    from motionbricks.io import utils as io_utils
    from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, ModelRig
    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine
    from motionbricks.runtime.motion_player import GeneratedMotionPlayer
    from motionbricks.runtime.target_pose import TargetPoseInferenceParams, adjust_target_pose_speed, place_target_poses
except ImportError as exc:
    sys.exit(f"{exc}\nRun this from the motionbricks folder with: uv run python path/to/mb_walk.py ...")

FPS = 30
FORMAT = "motionbricks-blender-npz"
FORWARD = np.array([0.0, 1.0, 0.0])  # hips-local axis that points where the body faces
LEAD_SECONDS = 1.0  # how far ahead the moving target is placed
TRANSITION_EARLY_BY = 4
TURN_THRESHOLD = 15.0  # degrees of steering change that trigger a new generation
ARRIVE = 0.7  # metres; a waypoint counts as reached inside this radius


# --- poses -----------------------------------------------------------------------------------

def fk(local, parents):
    """World positions and rotations of every joint for local transforms [J,7] (x y z qx qy qz qw)."""
    rot, pos = [None] * len(parents), np.zeros((len(parents), 3))
    for j, p in enumerate(parents):
        r = R.from_quat(local[j, 3:7])
        if p < 0:
            rot[j], pos[j] = r, local[j, :3]
        else:
            rot[j], pos[j] = rot[p] * r, pos[p] + rot[p].apply(local[j, :3])
    return pos, rot


def aim_joint(local, parents, names, joint, child, direction):
    """Rotate `joint` so the bone from it to `child` points along `direction` (world space)."""
    j, c = names.index(joint), names.index(child)
    pos, rot = fk(local, parents)
    d0 = (pos[c] - pos[j]) / np.linalg.norm(pos[c] - pos[j])
    d1 = np.asarray(direction, float) / np.linalg.norm(direction)
    axis = np.cross(d0, d1)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        return
    delta = R.from_rotvec(axis / s * math.atan2(s, float(np.dot(d0, d1))))
    local[j, 3:7] = (rot[parents[j]].inv() * delta * rot[j]).as_quat()


def make_stand(bind, parents, names, m2r):
    """The bind T-pose in runtime space, with the arms lowered to the sides."""
    pose = bind.astype(np.float64).copy()
    q = R.from_quat(m2r.q)
    pose[0, :3] = q.apply(pose[0, :3]) + np.array(m2r.p)
    pose[0, 3:7] = (q * R.from_quat(pose[0, 3:7])).as_quat()
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        aim_joint(pose, parents, names, f"{side}Arm", f"{side}ForeArm", [0.16 * sx, -0.04, -1.0])
        aim_joint(pose, parents, names, f"{side}ForeArm", f"{side}Hand", [0.10 * sx, -0.30, -1.0])
    return pose


def place(pose, x, y, yaw_deg):
    """Move a pose [J,7] so its root stands at (x, y) facing yaw (0 = -Y, 90 = +X)."""
    out = pose.copy()
    yaw0 = math.atan2(*R.from_quat(pose[0, 3:7]).apply(FORWARD)[[1, 0]])
    want = math.atan2(-math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)))
    rz = R.from_euler("z", want - yaw0)
    out[0, :3] = rz.apply(pose[0, :3] * [0.0, 0.0, 1.0]) + [x, y, 0.0]
    out[0, 3:7] = (rz * R.from_quat(pose[0, 3:7])).as_quat()
    return out


def canonical(window):
    """Move a [4,J,7] window so its first root is at the ground origin and it travels along +X."""
    out = window.copy()
    roots = window[:, 0, :3]
    off = roots[-1, :2] - roots[0, :2]
    rz = R.from_euler("z", -math.atan2(off[1], off[0]))
    origin = np.array([roots[0, 0], roots[0, 1], 0.0])
    for k in range(len(window)):
        out[k, 0, :3] = rz.apply(window[k, 0, :3] - origin)
        out[k, 0, 3:7] = (rz * R.from_quat(window[k, 0, 3:7])).as_quat()
    return out


def fastest_window(motion, lo, hi):
    """Four consecutive frames with the highest ground speed, from the [lo, hi] part of motion."""
    a, b = int(len(motion) * lo), int(len(motion) * hi) - MODEL_POSES_PER_TOKEN
    speed = [np.linalg.norm(motion[i + 3, 0, :2] - motion[i, 0, :2]) for i in range(a, b)]
    i = a + int(np.argmax(speed))
    return motion[i:i + MODEL_POSES_PER_TOKEN].copy(), float(max(speed)) * FPS / 3


def to_wp(a):
    return wp.array(a.astype(np.float32), dtype=wp.transform)


# --- generation ------------------------------------------------------------------------------

class Walker:
    def __init__(self, engine, skeleton, stand):
        self.engine, self.sk, self.stand = engine, skeleton, stand
        self.settings = TargetPoseInferenceParams(transition_early_by_frames=TRANSITION_EARLY_BY, num_poses_to_use=4)
        self.generations = 0

    def first_steps(self, distance):
        """Stand at the origin facing +X, generate once toward a stand `distance` ahead."""
        engine = self.engine
        ctx = np.repeat(place(self.stand, 0.0, 0.0, 90.0)[None], MODEL_POSES_PER_TOKEN, axis=0)
        tgt = np.repeat(place(self.stand, distance, 0.0, 90.0)[None], MODEL_POSES_PER_TOKEN, axis=0)
        context, target, output = engine.create_input(self.sk), engine.create_input(self.sk), engine.create_output(self.sk)
        context.update_from_poses(to_wp(ctx), stream=engine.stream)
        target.update_from_poses(to_wp(tgt), stream=engine.stream)
        engine.generate_animation(context, target, output)
        engine.synchronize()
        self.generations += 1
        n = output.num_frames
        return output.poses[:n].numpy().reshape(n, -1, 7).astype(np.float64)

    def walk(self, template, speed, start, waypoints, end_yaw=None, max_seconds=120.0):
        """Walk from start through the waypoints and stop at the last one. Returns poses at 30 fps."""
        engine = self.engine
        tmpl = to_wp(canonical(template))
        adjust_target_pose_speed(tmpl, speed, stream=engine.stream)
        target_poses = wp.empty_like(tmpl)
        player = GeneratedMotionPlayer(engine, self.sk)
        current = to_wp(place(self.stand, *start))
        player.seed_history_from_pose(current)
        contacts = wp.zeros(4, dtype=wp.float32)
        dt = 1.0 / FPS
        shown, steer, wi, stop_left = [], None, 0, None
        for _ in range(int(max_seconds * FPS)):
            pos = current.numpy().reshape(-1, 7)[0, :3] * [1.0, 1.0, 0.0]
            goal = np.array([*waypoints[wi], 0.0])
            if wi < len(waypoints) - 1 and np.linalg.norm(goal - pos) < ARRIVE:
                wi += 1
                goal = np.array([*waypoints[wi], 0.0])
            dist = float(np.linalg.norm(goal - pos))
            direction = ((goal - pos) / max(dist, 1e-6)).astype(np.float32)
            if stop_left is None and wi == len(waypoints) - 1 and dist < speed * LEAD_SECONDS * 1.2:
                yaw = end_yaw if end_yaw is not None else math.degrees(math.atan2(direction[0], -direction[1]))
                player.generate(to_wp(np.repeat(place(self.stand, goal[0], goal[1], yaw)[None], 4, axis=0)), self.settings)
                self.generations += 1
                stop_left = player.time_to_transition + 0.5
            elif stop_left is None:
                turned = steer is not None and math.degrees(
                    math.acos(max(-1.0, min(1.0, float(direction @ steer))))) > TURN_THRESHOLD
                if steer is None or turned or player.time_to_transition <= dt:
                    ground = pos + direction * speed * LEAD_SECONDS
                    place_target_poses(wp.vec3(*ground), wp.vec3(*direction), tmpl, target_poses, stream=engine.stream)
                    player.generate(target_poses, self.settings)
                    self.generations += 1
                    steer = direction
            player.sample(dt, current, contacts)
            player.record_pose(dt, current)
            shown.append(current.numpy().reshape(-1, 7).astype(np.float64))
            if stop_left is not None:
                stop_left -= dt
                if stop_left <= 0:
                    break
        engine.synchronize()
        return np.stack(shown)


def save(path, human, names, parents, m2r, motion, cubes, source):
    override = human.override_poses["RelaxedHands"].numpy().reshape(len(names), 7)
    fingers = human.override_pose_masks["Fingers"].numpy() > 0.5
    motion = motion.astype(np.float32)
    motion[:, fingers] = override[fingers]
    mesh = human.skinned_meshes[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format=np.array(FORMAT),
        format_version=np.int32(1),
        source=np.array(json.dumps(source)),
        fps=np.int32(FPS),
        joint_names=np.array(names),
        parents=parents,
        inverse_bind=human.inverse_bind_transforms.numpy().reshape(-1, 7),
        model_to_runtime=np.array([*m2r.p, *m2r.q], dtype=np.float32),
        points=mesh.points.numpy(),
        faces=mesh.indices.numpy().reshape(-1, 3),
        joint_indices=mesh.joint_indices.numpy().reshape(mesh.num_points, -1),
        joint_weights=mesh.joint_weights.numpy().reshape(mesh.num_points, -1),
        poses=motion,
        cube_names=np.array([c["name"] for c in cubes]) if cubes else np.array([], dtype="<U1"),
        cube_xf=np.array([[*c["center"], 0, 0, 0, 1] for c in cubes], np.float32).reshape(-1, 7),
        cube_size=np.array([c["size"] for c in cubes], np.float32).reshape(-1, 3),
        cube_color=np.array([c["color"] for c in cubes], np.float32).reshape(-1, 3),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--course", type=Path, required=True, help="course JSON")
    parser.add_argument("--out", type=Path, required=True, help="output .npz for the Blender add-on")
    parser.add_argument("--rounds", type=int, default=2, help="straight-line walks that refine the walk template")
    parser.add_argument("--first-step", type=float, default=1.3, help="metres of the very first generation")
    parser.add_argument("--hold", type=float, default=1.0, help="seconds of standing still at the start and end")
    parser.add_argument("--keep-arms", action="store_true", help="keep the generated arm swing in the template "
                        "(by default the template gets the stand pose's relaxed arms)")
    args = parser.parse_args()
    course = json.loads(args.course.read_text(encoding="utf-8-sig"))
    speed = float(course.get("speed", 1.2))
    waypoints = [tuple(p[:2]) for p in course["waypoints"]]

    wp.init()
    wp.set_device("cuda:0")
    human = io_utils.get_soma_human_model()
    sk = human.skeleton
    names = list(sk.joint_names)
    parents = sk.parent_indices.numpy()
    m2r = model_to_runtime(ModelRig.SOMA)
    stand = make_stand(sk.reference_local_transforms.numpy().reshape(-1, 7), parents, names, m2r)
    arms = [i for i, n in enumerate(names) if n.startswith(("Left", "Right")) and
            any(k in n for k in ("Shoulder", "Arm", "Hand"))]

    def template_from(motion, lo, hi):
        window, v = fastest_window(motion, lo, hi)
        if not args.keep_arms:
            window[:, arms, 3:7] = stand[arms, 3:7]
        return window, round(v, 2)

    print("Loading MotionBricks (the first run builds a TensorRT plan, which takes a few minutes)...", flush=True)
    engine = MotionBricksSOMAEngine(SOMA_N1_FP32_ONNX_PATH, model_to_runtime=m2r)
    with engine, wp.ScopedStream(engine.stream):
        walker = Walker(engine, sk, stand)
        template, v = template_from(walker.first_steps(args.first_step), 0.25, 0.75)
        speeds = [v]
        print(f"Round 0: first steps toward a stand {args.first_step} m ahead, template speed {v} m/s", flush=True)
        for k in range(args.rounds):
            line = walker.walk(template, speed, (0.0, 0.0, 90.0), [(10.0, 0.0)])
            template, v = template_from(line, 0.35, 0.65)
            speeds.append(v)
            print(f"Round {k + 1}: 10 m straight walk, new template speed {v} m/s", flush=True)
        before = walker.generations
        sx, sy, syaw = course["start"]
        motion = walker.walk(template, speed, (sx, sy, syaw), waypoints, course.get("end_yaw"))
        course_gens = walker.generations - before
    hold = int(args.hold * FPS)
    motion = np.concatenate([np.repeat(motion[:1], hold, 0), motion, np.repeat(motion[-1:], hold, 0)])
    end = motion[-1, 0, :2]
    print(f"Course: {course_gens} generations, ended at ({end[0]:.2f}, {end[1]:.2f}), "
          f"last waypoint ({waypoints[-1][0]}, {waypoints[-1][1]})", flush=True)
    save(args.out, human, names, parents, m2r, motion, course.get("cubes", []), {
        "generator": "mb_walk.py",
        "targets": "SOMA bind pose with lowered arms, and walk templates cut from the model's own output; no motion data",
        "course": args.course.name, "speed": speed, "rounds": args.rounds, "template_speeds": speeds,
        "generations": walker.generations,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    print(f"Saved {len(motion)} frames ({len(motion) / FPS:.1f} s) at {FPS} fps to {args.out}")


if __name__ == "__main__":
    main()
