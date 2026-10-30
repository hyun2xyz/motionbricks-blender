# Instructions for AI coding agents (Codex, Claude Code, and others)

A user pointed you at this repository and asked something like:

> Use https://github.com/hyun2xyz/motionbricks-blender to make a MotionBricks demo in Blender and show me what it can do.

Your job: set up NVIDIA MotionBricks and this add-on on their machine, generate a walking motion with the walker
(no motion data needed), open it in Blender as a demo scene, then show a few things it can do. Work step by step,
report what you did after each step, and stop with a clear explanation if a requirement is missing.

## Ground rules

- **Ask before** installing software system-wide, and before the large downloads (about 12 GB in total).
- Do not change global settings (git config, PATH, Blender preferences beyond enabling this add-on) without asking.
- Do not commit, push, upload or post anything. Everything stays on the user's machine.
- Do not use MotionBricks' sample motions (`assets/motions/`) for anything the user may publish: they are licensed
  for internal evaluation only. The walker in this repository never reads them.

## 0. Check the machine

- An **NVIDIA RTX GPU** with a recent driver: `nvidia-smi` must list it. MotionBricks runs on TensorRT, so macOS
  and non-NVIDIA GPUs are not supported. If there is no NVIDIA GPU, stop and tell the user.
- Windows or Linux (x86-64), and about 12 GB of free disk space. Ask the user which folder to use.

## 1. Tools

Check each with its version command, and install only what is missing (ask first):

| Tool | Check | Windows | Linux |
|---|---|---|---|
| Git | `git --version` | `winget install Git.Git` | `sudo apt install git` |
| Git LFS | `git lfs version` | included with Git for Windows | `sudo apt install git-lfs` |
| uv | `uv --version` | `winget install astral-sh.uv` | see https://docs.astral.sh/uv/getting-started/installation/ |
| Blender 4.2+ | `blender --version` | `winget install BlenderFoundation.Blender` | https://www.blender.org/download/ |

Then run `git lfs install` once. On Windows, Blender is usually at
`C:\Program Files\Blender Foundation\Blender <version>\blender.exe`; use that full path if `blender` is not on PATH
(in PowerShell, put `&` before the quoted path: `& "C:\Program Files\...\blender.exe" --version`).

## 2. MotionBricks

In the chosen folder:

```sh
git clone --branch v0.2.0 https://github.com/NVlabs/motionbricks.git
cd motionbricks
git rev-parse HEAD
git lfs pull
uv sync
```

`git rev-parse HEAD` must print `54fc563aeaaf819a753d72cc9fdc6713989cdd43` (the v0.2.0 release). If it prints
anything else, stop and tell the user. The clone with model files is about 2.5 GB; `uv sync` builds a Python environment of about 4 GB. Each takes a few
minutes. If git reports "dubious ownership" (common on exFAT or network drives), run that one command with
`git -c safe.directory=* ...` instead of changing the global config.

## 3. This repository and the add-on

Next to the `motionbricks` folder:

```sh
git clone https://github.com/hyun2xyz/motionbricks-blender.git
curl -L -o motionbricks_importer-0.1.1.zip https://github.com/hyun2xyz/motionbricks-blender/releases/download/v0.1.1/motionbricks_importer-0.1.1.zip
```

In Windows PowerShell, write `curl.exe` (plain `curl` is an alias there). Then compare the zip's SHA-256, and if it
does not match, **stop**: delete the zip and tell the user.

- Linux: `echo "a013822cda017634bd5f690a8946760cf4190e9fd4e196bf9ae6d50a7d4fb2cf  motionbricks_importer-0.1.1.zip" | sha256sum -c` must print `OK`.
- PowerShell: `(Get-FileHash motionbricks_importer-0.1.1.zip -Algorithm SHA256).Hash -eq 'A013822CDA017634BD5F690A8946760CF4190E9FD4E196BF9AE6D50A7D4FB2CF'` must print `True`.

Install and enable the add-on (Blender stays closed):

```sh
blender --command extension install-file -r user_default -e motionbricks_importer-0.1.1.zip
```

## 4. Generate a walk

From the `motionbricks` folder:

```sh
uv run python ../motionbricks-blender/walker/mb_walk.py --course ../motionbricks-blender/walker/courses/zigzag.json --out demo_zigzag.npz
```

The very first run builds a TensorRT plan for the GPU, which takes a few minutes; later runs take seconds. It ends
with `Saved N frames ... to demo_zigzag.npz`.

## 5. Build and open the demo scene

```sh
blender --background --python ../motionbricks-blender/tools/make_demo.py -- --out demo.blend demo_zigzag.npz
blender demo.blend
```

Tell the user: press **Space** to play, **Numpad 0** (or the Camera View button in the N sidebar, MotionBricks
tab) for the follow camera, and **Render › Render Animation** to make a video. Add `--render demo_preview.mp4` to the
first command to also get a quick preview video without opening Blender.

## 6. Show what it can do

Pick two or three, explain each in one sentence, and open the result in Blender:

- **Change the course.** Copy `walker/courses/zigzag.json`, change `speed` (0.8 to 1.6 m/s looks natural), move the
  `waypoints`, or the `cubes` (scenery only; the character does not avoid them, so route the waypoints around them).
- **Draw a path.** `walker/courses/heart.json` traces a heart; render it from above.
- **Several characters.** Generate a few `.npz` files from different courses and pass them all to `make_demo.py`
  (the first file brings the floor, sun, boxes and camera; add `--spacing 2` to offset the others).
- **Speeds.** Generate the same straight course at 0.8, 1.2 and 1.6 m/s and compare them side by side.

Limits to mention: the walker only walks, turns and stops. Jumps, vaults and sitting need target poses (MotionBricks
"Smart Primitives"); the bundled ones come from the evaluation-only sample motions, so they are not used here.

## Licences (tell the user if they plan to publish)

- This add-on: GPL-3.0-or-later. Walker, recorder and tools: Apache-2.0.
- MotionBricks code: Apache-2.0. MotionBricks models: NVIDIA Open Model Agreement (commercial use allowed; it is
  not an open-source licence). MotionBricks sample motions: NVIDIA Sample Data Evaluation License, evaluation only.
- This project is not affiliated with or endorsed by NVIDIA.
