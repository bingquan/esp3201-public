# Week 6 Rollout Bank

Pre-generated video frames so students **without a GPU** can do the Week 6
video lab (and the Week 6 mini-assignment Track B) on real generated output.

## What is committed here

A **compact verification sample**: two clips, 8 consecutive frames each,
downscaled to 384 px wide (~1.7 MB total). This is enough to confirm the
loader and the evaluation metrics work end-to-end on real frames.

```
clip01_arm_pick/   frame_000.png ... frame_007.png
clip02_ball_drop/  frame_000.png ... frame_007.png
manifest.json      generation settings + measured peak VRAM
```

## Regenerating the full bank

The committed sample is intentionally small. Generate the full-resolution,
full-length bank (and add more probe clips) with the starter script:

```bash
cd ../starter
CUDA_VISIBLE_DEVICES=0 python generate_rollout_bank.py --out ../rollout_bank --clips 4 --frames 24
```

Run it on any GPU (or a Colab T4); it writes PNG frame sequences plus a
`manifest.json` and reports measured peak VRAM.

## Verified configuration

These frames were generated with `cerspense/zeroscope_v2_576w` in fp16 with
CPU offload — see `manifest.json` and `../starter/pinned_models.md` for the
measured peak VRAM and settings.
