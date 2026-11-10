# Crown Roll

I wanted my Microduck to skate into a forward roll, go right over its head, land on both skates and carry on. I entered this 3.48-second act in Microduck Got a Talent as **JackDH**. [Watch the official replay](https://pollen-robotics-microduck-arena.hf.space/#/r/01M735W7J7182FS4RY6JDN) · [Watch the side view](https://huggingface.co/JackDH/microduck-crown-roll/resolve/740da39da8a8436ae1d01daab502a37f14ae0838/crown-roll.mp4)

I used my [Roller Sprint checkpoint 500](https://huggingface.co/JackDH/microduck-roller-sprint-20261110-best/tree/43e9b57825fbfe08cf4b510368b643ff9ae47d93) for the skating. The first version looked fine if I only checked the speed, but the duck was scraping along on its back. I changed the reward to check what touched the floor. In the submitted run, both skates catch together 45 milliseconds after the head comes off the ground.

I kept the roll I liked and trained 12 feedback gains on the CPU. They let the duck adjust its motors using its tilt, rotation speed and joint positions. I tested the exported policy on 64 fresh simulated starts. It landed cleanly and skated away on 62, compared with 58 on the same starts with the roll feedback switched off. [Results for all 64 starts](assessment.json) · [Submitted model and weights](https://huggingface.co/JackDH/microduck-crown-roll/tree/740da39da8a8436ae1d01daab502a37f14ae0838)

## Rebuild my entry

From the root of this fork, using Python 3.12 and the locked Workshop environment:

```bash
uv sync --locked --group workshop
uv run --frozen --group workshop python experiments/crown-roll/training/export.py \
  experiments/crown-roll/checkpoint.json logs/crown-roll/rebuilt
```

That writes the submitted ONNX policy and timeline to `logs/crown-roll/rebuilt/`. The exporter downloads the sprint model at the revision I used and checks its checksum. The [saved checkpoint](checkpoint.json) and [roll reference](training/reference.npz) are in this folder.

## Train the landing

To try the same training run with 12 CPU workers:

```bash
mkdir -p logs/crown-roll
uv run --frozen --group workshop python experiments/crown-roll/training/train.py \
  --workers 12 --output logs/crown-roll/my-checkpoint.json
uv run --frozen --group workshop python experiments/crown-roll/training/export.py \
  logs/crown-roll/my-checkpoint.json logs/crown-roll/my-policy
```

I started with feedback fitted to 128 simulated starts, then used 18 generations of cross-entropy search to tune it by episode reward. The reference and initial feedback are bundled with the [training code](training/train.py). The [timeline](timeline.json) cues the roll, and the [manifest](manifest.json) selects the roller rig.
