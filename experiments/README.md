# My training setup

I started with Pollen's baseline recipes to get the setup working and enter
Sprint, Sock Heist and Hill Climb. The [main README](../README.md#my-runs-so-far)
has my official times, model downloads and Arena replays from 2 November 2026.

Here's the setup and the commands I used if you want to try the same runs.

## What I used

- Linux x86_64 workstation, NVIDIA RTX 4090 with 24 GB VRAM.
- Python 3.12.14 and uv 0.12.7.
- Training source commit: `5cee942d8b434aafd395070facb659034aac3b5e`.
- Locked mjlab-microduck base: `1efd853be9fcdf050da24b5723cb74ada9a731d7`;
  Arena/Workshop 0.7.0. The repository's `uv.lock` pins the dependencies.
- 4096 parallel environments, training seed 1, TensorBoard logging.
  Sprint and Hill: 3000 training iterations; Sock: 1000.

You'll need [uv](https://docs.astral.sh/uv/getting-started/installation/) and a
working NVIDIA driver. This checks out the version I trained with:

```bash
git clone https://github.com/JackDanHollister/microduck-challenges
cd microduck-challenges
git checkout 5cee942d8b434aafd395070facb659034aac3b5e
uv sync --locked --group workshop
```

That commit is from before I added these notes. I also had ROS packages in
`PYTHONPATH`; if you have the same setup, put `env -u PYTHONPATH` before `uv`
in the commands below to avoid Python conflicts.

## Training the three events

These are the settings I used. Run one command at a time:

```bash
uv run --frozen train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 \
    --agent.max-iterations 3000 --agent.seed 1 --agent.logger tensorboard \
    --agent.run-name sprint-baseline-20261102

uv run --frozen train Mjlab-SockHeist6m-MicroDuck --env.scene.num-envs 4096 \
    --agent.max-iterations 1000 --agent.seed 1 --agent.logger tensorboard \
    --agent.run-name sock-baseline-20261102

uv run --frozen train Mjlab-HillClimb1m5-MicroDuck --env.scene.num-envs 4096 \
    --agent.max-iterations 3000 --agent.seed 1 --agent.logger tensorboard \
    --agent.run-name hill-baseline-20261102
```

Running the same recipe again won't give exactly the same weights. If you
want the actual models I entered, use the downloads linked in the main README.
Training runs are saved under `logs/rsl_rl/<event-folder>/<timestamp>_<run-name>/`.

The settings and environment code for each event are here:

- [Sprint](../src/microduck_challenges/sprint_2m): `env.py` and `challenge.toml`.
- [Sock Heist](../src/microduck_challenges/sock_heist_6m): `env.py` and `challenge.toml`.
- [Hill Climb](../src/microduck_challenges/hill_climb_1m5): `env.py` and `challenge.toml`.

## Watching the runs

```bash
uv run --frozen --group workshop workshop
```

I like being able to see what the duck is doing. The Workshop lets you open
a challenge, pick a run and watch the saved checkpoints, with scores using
the Arena rules. Choose **This machine** for local training.

**Publish and compete** uploads the selected policy publicly to Hugging Face
and enters it in the Arena.

I picked checkpoint 1000 for Sprint, 999 for Sock and 1000 for Hill. Each model
repo has `policy.onnx` with its observation normalisation included, the full
`checkpoint.pt`, a `manifest.json` and the training details. I keep the models
on Hugging Face and the code here rather than putting training logs and
videos into Git.

## A couple of other things I tried

- I tried increasing Sprint's upright reward from 2 to 3. At checkpoint 500
  it managed 18.450 s with 4/5 finishes. I stopped that run after 501 iterations;
  it wasn't a full-length comparison and I didn't submit it.
- I repeated Sock with seed 2 and the same recipe for 1000 iterations. Its
  best was checkpoint 900 at 13.060 s, 5/5. I entered the faster seed-1 model.
- Both Sock models finished 25/25 extra local timetables without catches,
  falls or timeouts, averaging 11.66 s and 13.10 s.

## Checks

- The supplied test suite: 181 passed, one marginal walker-distance assertion
  failed (0.29952 m against a strict 0.3 m cutoff after 150 steps). Extended
  walking/standing checks and two-environment CPU training smokes passed.
- All three submitted exports passed shape/finite-output checks and matched
  their normalized PyTorch actors over 128 inputs each, with maximum errors
  below 1e-6. Fresh publication exports were byte-identical to evaluated ones.

I've kept [the full results in JSON](results-2026-11-02.json), including the
five local attempts and the three public entries. Its earlier private-upload
plan is historical; the accepted entries use the public models linked above.

This is all simulation so far. I'm using Pollen's training code and recipes
as a starting point, and I'll add my own changes and tools here as I go.
