# Project TODO
## Ablations
Ablations are meant to be experiments that justify each and every architectural choice so that one can conretely say why a certain choice was made. The current approach is a more "reward focused" approach where we have adapted a particular reward function and weights to ensure a natural gait. Config present in `configs/go2_natural_gait_tf.json`. Ablations are basically there to justify why this method and how it compares against other methods.

### Full rough-terrain baseline (2000 epochs) — E003

Launched 2026-08-29 22:35 via `./run_baseline_rough_2000.sh`, which starts it in
a detached tmux session `baseline2000` (survives closing the terminal, and can be
attached to live).

- Config `configs/baseline_rough_2000.json` = `configs/ablation_baseline.json`
  with `num_epochs: 500 -> 2000` and `save_interval: 50 -> 100`. Every other key
  is byte-identical, so epochs 0-500 remain comparable to the ablation matrix.
- `scene_rough.xml`, depth camera on, AttnRes 2x `[4, 128]`, `attn_res_heads: 4`,
  `--vec_env_nums 16 --proc_nums 16`, seed 0, GPU 0 (RTX 3070).
- Tier-1 before launch: 2-epoch smoke run on the same config in scratch, rc=0,
  66 s and 71 s per epoch, `Running_Average_Rewards` 529.5 / 526.4, all 40
  `reward/*` and `diag/*` columns written to `log.csv`, checkpoints saved.
- Budget: ~70 s/epoch x 2000 = ~39 h.
- Attach: `tmux attach -t baseline2000` (detach with `Ctrl-b d`).
  Watch: `tail -f log_baseline2000/baseline_rough_2000.out`.
  Stop: `tmux kill-session -t baseline2000`.
- [ ] On completion: `scripts/compare_experiments.py --log-dir ./log_baseline2000`,
      replay `model_pf_best.pth` with `scripts/play.py`, write up as E003 in
      `report.tex`.

### `use_feet_gait` reward was never added to the total

- [x] Fixed in `envs/mujoco_env.py` (uncommitted at the time of the E003 launch):
      `r_feet_gait` was computed and logged but omitted from `total`, and
      `_compute_feet_gait_reward` recomputed foot contacts instead of reusing the
      ones already gathered. Verified by `scripts/verify_feet_gait.py`
      (`total(on) - total(off) == feet_gait`, PASS).
- All shipped configs currently set `use_feet_gait: false` (it overlaps
      `use_gait_phase`), so no completed run is affected by the bug.

### Replaying pre-gait-phase checkpoints (84-dim state)

`STATE_DIM` went 84 -> 86 when the 2-dim gait phase `[sin, cos]` was appended to
the END of the state block (`envs/mujoco_env.py:57,59,525`). Checkpoints trained
before that (`log_gait/natural_gait_tf-summer`, `log_gait/natural_gait_mlp`,
`log/go2_natural_gait_tf`, `log/ablation_attnres`) expect 84; everything under
`log_ablation/` expects 86.

- [x] `scripts/play.py` crashed on the transformer path for 84-dim checkpoints:
      it built the encoder at the env's `state_dim` and `load_state_dict` raised
      `size mismatch for encoder.base.seq_fcs.0.weight: [256, 84] vs [256, 86]`.
      Fixed by reading the checkpoint's own state dim off its first linear layer
      (`encoder.base.seq_fcs.0.weight`, or `base.seq_fcs.0.weight` for MLP),
      building the policy at that dim, and indexing the observation as
      `state[:ckpt_dim] ++ depth[env_state_dim:]`.
- The old MLP-only workaround (slice the first N dims of the whole observation)
      was correct only because a state-only obs has no image after it; on a
      transformer it would have silently dropped all 16384 depth dims. Replaced.
- Verified: `natural_gait_tf-summer` 3686.3 +/- 11.5, `natural_gait_mlp` 4196.5
      +/- 14.1, `ablation_baseline` 3940.0 +/- 98.4, `ablation_visual_pool` 5062,
      `ablation_stateonly` 4519 -- all 1000-step episodes, no falls. 86-dim
      checkpoints print no compatibility note.

- [x] `envs/builder.py:50` hard-coded `NormObsWithImg.state_shape = 84`. It was
      right only by luck: the gait phase is appended last, so the literal 84
      happened to still mean "proprio block". Now derived as
      `STATE_DIM - GAIT_PHASE_DIM`. Behaviour is unchanged by construction --
      the gait phase and depth still pass through unnormalised, and the three
      replay rewards above are bit-identical before and after the change, so
      the in-flight 2000-epoch run and every completed run stay comparable.
- Added `UnitreeMujocoGymEnv.proprio_dim` (84) alongside `state_dim` (86) and
      corrected the stale `(84)` docstrings on `state_dim` and
      `_build_observation`.

### Architecture Ablations

**Six arms, three seeds each (0, 1, 2) — 18 arm-seeds.** Every arm uses the same
seed set, the same `--vec_env_nums 16 --proc_nums 16`, and the same 500 epochs.

| arm | what it is | source |
|---|---|---|
| `ablation_baseline` | AttnRes, 2x `[4, 128]`, `attn_res_heads: 4` | `main`: **AttnRes** |
| `ablation_visual_pool` | visual encoder, tokens averaged, no attention layers | `main`: **Visual pool** |
| `ablation_stateonly` | state-only, no visual terrain features | `main`: **MLP** |
| `ablation_heads1` | `attn_res_heads: 1` | head-count sweep |
| `ablation_heads2` | `attn_res_heads: 2` | head-count sweep |
| `ablation_heads8` | `attn_res_heads: 8` | head-count sweep |

The first three are the architecture comparison `main`'s `TODO.md` asks for
(live: `origin/main` @ `7a4304d`, "Revise ablation section in TODO.md" — the
*local* `main` ref is 5 commits stale and still shows the older layout). The
head-count sweep comes from the previous revision of that TODO ("Experiment with
different `attn_res_heads`") and stays in the matrix: it justifies the head count
the AttnRes arm would otherwise pick arbitrarily.

`scripts/run_ablations.py` prints one summary section per bullet.

#### Matrix status (2026-08-22)

15 / 18 arm-seeds complete at 500 epochs (50 eval rows). `run_remaining_seeds.sh`
reported `ALL RUNS FINISHED after 42 h at 2026-08-20 11:39:31`; all five
non-`visual_pool` arms are done across seeds 0, 1, 2. (`stateonly` seed 0 has 52
rows rather than 50 — a resume artifact, two duplicated eval rows near the resume
point; de-duplicate on EPOCH when aggregating.)

| arm-seed | state |
|---|---|
| `baseline`, `stateonly`, `heads1`, `heads2`, `heads8` seeds 0,1,2 | complete (50 eval rows each) |
| `visual_pool` seed 0 | running since 2026-08-22 17:44, `run_visualpool_seeds.sh` |
| `visual_pool` seeds 1, 2 | queued in `run_visualpool_seeds.sh` |

`visual_pool` is the last arm. Launched 2026-08-22 17:44 after confirming the GPU
was idle; ~55 s/epoch at 500 epochs is ~7.5 h per seed, so ~23 h for all three
(faster than the ~10 h/seed of the AttnRes arms, as expected with no attention
layers). Watch with `tail -f log_ablation/visualpool.out`; stop with
`kill -- -$(cat log_ablation/visualpool.pgid)`.

Tier-1 check before launch: the arm was smoke-tested for 2 epochs on a throwaway
short config in scratch (rc=0), and the printed policy confirms
`visual_append_layers` is an empty `ModuleList` while `depth_visual_base` /
`depth_up_conv` remain — i.e. visual tokens present, attention over them removed,
which is the single variable this arm isolates against `ablation_baseline`.

**Bullet 1 — "Run a baseline training session (Tier 2/3)"**

- [x] `configs/ablation_baseline.json` — AttnRes, 2x `[4, 128]` layers,
      `attn_res_heads: 4`, depth camera on. Reference arm for every other bullet.

**Bullet 2 — "Experiment with different `attn_res_heads` in `LocoAttnResTransformer`"**

- [x] `configs/ablation_heads1.json` — `attn_res_heads: 1`
- [x] `configs/ablation_heads2.json` — `attn_res_heads: 2`
- [x] `configs/ablation_heads8.json` — `attn_res_heads: 8`
- 4 heads is the baseline arm, reused rather than retrained. `net.attn_res_heads`
      is the *only* key that differs across these four configs.
- `AttnResLayer` asserts `dim % num_heads == 0`; `encoder.visual_dim` is 64, so
      1/2/4/8 are all valid and each has been checked with a forward pass.

**Bullet 3 — "Abalate the effect of depth camera by training a state-only policy"**

- [x] `configs/ablation_stateonly.json` — `policy_type: "mlp"`,
      `get_image: false`, compared against `ablation_baseline`.
- Caveat: this arm cannot isolate the camera alone. `scripts/train.py:148-171`
      builds the transformer path with a hard-coded `visual_input_shape=(4,64,64)`,
      so a state-only policy is only reachable via `policy_type: mlp`, which drops
      the encoder and transformer with it. The gap therefore measures
      "vision + attention stack" vs "state-only MLP", not the camera in isolation.
      Isolating the camera would need a state-only transformer path in `train.py`.

**Bullet 4 — `main`'s "Visual pool" arm: visual encoder, tokens averaged, no
attention layers**

Straight from `origin/main`'s `TODO.md:10`. This branch's `report.tex` E002
("MLP vs. Visual Pool vs. AttnRes") already specifies the same arm, and `main`'s
`report.tex:70` describes the same reduction inside `LocoTransformer`.

- [ ] `configs/ablation_visual_pool.json` — `transformer_params: []` and no
      `attn_res_heads`, so `train.py` builds `GaussianContPolicyLocoTransformer`
      with zero layers: encoder tokens are mean-pooled and concatenated with the
      state token, no attention at all. 288,312 policy parameters (matches the
      288k in `report.tex` E002) vs 422k for the baseline.
- The arm `main` asks for and the one E002 already specifies; the file existed as
      `configs/ablation_visual_pool.json`, was renamed to `extra_visual_pool.json`
      in `2eedb0b` — which dropped it out of `run_ablations.py`'s `ablation_*.json`
      glob — and is now renamed back. Its contents are unchanged and differ from
      `ablation_baseline.json` on exactly `net.transformer_params` and
      `net.attn_res_heads`.
- Why it is worth a slot: `ablation_stateonly` compares vision+attention against
      a state-only MLP and so cannot say which half paid off. `visual_pool` keeps
      the camera and the encoder and removes only the attention stack, so
      `baseline - visual_pool` isolates the AttnRes layers and
      `visual_pool - stateonly` isolates the camera.
- `scripts/play.py:207` selects the non-AttnRes path from the absence of
      `attn_res_heads`, so `./play_ablation.sh visual_pool` replays it with no
      extra flag.
- Not yet run — `./run_visualpool_seeds.sh` queues seeds 0/1/2. It must start
      only after `run_remaining_seeds.sh` finishes; two concurrent arms at
      `--vec_env_nums 16` OOM the 8 GB RTX 3070.
- Open: mean vs. max reduction (`net.max_pool`) is a separate untested choice.
      Both arms above use `max_pool: false`. Not currently scheduled.

#### `main`'s "Things to keep in mind"

- [x] *Logging should prevent loss of progress.* Each arm-seed writes
      `log_ablation/<arm>/UnitreeMujocoGymEnv/<seed>/log.csv` incrementally;
      `run_ablations.py` reads finished and partial runs alike, so a crashed arm
      still contributes its epochs.
- [x] *Checkpoints at regular intervals.* `save_interval: 50`,
      `eval_interval: 10` in every ablation config, plus a `best` checkpoint.
      `train.py --resume` restarts from the latest one.
- [x] *Do not be fooled by high reward alone.* `run_ablations.py` prints
      `reward/foot_slip`, `reward/foot_clearance`, `reward/feet_air_time` and
      `diag/base_height` next to reward, compares on a tail mean rather than a
      single eval point, and ends by telling you to replay the `best` checkpoint
      (`play_ablation.sh`) before calling a winner. Replay is still a manual step
      — not yet done for any arm.

#### Can an existing checkpoint serve as the fixed baseline? — No.

Checked every `params.json` under `log*/` against `configs/ablation_baseline.json`:

- **Architecture is compatible.** `log_gait/natural_gait_tf-summer/.../model_pf_best.pth`
  loads into a freshly built `GaussianContPolicyLocoAttnResTransformer` under
  `strict=True` (79 tensors, `attn_res` layers present, `state_dim=84`,
  `act_dim=12`). The "Architecture Incompatible" note on `main` refers to older
  pre-AttnRes checkpoints (e.g. `ckpt_850.zip`), not these.
- **But every arch-matching run was trained on `scene_flat.xml`,** while the
  ablations run `scene_rough.xml`. Different terrain is a different task, so
  reusing one as the reference would confound every arm.
- **And none converged.** Eval epochs logged: `natural_gait_tf-summer` 38
  (reward still climbing steeply, 1797 and rising, stopped ~epoch 380/2000),
  `log/go2_natural_gait_tf` 6, `log/ablation_attnres` 1.

So the baseline has to be trained fresh on rough terrain as `ablation_baseline` —
which is exactly what bullet 1 is for. Run it first; the other arms are only
interpretable against it.

All arms run on `robot/go2/scene_rough.xml` (the natural-gait reference config
`configs/go2_natural_gait_tf.json` uses `scene_flat.xml`) and share its reward
weights, `use_*` flags, env settings and PPO hyperparameters. `num_epochs` is
500 across all arms — reduced locally from the 2000 that is committed at HEAD;
keep it uniform across arms whichever value is used.

`run_ablations.py` verifies the one-variable property at startup and exits
non-zero if an arm differs from the baseline on any key outside its whitelist.

#### Environment / hardware constraints

- Run with the **`resloco` conda env** (`conda run -n resloco python3 ...`). Base
  python has no `gymnasium`; `run_ablations.py` now preflights `sys.executable`
  and exits with instructions rather than failing per-arm.
- **Only GPU 0 (RTX 3070, 8 GB) is usable.** GPU 1 is a GTX 1080 (sm_61) and the
  installed torch 2.11+cu130 ships sm_75+ kernels only — a plain matmul on it
  raises `no kernel image is available for execution on the device`. Ablations
  cannot be parallelised across the two GPUs on this box; they run sequentially
  on GPU 0, or GPU 1 needs a torch build with sm_61 kernels.
- **One arm at a time on GPU 0.** A single arm at `--vec_env_nums 16` settles
  around 6.3 GB of the 7.64 GB usable; a second concurrent arm OOMs during
  startup (measured, not estimated). At `--vec_env_nums 8` an arm uses ~4.6 GB —
  still too much for two, and 2.9x slower per epoch (208 s vs 72 s), so 16 is
  strictly better given arms cannot overlap anyway.
- **Budget.** 72 s/epoch x 500 epochs = ~10 h per arm-seed. The full matrix
  (6 arms x 3 seeds, `ablation_visual_pool` included) is ~180 h ≈ 7.5 days of
  wall clock. Decide before launching the rest: fewer seeds, fewer epochs, or a
  bigger box.

Run them with:

```
python3 scripts/run_ablations.py --seeds 0 1 2   # 6 arms x 3 seeds, grouped summary
python3 scripts/run_ablations.py --filter visual_pool --seeds 0 1 2  # bullet 4 only
python3 scripts/run_ablations.py --filter heads  # bullet 2 only
```

`configs/extra_visual_pool.json` (mean-pooled tokens, no attention) is kept
outside the `ablation_*` glob because no bullet on `main` asks for it. Rename it
back to `configs/ablation_visual_pool.json` to fold it into the sweep again.

Things to keep in mind:
- [x] Logging should be nicely written and done to prevent loss of progress.
      Every eval epoch appends to `log.csv` (file reopened and closed per write,
      so an interrupted run keeps everything up to that epoch) and to
      TensorBoard; all `reward/*` and `diag/*` terms are logged per epoch.
- [x] Checkpoints need to be saved at regular intervals.
      `save_interval: 50` plus a `best` snapshot on every eval improvement and a
      `finish` snapshot; `scripts/run_ablations.py --resume` picks up from the
      latest checkpoint.
- [x] Do not be fooled by just "high rewards" or low loss values - the final gait
      must be practical without the Quadruped dragging across the floor or
      something. `run_ablations.py` and `compare_experiments.py` report
      converged `reward/foot_slip`, `reward/foot_clearance`,
      `reward/feet_air_time` and `diag/base_height` next to reward, and print a
      reminder to replay the `best` checkpoint of each arm with `scripts/play.py`
      before calling a winner.

- [ ] Remaining: actually run the sweep on a GPU box and write the results into
      `report.tex`.
- [ ] `log_ablation/ablation_attnres/` is an orphaned run dir from the old config
      names (no `log.csv`, never reached an eval epoch). Delete it before the
      real sweep so it is not mistaken for a result.

### Audit of the seed-0 baseline run (2026-08-13, during `log_ablation/baseline.out`)

Findings from auditing the live baseline run. None were acted on mid-sweep except
the logging fix, since changing the reward or hyperparameters now would make the
remaining arms uncomparable to the baseline already in flight.

- [x] `Running_Training_Average_Rewards` logged as NaN every eval epoch.
      `torchrl/collector/on_policy.py` only appended to `train_rews` on `dones`,
      never on the `max_episode_frames` path. With `max_episode_frames: 999`
      against `horizon: 1000`, episodes essentially always end by time limit, so
      the deque stayed empty and `np.mean([])` produced the NaN that tensorboardX
      warned about. Fixed in `on_policy.py` and the same latent bug in
      `torchrl/collector/base.py` (`VecCollector`, off-policy path, which also
      never reset `train_rew` on time-limit resets). Logging-only: gradients,
      checkpoints and eval are unaffected, so the seed-0 baseline stays valid.
- [ ] **`use_feet_gait: true` is a no-op.** `r_feet_gait` is computed
      (`envs/mujoco_env.py:757`) and logged (`:795`) but never added to `total` --
      every other gated term has an `if self.use_X: total += r_X` line at
      `:762-775` and this one does not. Verified against the log at epoch 220:
      logged total 2.02581, sum of all terms 2.78470, sum excluding feet_gait
      2.02580. Also `_compute_feet_gait_reward` scales by `gait_phase_weight`
      (`:615`) -- no `feet_gait_weight` exists -- and measures nearly the same
      trot-phase agreement as `r_gait_phase`, which is why the two logged series
      match to 3 decimals across all epochs. Decide after the sweep: either wire
      it in with its own weight and rerun all arms, or drop the flag. Do not
      change it mid-sweep.
- [ ] Eval collapsed at epochs 110-130 (`Running_Average_Rewards` 1788 -> 41.8 ->
      recovered) while training `reward/total` only dipped 1.89 -> 1.37 per step,
      which predicts a ~1370 return. A ~30x train/eval gap. Only structural
      difference is deterministic `pf.eval_act` vs stochastic `explore`; the obs
      normalizer is copied correctly. Investigate if it recurs.
- [ ] Model selection is noise-dominated. `best` is a running max over ~50 eval
      points on a metric that swings 41 <-> 2050, which systematically favours the
      noisiest arm. `run_ablations.py` now reports a `Tail` column (mean over the
      final 10% of evals) and aggregates on it instead of on the last point, and
      warns when only one seed is present. Run the sweep with >=3 seeds.
- [ ] Both optimizers are permanently grad-clip-saturated: `clip_grad_norm_(..., 0.5)`
      at `torchrl/algo/on_policy/ppo.py:73` and `:118`, against observed pre-clip
      norms of 1300-2200 (vf) and ~5 (pf). `vf_loss` climbed 5 -> 250 as returns
      grew, i.e. the critic never tracks. With `opt_epochs: 10` on a stale
      `target_pf` this explains `ratio/max` reaching 76810 at epoch 210. Shared
      across arms so the comparison is fair, but it caps absolute performance.
- [ ] `diag/base_height` sits at 0.259 against `target_height: 0.29` for 220+
      epochs and `reward/height` never improves off -0.042, despite
      `height_weight: 30` being the largest weight in the config. Either the
      target is unreachable in the learned gait or the height is read from a
      different frame than the target assumes.
- [ ] `_is_fallen` triggers only below 0.15 m CoM and `alive_reward` is 0.0, so
      termination never fires -- `eval_traj_length` is exactly 1000.0 at every
      eval. The ablation never exercises falling, and this is the upstream reason
      the train-reward deque was empty.
