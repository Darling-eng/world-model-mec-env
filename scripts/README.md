# Script Entrypoints

## Evaluate heuristic baselines

Run all heuristic baselines:

```bash
python eval_baselines.py --episodes 10 --seed 7
```

Run one baseline at a time:

```bash
python eval_baselines.py --policy random --episodes 10 --seed 7
python eval_baselines.py --policy local_only --episodes 10 --seed 7
python eval_baselines.py --policy best_uplink --episodes 10 --seed 7
python eval_baselines.py --policy largest_queue --episodes 10 --seed 7
```

Save aggregate metrics:

```bash
python eval_baselines.py --episodes 50 --seed 7 --output outputs/baselines.csv
python eval_baselines.py --episodes 50 --seed 7 --output outputs/baselines.jsonl
```

## Train lightweight PPO with MEC

This entrypoint is a dependency-free PPO smoke baseline built with NumPy. It is intended for short training checks before moving to a full DRL library baseline.

```bash
python scripts/run_ppo_mec.py --steps 2000 --rollout-steps 256 --eval-episodes 5 --seed 7
```

## Run DreamerV3 with MEC in Colab

Example:

```bash
python /content/world-model-mec-env/scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 100
```
