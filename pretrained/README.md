# Teacher setup for Bark

Bark now uses the **working external Go1 teacher** from:

`D:\quadruped-rl-locomotion\models\2024-04-27_18-04-12=1_pos_ctrl_20mil_iter_walking_with_fast_steps\best_model.zip`

The goal of this repo is now simple:

1. generate rollout data from that teacher
2. train a supervised student to copy leg 3
3. train an RL student to control leg 3
4. compare which follows the teacher best

## Canonical entry points

```bash
PYTHONPATH=. python pretrained/load_teacher.py --steps 300 --render
PYTHONPATH=. python train/generate_teacher_data.py --steps 200000
PYTHONPATH=. python train/train_supervised.py --config configs/supervised_go1.yaml
PYTHONPATH=. python train/train_prosthetic_rl.py --config configs/prosthetic_rl_go1.yaml
PYTHONPATH=. python evaluate/compare.py --episodes 20
PYTHONPATH=. python -m postpro.render_teacher --steps 300 --fps 30
PYTHONPATH=. python -m postpro.render_students --steps 300 --fps 30
```

## Important note

This teacher uses the external Go1 env from `D:\quadruped-rl-locomotion` with:

- observation dim: **48**
- action dim: **12**
- student observation dim after masking leg 3: **39**
- student action dim: **3**

Bark no longer tries to make the old internal `(37,)` teacher work.
