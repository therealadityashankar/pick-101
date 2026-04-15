"""Collect qpos snapshots from successful T1 (straddle) episodes.

Runs the T1 best_model and saves terminal qpos arrays whenever the straddle
success condition fires. These snapshots are used to initialise T2 training
so the agent starts in a straddled position.

Usage:
    python collect_straddle_states.py \
        --model runs/lift_proj_t1_s3/<timestamp>/best_model/best_model.zip \
        --output data/straddle_states.npy \
        --n 200 \
        --normalize runs/lift_proj_t1_s3/<timestamp>/vec_normalize.pkl \
        --save-images  # optionally save sample renders to data/straddle_states_preview/
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube_projection import LiftCubeProjectionEnv


def make_env():
    env = LiftCubeProjectionEnv(
        max_episode_steps=150,
        action_scale=0.02,
        lift_height=0.08,
        hold_steps=10,
        reward_version="v_straddle",
        curriculum_stage=3,
        lock_wrist=True,
        cam_pos_noise=0.003,
        cam_euler_noise=0.05,
        obs_noise=0.01,
    )
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to T1 best_model.zip")
    parser.add_argument("--output", default="data/straddle_states.npy")
    parser.add_argument("--n", type=int, default=200, help="Number of snapshots to collect")
    parser.add_argument("--normalize", default=None, help="Path to VecNormalize pickle (optional)")
    parser.add_argument("--save-images", action="store_true", help="Save sample renders of collected states")
    parser.add_argument("--n-images", type=int, default=20, help="How many sample images to save (default 20)")
    args = parser.parse_args()

    # Separate render env for image saving (only created if needed)
    render_env = None
    if args.save_images:
        render_env = LiftCubeProjectionEnv(
            render_mode="rgb_array",
            max_episode_steps=150,
            action_scale=0.02,
            lift_height=0.08,
            hold_steps=10,
            reward_version="v_straddle",
            curriculum_stage=3,
            lock_wrist=True,
        )
        render_env.reset()
        image_dir = Path(args.output).parent / (Path(args.output).stem + "_preview")
        image_dir.mkdir(parents=True, exist_ok=True)
        print(f"Will save up to {args.n_images} preview images to {image_dir}/")

    vec_env = DummyVecEnv([make_env])
    if args.normalize and Path(args.normalize).exists():
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    snapshots = []
    obs = vec_env.reset()
    ep_steps = 0

    print(f"Collecting {args.n} straddle snapshots...")
    while len(snapshots) < args.n:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = vec_env.step(action)
        ep_steps += 1

        # Access the underlying env to get raw qpos
        raw_env: LiftCubeProjectionEnv = vec_env.envs[0]
        info = infos[0]

        if info.get("is_success", False):
            qpos = raw_env.data.qpos.copy()
            ctrl = raw_env.data.ctrl.copy()
            snapshots.append(np.concatenate([qpos, ctrl]))
            if len(snapshots) % 10 == 0:
                print(f"  {len(snapshots)}/{args.n}")

            # Save a render if requested and we haven't hit the image limit yet
            if render_env is not None and len(snapshots) <= args.n_images:
                import mujoco
                render_env.data.qpos[:] = qpos
                render_env.data.ctrl[:] = ctrl
                mujoco.mj_forward(render_env.model, render_env.data)
                img = render_env.render(camera="closeup")
                if img is not None:
                    from PIL import Image
                    Image.fromarray(img).save(image_dir / f"state_{len(snapshots):04d}.png")

        if done[0]:
            obs = vec_env.reset()
            ep_steps = 0

    vec_env.close()
    if render_env is not None:
        render_env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(snapshots))
    print(f"Saved {len(snapshots)} snapshots to {output_path}")

    # Render what T2 will actually see at reset (step 0) using the saved states
    if args.save_images:
        import mujoco
        from PIL import Image
        reset_image_dir = Path(args.output).parent / (Path(args.output).stem + "_reset_preview")
        reset_image_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rendering T2 reset positions to {reset_image_dir}/...")
        t2_env = LiftCubeProjectionEnv(
            render_mode="rgb_array",
            max_episode_steps=150,
            action_scale=0.02,
            lift_height=0.08,
            hold_steps=10,
            reward_version="v_grip",
            curriculum_stage=3,
            lock_wrist=True,
            straddle_states_path=str(output_path),
        )
        for i in range(min(args.n_images, len(snapshots))):
            t2_env.reset()
            img = t2_env.render(camera="closeup")
            if img is not None:
                Image.fromarray(img).save(reset_image_dir / f"reset_{i + 1:04d}.png")
        t2_env.close()
        print(f"Saved {min(args.n_images, len(snapshots))} reset preview images to {reset_image_dir}/")


if __name__ == "__main__":
    main()
