"""Train a projection-based lift policy using PPO with parallel envs.

Uses LiftCubeProjectionEnv: 13-dim obs derived from analytic 3D->2D projection
(fingertip keypoints + cube top-face corners + gripper opening).
No renderer needed — fully parallelisable with SubprocVecEnv.
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import subprocess
import sys

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.callbacks.plot_callback import PlotLearningCurveCallback
from src.envs.lift_cube_projection import LiftCubeProjectionEnv
from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env


class SuccessThresholdCallback(BaseCallback):
    """Stop training when success rate hits the target threshold for N consecutive evals
    AND N consecutive rollouts."""

    def __init__(self, threshold: float = 1.0, consecutive: int = 10,
                 eval_log_path: Path = None, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold
        self.consecutive = consecutive
        self.eval_log_path = eval_log_path
        self._rollout_srs: list[float] = []  # per-rollout success rates

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return
        successes = [ep["is_success"] for ep in buf if "is_success" in ep]
        if successes:
            self._rollout_srs.append(float(np.mean(successes)))

    def _on_step(self) -> bool:
        # Check eval success rate (from evaluations.npz)
        eval_ok = False
        if self.eval_log_path is not None:
            npz = self.eval_log_path / "evaluations.npz"
            if npz.exists():
                data = np.load(npz)
                if "successes" in data and len(data["successes"]) >= self.consecutive:
                    recent_srs = [float(ep.mean()) for ep in data["successes"][-self.consecutive:]]
                    eval_ok = all(sr >= self.threshold for sr in recent_srs)

        # Check rollout success rate (from ep_info_buffer across rollouts)
        rollout_ok = (
            len(self._rollout_srs) >= self.consecutive
            and all(sr >= self.threshold for sr in self._rollout_srs[-self.consecutive:])
        )

        if eval_ok and rollout_ok:
            if self.verbose:
                print(f"\n[SuccessThresholdCallback] Success rate >= {self.threshold:.0%} "
                      f"for last {self.consecutive} evals and rollouts — stopping training.")
            return False  # stops training
        return True


class BestModelEvalCallback(EvalCallback):
    """EvalCallback that saves best model on >= (not just >) improvement."""

    def _on_step(self) -> bool:
        result = super()._on_step()
        # SB3 saves when mean_reward > best_mean_reward; re-save if equal too
        if (self.last_mean_reward is not None
                and self.best_mean_reward is not None
                and self.last_mean_reward >= self.best_mean_reward
                and self.best_model_save_path is not None):
            self.model.save(Path(self.best_model_save_path) / "best_model")
        return result


class VideoEvalCallback(BaseCallback):
    """Runs eval_projection.py after each checkpoint to record a video."""

    def __init__(self, run_dir: Path, save_freq: int, episodes: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.save_freq = save_freq
        self.episodes = episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save current vec_normalize stats so eval can use them
            normalize_path = self.run_dir / "vec_normalize_eval.pkl"
            if hasattr(self.training_env, "save"):
                self.training_env.save(str(normalize_path))
            cmd = [
                sys.executable, "eval_projection.py",
                "--run", str(self.run_dir),
                "--episodes", str(self.episodes),
            ]
            if normalize_path.exists():
                cmd += ["--normalize", str(normalize_path)]
            if self.verbose:
                print(f"\n[VideoEvalCallback] Launching eval at step {self.num_timesteps}")
            subprocess.Popen(cmd)

            # Delete old eval videos, keeping only the 5 most recent
            videos = sorted(self.run_dir.glob("eval_*.mp4"), key=lambda p: p.stat().st_mtime)
            for old_video in videos[:-5]:
                old_video.unlink()

        return True


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict):
    env_class_name = env_cfg.get("env_class", "base")
    if env_class_name == "t1v2":
        EnvClass = LiftCubeProjectionT1V2Env
        extra_kwargs = {
            "cube_size_range": tuple(env_cfg.get("cube_size_range", [0.011, 0.019])),
            "cube_height_range": tuple(env_cfg.get("cube_height_range", [0.010, 0.035])),
            "fov_noise": env_cfg.get("fov_noise", 7.0),
        }
    else:
        EnvClass = LiftCubeProjectionEnv
        extra_kwargs = {}

    def _init():
        return EnvClass(
            render_mode=None,
            max_episode_steps=env_cfg.get("max_episode_steps", 200),
            action_scale=env_cfg.get("action_scale", 0.02),
            lift_height=env_cfg.get("lift_height", 0.08),
            hold_steps=env_cfg.get("hold_steps", 10),
            reward_type=env_cfg.get("reward_type", "dense"),
            reward_version=env_cfg.get("reward_version", "v11"),
            curriculum_stage=env_cfg.get("curriculum_stage", 1),
            lock_wrist=env_cfg.get("lock_wrist", True),
            cam_pos_noise=env_cfg.get("cam_pos_noise", 0.003),
            cam_euler_noise=env_cfg.get("cam_euler_noise", 0.05),
            obs_noise=env_cfg.get("obs_noise", 0.01),
            straddle_states_path=env_cfg.get("straddle_states_path", None),
            freeze_cube=env_cfg.get("freeze_cube", False),
            **extra_kwargs,
        )
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vision/curriculum_stage1_vision.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model .zip file (transfer learning)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps from config")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    ppo_cfg = config["ppo"]
    env_cfg = config["env"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise ValueError(f"Resume directory not found: {resume_dir}")
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / f"{timestamp}_resumed"
    else:
        resume_dir = None
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_dir / "config.yaml")

    if args.resume:
        with open(output_dir / "RESUME_INFO.txt", "w") as f:
            f.write(f"Resumed from: {resume_dir}\n")
            f.write(f"Timestamp: {timestamp}\n")

    pretrained = args.pretrained or train_cfg.get("pretrained")
    vec_normalize_path = None
    if args.resume:
        vec_normalize_path = resume_dir / "vec_normalize.pkl"

    n_envs = train_cfg.get("n_envs", 4)
    env_fns = [make_env(env_cfg) for _ in range(n_envs)]
    env = SubprocVecEnv(env_fns)

    if vec_normalize_path and vec_normalize_path.exists():
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = True
        print(f"Loaded normalization stats from {vec_normalize_path}")
    elif vec_normalize_path:
        raise ValueError(f"vec_normalize.pkl not found: {vec_normalize_path}")
    else:
        env = VecNormalize(
            env,
            norm_obs=env_cfg.get("normalize_obs", True),
            norm_reward=env_cfg.get("normalize_reward", True),
        )

    eval_env = DummyVecEnv([make_env(env_cfg)])
    if vec_normalize_path and vec_normalize_path.exists():
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=env_cfg.get("normalize_obs", True),
            norm_reward=False,
            training=False,
        )

    resume_step = 0
    if args.resume:
        checkpoints = list((resume_dir / "checkpoints").glob("*.zip"))

        def get_step_number(path):
            for part in path.stem.split("_"):
                if part.isdigit():
                    return int(part)
            return 0

        checkpoints = sorted(checkpoints, key=get_step_number)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            resume_step = get_step_number(latest_checkpoint)
            model = PPO.load(latest_checkpoint, env=env, device=device)
            model.tensorboard_log = str(output_dir / "tensorboard")
            print(f"Resumed from {latest_checkpoint} (step {resume_step})")
            with open(output_dir / "RESUME_INFO.txt", "a") as f:
                f.write(f"Checkpoint: {latest_checkpoint}\n")
                f.write(f"Resume step: {resume_step}\n")
        else:
            raise ValueError(f"No checkpoints found in {resume_dir / 'checkpoints'}")
    elif pretrained:
        pretrained_path = Path(pretrained)
        if not pretrained_path.exists():
            raise ValueError(f"Pretrained model not found: {pretrained_path}")
        model = PPO.load(pretrained_path, env=env, device=device)
        model.tensorboard_log = str(output_dir / "tensorboard")
        model.num_timesteps = 0
        model._episode_num = 0
        print(f"Loaded pretrained weights from {pretrained_path}")
        with open(output_dir / "PRETRAINED_INFO.txt", "w") as f:
            f.write(f"Pretrained from: {pretrained_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            verbose=1,
            seed=train_cfg["seed"],
            device=device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_lift_proj",
    )

    eval_callback = BestModelEvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg["eval_freq"],
        deterministic=True,
        render=False,
    )

    plot_callback = PlotLearningCurveCallback(
        run_dir=output_dir,
        save_freq=train_cfg["save_freq"],
        verbose=1,
        resume_step=resume_step,
    )

    video_eval_freq = train_cfg.get("video_eval_freq", train_cfg["save_freq"])
    video_callback = VideoEvalCallback(
        run_dir=output_dir,
        save_freq=video_eval_freq,
        episodes=1,
        verbose=1,
    )

    timesteps = args.timesteps if args.timesteps is not None else train_cfg["timesteps"]

    if args.resume:
        print(f"Loaded model num_timesteps: {model.num_timesteps}")
        reset_num_timesteps = False
        learn_timesteps = timesteps
        print(f"\nResuming vision training from step {model.num_timesteps}...")
    else:
        reset_num_timesteps = True
        learn_timesteps = timesteps
        print(f"\nStarting projection-based lift training (PPO, {n_envs} envs) for {timesteps} timesteps...")

    obs_dim = 16 if env_cfg.get("env_class") == "v11" else 13
    print(f"Obs space: {obs_dim}-dim (fingertips + cube corners + gripper{' + stage extras' if obs_dim == 16 else ''})")
    print(f"Output directory: {output_dir}")

    success_threshold = train_cfg.get("success_threshold", None)
    callbacks = [checkpoint_callback, eval_callback, plot_callback, video_callback]
    if success_threshold is not None:
        callbacks.append(SuccessThresholdCallback(
            threshold=success_threshold,
            consecutive=train_cfg.get("success_consecutive", 10),
            eval_log_path=output_dir / "eval_logs",
            verbose=1,
        ))

    model.learn(
        total_timesteps=learn_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
