import argparse
import json
import time
import warnings
from functools import partial

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from . import config
from .env import GlucoseEnv

ALGOS = {"ppo": PPO, "sac": SAC}

HYPERPARAMS = {
    "ppo": dict(n_steps=480, batch_size=960, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0)),
    "sac": dict(buffer_size=300_000, learning_starts=10_000, batch_size=256, learning_rate=3e-4, gamma=0.99,
                tau=0.005, train_freq=1, gradient_steps=4, policy_kwargs=dict(net_arch=[64, 64])),
}


def make_env(patients, seeds, rank, seed):
    warnings.filterwarnings("ignore")
    env = GlucoseEnv(patients, seeds)
    env.reset(seed=seed + rank)
    return Monitor(env)


def vec_env(patients, seeds, n, seed, subproc):
    fns = [partial(make_env, patients, seeds, i, seed) for i in range(n)]
    return VecMonitor(SubprocVecEnv(fns) if subproc and n > 1 else DummyVecEnv(fns))


def val_env(subproc):
    fns = [partial(make_env, [p], list(config.VAL_SEEDS), 0, 1000) for p in config.VAL_PATIENTS]
    return VecMonitor(SubprocVecEnv(fns) if subproc else DummyVecEnv(fns))


class SaveBest(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        self.parent.model.save(self.path)
        if self.verbose:
            print(f"nouveau meilleur modèle -> {self.path} (score {self.parent.best_mean_reward:.1f})")
        return True


def train(algo, timesteps, n_envs, eval_freq, seed, subproc=True, device="cuda"):
    set_random_seed(seed)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = config.MODELS_DIR / f"{algo}_logs"
    env = vec_env(config.TRAIN_PATIENTS, list(config.TRAIN_SEEDS), n_envs, seed, subproc)
    eval_env = val_env(subproc)
    model = ALGOS[algo]("MlpPolicy", env, seed=seed, verbose=0, device=device, **HYPERPARAMS[algo])
    model.set_logger(configure(str(log_dir), ["csv", "stdout"]))
    best_path = config.MODELS_DIR / f"{algo}_best.zip"
    callback = EvalCallback(
        eval_env,
        callback_on_new_best=SaveBest(best_path),
        n_eval_episodes=len(config.VAL_PATIENTS) * 2,
        eval_freq=max(eval_freq // n_envs, 1),
        log_path=str(log_dir),
        deterministic=True,
        verbose=1,
    )
    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(config.MODELS_DIR / f"{algo}_last.zip")
    if not best_path.exists():
        model.save(best_path)
    summary = {"algo": algo, "timesteps": timesteps, "n_envs": n_envs, "seed": seed, "minutes": round((time.time() - start) / 60, 1),
               "best_val_reward": float(callback.best_mean_reward), "hyperparams": {k: str(v) for k, v in HYPERPARAMS[algo].items()}}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    env.close()
    eval_env.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS), default="ppo")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-subproc", action="store_true")
    args = parser.parse_args()
    summary = train(args.algo, args.timesteps, args.n_envs, args.eval_freq, args.seed, not args.no_subproc, args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
