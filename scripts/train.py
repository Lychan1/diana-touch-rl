"""训练 Diana 左臂触碰目标方块（V8 最终版），PPO，每 10k 步存检查点，支持断点续训。"""
import os
os.environ["HEADLESS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from diana_rl.envs.diana_touch_env import DianaTouchEnv
from diana_rl.utils.monitor import TrainMonitor


def main():
    tensorboard_log = "./logs/"
    checkpoint_path = "./checkpoints/"
    model_path = "checkpoints/diana_touch"

    env = DianaTouchEnv()

    if os.path.exists(model_path + ".zip"):
        print("[继续] 已存在模型，继续训练至 200K 步")
        model = PPO.load(model_path, env=env, tensorboard_log=tensorboard_log)
        remaining = 200_000 - model.num_timesteps
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=[
                    CheckpointCallback(
                        save_freq=10_000, save_path=checkpoint_path,
                        name_prefix="diana_touch",
                        save_replay_buffer=False, save_vecnormalize=False,
                    ),
                    TrainMonitor(check_freq=5000),
                ],
                tb_log_name="touch",
                reset_num_timesteps=False,
            )
            model.save(model_path)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 128]),
            tensorboard_log=tensorboard_log,
        )
        model.learn(
            total_timesteps=200_000,
            callback=[
                CheckpointCallback(
                    save_freq=10_000, save_path=checkpoint_path,
                    name_prefix="diana_touch",
                    save_replay_buffer=False, save_vecnormalize=False,
                ),
                TrainMonitor(check_freq=5000),
            ],
            tb_log_name="touch",
        )
        model.save(model_path)

    print(f"训练完成 → {model_path}.zip")


if __name__ == "__main__":
    main()
