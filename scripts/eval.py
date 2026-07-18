"""评估训练好的触碰模型：统计成功率（距离<0.12m，与 env.success_threshold 对齐）、终止原因分布。"""
import os
import sys
import argparse
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "HEADLESS" not in os.environ:
    os.environ["HEADLESS"] = "0"

from stable_baselines3 import PPO
from diana_rl.envs.diana_touch_env import DianaTouchEnv


def evaluate(model_path, episodes=20, deterministic=True):
    env = DianaTouchEnv()
    model = PPO.load(model_path, env=env)

    stats = Counter()
    distances = []
    steps_list = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            ep_reward += reward

        stats[info.get("term_reason", "unknown")] += 1
        distances.append(info["distance"])
        steps_list.append(step)
        print(f"Ep {ep+1:2d}: {step:3d}步 | 距离={info['distance']:.3f}m | "
              f"原因={info['term_reason']} | 累计奖励={ep_reward:.1f}")

    print(f"\n{'='*50}")
    print(f"评估结果（{episodes} 回合）")
    for reason, count in stats.most_common():
        print(f"  {reason}: {count}/{episodes} ({count/episodes*100:.0f}%)")
    print(f"  平均距离: {np.mean(distances):.4f}m")
    print(f"  平均步数: {np.mean(steps_list):.1f}")
    print(f"  距离<{env.success_threshold}m (触碰成功): {sum(1 for d in distances if d < env.success_threshold)}/{episodes}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Diana 触碰模型评估")
    parser.add_argument("--model", type=str, default="checkpoints/diana_touch",
                        help="模型路径（不含 .zip 也可）")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true", help="关闭确定性推理")
    args = parser.parse_args()

    path = args.model
    if not path.endswith(".zip"):
        path += ".zip"

    evaluate(path, args.episodes, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
