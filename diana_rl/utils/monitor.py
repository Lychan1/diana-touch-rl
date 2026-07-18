"""训练监控回调：检测平台期 / 退化 / 熵崩溃 / 价值失效等异常并打印中文警告。"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainMonitor(BaseCallback):
    """每 check_freq 步检查关键指标，异常时打印中文警告。"""

    def __init__(self, check_freq=5000, window=20000):
        super().__init__()
        self.check_freq = check_freq
        self.window = window
        self.reward_history = []
        self.value_loss_history = []
        self.best_mean_reward = -np.inf
        self.stall_counter = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        logs = self.model.logger.name_to_value if hasattr(self.model.logger, "name_to_value") else {}
        ts = self.num_timesteps

        ep_rew = logs.get("rollout/ep_rew_mean", None)
        entropy_loss = logs.get("train/entropy_loss", None)
        value_loss = logs.get("train/value_loss", None)
        explained_var = logs.get("train/explained_variance", None)
        ep_len = logs.get("rollout/ep_len_mean", None)

        # ---- 奖励平台 / 退化 ----
        if ep_rew is not None:
            self.reward_history.append((ts, ep_rew))
            self.reward_history = [(t, r) for t, r in self.reward_history if ts - t <= self.window]

            recent = [r for _, r in self.reward_history[-4:]]
            if len(recent) >= 4 and ep_rew > 0:
                if max(recent) <= self.best_mean_reward * 1.03:
                    self.stall_counter += 1
                    if self.stall_counter >= 4:
                        print(f"⚠️  [平台期] {ts}步: ep_rew_mean={ep_rew:.1f}，近{self.window // 1000}K步无明显增长")
                        self.stall_counter = 0
                else:
                    self.stall_counter = 0
                if ep_rew > self.best_mean_reward:
                    self.best_mean_reward = ep_rew

            if len(self.reward_history) >= 3:
                old_rew = self.reward_history[0][1]
                if old_rew > 0 and ep_rew < old_rew * 0.7:
                    print(f"⚠️  [退化] {ts}步: ep_rew_mean 从 {old_rew:.1f} 跌至 {ep_rew:.1f}（降幅>30%）")

        # ---- 熵（总熵 / 动作维度 = 每维熵） ----
        if entropy_loss is not None:
            ent_val = -entropy_loss  # SB3 记录的是 entropy_loss = -total_entropy
            act_dim = self.model.action_space.shape[0]
            ent_per_dim = ent_val / act_dim
            if ent_per_dim < 0.1:
                print(f"⚠️  [熵崩溃] {ts}步: entropy_per_dim={ent_per_dim:.4f} — 策略过早收敛")
            elif ent_per_dim > 2.5:
                print(f"⚠️  [过度探索] {ts}步: entropy_per_dim={ent_per_dim:.2f} — 策略可能没学到东西")

        # ---- 价值函数 ----
        if explained_var is not None and explained_var < 0:
            print(f"⚠️  [价值失效] {ts}步: explained_variance={explained_var:.3f}")

        if value_loss is not None:
            self.value_loss_history.append((ts, value_loss))
            self.value_loss_history = [
                (t, v) for t, v in self.value_loss_history if ts - t <= self.window
            ]
            if len(self.value_loss_history) >= 5:
                old_vl = self.value_loss_history[0][1]
                if old_vl > 0.01 and value_loss > old_vl * 10:
                    print(f"⚠️  [价值崩溃] {ts}步: value_loss {old_vl:.4f}→{value_loss:.4f}")

        # ---- 提前终止 ----
        if ep_len is not None and ep_len < 20:
            print(f"⚠️  [提前终止] {ts}步: 平均回合长度仅 {ep_len:.0f} 步")

        return True
