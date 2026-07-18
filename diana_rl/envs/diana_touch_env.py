"""Diana 双臂机器人 — 强化学习触碰目标体任务（V8 最终版）。

在 NVIDIA Isaac Sim 中用 PPO 训练左臂指尖（left_Link9）触碰指定目标方块。
观测 11 维（7 关节角 + 3 相对位置 + 1 欧氏距离），动作 7 维（左臂 7 活动关节的位置增量）。
指尖到目标距离 < 0.12m 判定触碰成功（含方块半径，对齐物理碰撞，避免手臂被碰撞体挡住却要求到达质心）。
"""
import os
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": os.environ.get("HEADLESS", "0") == "1"})

import gymnasium as gym
from gymnasium import spaces

from isaacsim.core.prims import SingleArticulation
from omni.isaac.core.prims import XFormPrim
from isaacsim.core.utils.stage import open_stage
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from pxr import PhysxSchema

import numpy as np


class DianaTouchEnv(gym.Env):
    """左臂触碰目标方块环境（V8 最终版：left_Link9 指尖 + 0.12 成功阈值）。"""

    def __init__(self):
        super().__init__()

        self.headless = os.environ.get("HEADLESS", "0") == "1"

        # USD 场景路径通过环境变量传入，避免写死绝对路径
        self.usd_path = os.environ.get("USD_PATH")
        if not self.usd_path:
            raise RuntimeError(
                "请先设置环境变量 USD_PATH 指向 USD 场景文件，例如：\n"
                "  export USD_PATH=/path/to/rlenvnewv8.usd"
            )
        self.robot_path = "/dual_arm"
        self.target_path = "/dual_arm/TargetCube"
        # V8：控制点从腕部 left_Link7 改为指尖 left_Link9，触碰判定更贴合真实接触点
        self.ee_path = "/dual_arm/left_Link9"

        # 左臂 7 个活动关节（每隔一个关节）
        self.active_dof_indices = [0, 2, 4, 6, 8, 10, 12]
        self.num_active_dof = len(self.active_dof_indices)

        open_stage(self.usd_path)
        self.world = World()
        self.robot = SingleArticulation(prim_path=self.robot_path, name="my_robot")
        self.world.scene.add(self.robot)
        self.target_cube = XFormPrim(prim_path=self.target_path, name="target_cube")
        self.end_effector = XFormPrim(prim_path=self.ee_path, name="end_effector")

        self.world.reset()
        self.robot.initialize()

        try:
            physx_api = PhysxSchema.PhysxArticulationAPI.Apply(self.robot.prim)
            physx_api.CreateEnabledSelfCollisionsAttr().Set(True)
        except Exception as e:
            print(f"[Env] 自碰撞启用失败: {e}")

        self.num_dof = self.robot.num_dof

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_active_dof + 4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_active_dof,), dtype=np.float32
        )

        self.step_count = 0
        self.max_steps = 200
        self.prev_distance = None
        self.best_distance = None
        self.table_height = 0.10
        # V8：成功阈值 0.12m（指尖基准，含方块半径，对齐物理碰撞）
        self.success_threshold = 0.12

    def _get_obs(self):
        joint_pos = self.robot.get_joint_positions()
        active_joint_pos = joint_pos[self.active_dof_indices] / np.pi
        ee_pos, _ = self.end_effector.get_world_pose()
        target_pos, _ = self.target_cube.get_world_pose()
        rel_pos = target_pos - ee_pos
        distance = np.linalg.norm(rel_pos)
        return np.concatenate([active_joint_pos, rel_pos, [distance]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        zero_positions = np.zeros(self.num_dof)
        self.robot.set_joint_positions(zero_positions)
        for _ in range(30):
            self.world.step(render=not self.headless)
        ee_pos, _ = self.end_effector.get_world_pose()
        target_pos, _ = self.target_cube.get_world_pose()
        self.prev_distance = np.linalg.norm(ee_pos - target_pos)
        self.best_distance = self.prev_distance
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        current_joint_pos = self.robot.get_joint_positions()
        active_targets = current_joint_pos[self.active_dof_indices] + action * 0.05
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=active_targets,
                joint_indices=self.active_dof_indices,
            )
        )
        self.world.step(render=not self.headless)

        obs = self._get_obs()
        ee_pos, _ = self.end_effector.get_world_pose()
        target_pos, _ = self.target_cube.get_world_pose()
        distance = np.linalg.norm(ee_pos - target_pos)

        # ==================== Reward (V8) ====================
        reward_progress = (self.prev_distance - distance) * 40.0
        reward_record = 0.0
        if distance < self.best_distance:
            reward_record = (self.best_distance - distance) * 40.0
            self.best_distance = distance
        reward_distance = -distance * 0.3
        # 倒数型 proximity：全程有梯度，近处不爆炸；偏移 0.12 与成功阈值一致
        reward_proximity = 1.0 / (distance + 0.12) * 0.5
        reward_step = -0.01
        reward_joint_limit = 0.0
        for jp in current_joint_pos[self.active_dof_indices]:
            if abs(jp) > 3.0:
                reward_joint_limit -= (abs(jp) - 3.0) * 10.0
        reward = (
            reward_progress
            + reward_record
            + reward_distance
            + reward_proximity
            + reward_step
            + reward_joint_limit
        )

        ee_z = ee_pos[2]
        if ee_z < self.table_height:
            reward += (ee_z - self.table_height) * 20.0

        self.prev_distance = distance

        # ==================== Done ====================
        terminated = False
        term_reason = ""
        if distance < self.success_threshold:
            reward += 100.0
            terminated = True
            term_reason = "success"
        if ee_z < self.table_height - 0.05:
            reward -= 20.0
            terminated = True
            term_reason = "collision" if not term_reason else term_reason
        if np.any(np.abs(current_joint_pos[self.active_dof_indices]) > 3.3):
            reward -= 30.0
            terminated = True
            term_reason = "joint_limit" if not term_reason else term_reason
        truncated = self.step_count >= self.max_steps
        if truncated and not term_reason:
            term_reason = "timeout"

        return (
            obs.astype(np.float32),
            reward,
            terminated,
            truncated,
            {"distance": distance, "term_reason": term_reason},
        )

    def close(self):
        simulation_app.close()


if __name__ == "__main__":
    env = DianaTouchEnv()
    obs, _ = env.reset()
    print(f"Gym 环境测试开始！obs={obs.shape} act={env.action_space.shape}")
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 50 == 0:
            print(
                f"  Step {i:03d} | 距离: {info['distance']:.3f}m | "
                f"Reward: {reward:.3f} | {info['term_reason']}"
            )
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
