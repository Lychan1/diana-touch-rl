 # Diana Touch-RL：双臂机器人强化学习触碰目标体（V8 最终版）


## 演示视频

[[Diana 左臂触碰 demo]](https://www.bilibili.com/video/BV123KF64EVm?t=1.7)


## 任务定义

- **机器人**：Diana 双臂（左臂 7 自由度），末端执行点 `left_Link9`（指尖）
- **目标**：场景中的红色目标方块（Target Cube）
- **观测 (11 维)**：7 个活动关节角(归一化) + 指尖→目标 3D 相对位置 + 1 欧氏距离
- **动作 (7 维)**：左臂 7 个活动关节的位置增量（每步 ×0.05）
- **成功条件**：指尖到目标距离 < 0.12m（含方块半径，对齐物理碰撞）
- **算法**：PPO（stable-baselines3），MLP 策略，网络 `[256, 128]`

## 方法

奖励函数（V8，倒数型 proximity，全程有梯度）：

```
reward = 进度奖励(progress) + 新纪录奖励(record) + 距离惩罚(-0.3·d)
       + 接近奖励(1/(d+0.12)·0.5) + 步数惩罚(-0.01) + 关节限位惩罚
成功: +100 | 碰撞/超限: 终止
```



## 迭代过程（V1 → V8）

本任务是整个项目的第一个 RL 里程碑。前 6 周完成了 URDF 清洗、Isaac Sim 部署、USD 资产转换、底层关节控制与观测接口打通，并封装出 `DianaEnv`；第 7 周正式设定"触碰目标方块"任务、跑通 PPO 训练闭环，**期间经历 V1→V4 四个版本收敛**，之后又做了 V5（泛化验证）与 V8（最终精度版）两轮 refinement，**最终落地版本为 V8**。

| 版本 | 关键改动 | 结果 |
|------|----------|------|
| V1 | 基线：指数型接近奖励 `exp(-25·d)·3`；场景 `rl_env_rl.usd`，桌面碰撞高度 0.05；PPO(lr=1e-4, n_steps=1024, net=[128,128]) | 智能体盲目探索，不收敛 |
| V2 | 抬高目标方块（换场景 `rlenvnew.usd`，桌面高度 0.05→0.10），排除"够不着"的可达性假设 | 仍不收敛 → 排除可达性原因 |
| V3 | 关键发现"**质心碰撞悖论**"：代码以方块*几何中心*算距离，但物理碰撞外壳挡住机械臂，使其物理上永远无法穿透表面达到极小成功阈值 | 定位根因，未单独存版 |
| V8（最终） | 换更精细场景 `rlenvnewv8.usd`；成功阈值 **0.05→0.12m**（指尖基准，含方块半径，完全对齐物理碰撞）；proximity 偏移同步 0.05→0.12；最终采用固定目标位置训练 | ✅ 最终版，触碰判定最贴合真实接触点 |

### 最关键的那个坑（V3 ）

训练一直不收敛，不是奖励弱，而是**任务成功判定和物理现实打架**。

代码里"距离"算的是方块质心，但物理引擎的碰撞体（Collider）不让手臂穿进方块表面——于是手臂再怎么努力，也永远到不了代码要求的"质心距离 < 阈值"。V4 把成功判定放宽到包含方块半径，并换成不会在近处爆炸的倒数型奖励，模型才收敛；**V8 又进一步改用指尖 `left_Link9` + 0.12m 阈值，让"触碰"在物理上真正可达、判定最贴合真实接触点**。

这条经验后来也用在了抓取任务里：**仿真里的成功判定必须对齐底层物理碰撞**。

## 结果

仓库已附带训练好的模型 `checkpoints/diana_touch.zip`（即原 `diana_v8_ppo.zip`，V8 最终版）。
在 Linux + Isaac Sim 环境中运行评估即可得出下表：

<img width="310" height="182" alt="e357d559ea9264b1b13505264c9431ce" src="https://github.com/user-attachments/assets/238dd8d9-ced6-41f9-8563-ba8c03507cf5" />




## 目录结构

```
diana-touch-rl/
├── diana_rl/
│   ├── envs/diana_touch_env.py   # 触碰环境（V8 最终版）
│   └── utils/monitor.py          # 训练监控回调
├── scripts/
│   ├── train.py                  # 训练（支持断点续训）
│   └── eval.py                   # 评估 / 闭环演示
├── checkpoints/diana_touch.zip   # 训练好的模型（V8 最终版 demo）
├── configs/ppo_touch.yaml        # 超参数
├── results/figures/             # reward 曲线（请补充截图）
├── docs/methodology.md           # 方法说明与踩坑
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 快速开始

```bash
# 1. 准备 Isaac Sim 的 Python 环境（isaacsim 不是普通 pip 包，见官方安装）
pip install -r requirements.txt

# 2. 设置 USD 场景路径（你的 rlenvnewv8.usd / 机器人资产，建议用外链下载）
export USD_PATH=/path/to/rlenvnewv8.usd

# 3. 训练（默认无头，200k 步，每 10k 存检查点）
python scripts/train.py

# 4. 评估 / 看闭环演示（会加载 checkpoints/diana_touch.zip）
python scripts/eval.py
```

## 资源（大文件外链）

- USD 场景 `rlenvnewv8.usd` 与机器人资产 → 见 Release→https://github.com/Lychan1/diana-touch-rl/releases/download/v1.0/rlenvnewv8.usd
- 演示视频 `demos/touch_demo.mp4` → (https://www.bilibili.com/video/BV123KF64EVm?t=1.7)

## 踩坑记录（节选自周记）

- **OOM / 驱动崩溃**：大场景编译瞬间占满内存被强杀 → 挂载 16GB Swap；系统自动升级内核/驱动导致渲染管线断裂 → `dkms` 锁驱动 + `apt-mark hold` 冻结升级。
- **URDF 路径不兼容**：`package://` 相对路径报错 → 重写为绝对路径 / 转 USD 资产。
- **质心碰撞悖论**：见上方"迭代过程"V3→V4，是本次任务收敛的真正卡点。

详见 [`docs/methodology.md`](docs/methodology.md)。

## 许可证

MIT © 2026 Lychan
