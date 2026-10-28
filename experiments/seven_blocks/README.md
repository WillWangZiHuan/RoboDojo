# 七色方块：开放式自主能力展示

研究环境第一版（2026-10-28）。**GPU 场景、七块抓放、真实 JSONL 接口及 21 段视频核验均已通过。** [验收记录](acceptance/2026-10-28/README.md)。

## 研究设定

一张 RoboDojo 原生桌子、七个同尺寸方块、原生 dual X5 双臂机器人，以及头部/腕部相机。颜色为红、橙、黄、绿、蓝、紫、白。

提示词保持开放，不给作品示例：

> 用这些小方块展示你模型的最强能力。不限动作，不限失败，可以反复尝试，直到你满意、愿意提交答案为止。

没有指定排列、堆叠目标、质量评分或失败次数上限。`submit` 是 agent 的主动提交；预览结束、输入关闭、程序错误、操作者停止都记录为 `interrupted`，不算满意或成功。该实验不加入官方固定答案 benchmark 的批量评测入口。

## 场景与实施选择

- 使用已安装的 `Assets/Eval_Layout/RoboDojo/arx_x5/1/stack_blocks_0.json`，复制其中第一个原生 block 的模型、缩放、方向及初始高度；七块几何一致。具体物理边长由 GPU smoke 测量，不假定未经核对的资产尺寸。
- 保留原生桌子、房间、地面和灯光；去除其他任务物体及装饰性相机支架。背景设施不作为可操作物体。
- 初始位置为桌面两排错列，坐标见 `protocol.py`；所有方块质量 50 g，静摩擦 0.8、动摩擦 0.6、恢复系数 0。
- 颜色通过 USD 材质覆盖原贴图，同时更新对象描述，避免画面颜色与文本描述冲突。
- 第一版允许 agent 显式重置整个场景：恢复方块位姿与机器人初始关节状态，静置后刷新渲染历史。累计动作数与重置历史保留；不静默恢复掉落方块。该选择是实现默认值，后续可调整。
- 控制接口使用原生关节目标，尚未选定或接入 agent、VLA、抓取规划器。动作仍受机器人和仿真物理限制。
- 默认返回 RGB 图像路径；物体真实位姿仅写入实验日志。是否把坐标提供给 agent 应作为单独实验条件。

## 运行

在已经安装 RoboDojo、Isaac Sim / IsaacLab、原生 Assets 和依赖的 Linux NVIDIA GPU 环境中，从仓库根目录运行。Python 必须使用原有 RoboDojo 环境。

```bash
# 本地协议测试，不需要 GPU
python -m unittest discover -s tests -p 'test_seven_blocks.py' -v

# 有界预览，输出头部/腕部 PNG 和状态记录；不调用模型
python -m experiments.seven_blocks.run --mode preview --headless \
  --output eval_result/seven_blocks/preview_001

# 物理场景 + 相机 + 重置 + 显式提交验收；提交是合成测试，非模型成果
python -m experiments.seven_blocks.run --mode smoke --headless \
  --output eval_result/seven_blocks/smoke_001

# 逐块独立抓起、移动 8 cm、放下；使用真实位姿的验收脚本，不是 agent 实验
python -m experiments.seven_blocks.run --mode manipulation --headless \
  --output eval_result/seven_blocks/manipulation_001

# 真实进程 JSONL 往返验收；不调用模型
python -m experiments.seven_blocks.check_stdio \
  --output eval_result/seven_blocks/stdio_check_001

# 给未来 agent 的持久 JSONL 接口；输入来自 stdin，回复到 stdout，Isaac 日志到 stderr
python -m experiments.seven_blocks.run --mode stdio --headless \
  --output eval_result/seven_blocks/session_001
```

输出目录必须不存在，以避免覆盖旧实验。原生布局文件不在默认位置时使用 `--template /absolute/path/stack_blocks_0.json`。

在 Slurm 集群上通过已有分区/资源配置启动上述命令；不要在登录节点启动 Isaac Sim。调度器时间上限是基础设施限制，不是 agent 自愿终点。SIGTERM 会尽力保存中断记录；SIGKILL 或 GPU 崩溃可能仅留下已刷盘的事件日志。

仓库提供 Bash 作业入口，避免 `sbatch --wrap` 默认使用 `sh` 导致 `source` 失败。从仓库根目录提交，例如：

```bash
sbatch --partition=RTX3090 --gres=gpu:3090:1 --cpus-per-task=8 --mem=48G \
  --time=00:20:00 experiments/seven_blocks/slurm.sh /path/to/activate.sh smoke
```

激活脚本应加载已有 RoboDojo Python 环境。分区和 GPU 类型按实际集群填写；检查抓放时把末尾 `smoke` 换为 `manipulation`。

### JSONL 协议

启动后收到 `ready`，包含提示词、图像路径、`control_template`、物理步长和操作列表。请求每行一个 JSON：

```json
{"op":"observe"}
{"op":"reset","reason":"我想重新尝试"}
{"op":"submit","description":"我的最终作品说明","reason":"我认为已经充分展示了能力"}
```

动作格式为 `{"op":"act","control":完整关节字典,"physics_steps":10}`。字典键和维度使用 `ready.control_template`；每个条目有 `position` 与可选 `velocity`。先将模板原样发送可测试保持姿态。`physics_steps` 每次为 1–250，可反复调用；默认仿真步长 0.004 秒。此接口不等同于尚未实现的自然语言抓取工具。

`stop` 用于操作者中断。`reset` 保持同一 episode，不清除此前动作计数；`submit` 后会保存成果并关闭 session，不再接受动作。

## 证据与验收

每次运行保存 `config.json`、`layout.json`、`events.jsonl`、相机 PNG 和 `episode.json`。每次观察、动作后状态与输入请求可核对。`manipulation` 另为每个方块保存头部、左腕、右腕三个完整 MP4（共 21 段），在每 10 个物理步采样一次，25 fps 按仿真时间播放；每段从重置完成开始，不包含重置过程或规划计算的现实等待时间。`frames.jsonl` 对应每帧的仿真时刻、动作阶段和物体状态。其中 `physics_tick` 是片段内实际物理步；`state.actions/physics_steps` 是已完成命令的累计计数，长命令执行中尚未结算。正式 agent 控制器尚未接入，`stdio` 当前仍是逐命令记录。

`smoke` 检查七个标签、有限且相同的物体尺寸、桌面支撑、初始无重叠、静置漂移、相机帧可读、重置保留累计记录、提交前不结束、提交后拒绝动作。全部通过才写 `acceptance.json`。还需人工检查输出图像是否七色清晰、是否都在可见区域，并实际验证抓取与放置；本地单元测试不能替代这一步。

`manipulation` 对每块从同一初始布局独立测试，并检查两臂起始关节状态一致，使用原生 cuRobo 关节轨迹和夹爪接触。前排垂直抓取，后排前倾 30° 抓取，两台 X5 底座朝向相同，采用相同腕部方向。开始录像前先从统一回零状态检查全部接近轨迹，并在逐块重置后复用这些已验证轨迹。测试要求块底离桌至少 7 cm、向机器人一侧移动 8 cm 后放下，最终平面误差小于 2 cm、距桌面小于 1 cm。逐阶段状态和图像写入日志，汇总写入 `manipulation.json`；任一块未通过则进程失败。此处真实位姿仅供验收器，移动目标也不写入 agent 提示词。该测试不能证明任意堆叠或任意动作可行。录像使用 `imageio-ffmpeg` 编码。下载完整运行目录后，可运行以下检查，逐段完整解码并核对帧数、时间线及 SHA-256：

```bash
python -m experiments.seven_blocks.verify_recordings /path/to/manipulation_run
```

调试时可用 `--mode manipulation --colors blue purple white` 只运行指定块；完整验收与视频核验要求七块齐全。

通过后生成 `video_manifest.json`；视频留在运行/交付目录，仓库保存验收汇总和视频校验清单。

## 维护

仓库：`WillWangZiHuan/RoboDojo`；工作分支：`will/open-ended-seven-blocks`。改动集中在 `experiments/seven_blocks/`，保留原生任务与子模块版本。每次修改先运行协议测试，涉及物理、视角或资产的改动再运行 GPU smoke，通过后提交并 push。本路线不恢复此前暂停的 Harness / VLA 批量实验。

进度与验收目标见 [GOAL.md](GOAL.md)。
