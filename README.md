# 基于手势指向的人机交互目标识别与 AR 增强系统

> 课程设计项目 · 人工智能 2302 · 张英鹏

非接触式人机交互系统，通过 **手指指向 → 注视停留 → 点头确认 → 目标跟踪 → 摇头解锁 → AR 信息增强** 的完整交互闭环，实现自然、直观的无接触控制。

---

## 完整流程演示

点击播放完整操作演示视频：

<video src="report/video/demo_video.mp4" controls width="100%" style="max-width:800px;border-radius:8px">
  您的浏览器不支持视频播放，请下载 <a href="report/video/demo_video.mp4">demo_video.mp4</a> 后观看。
</video>

---

## 功能演示

| 功能 | 演示 |
|------|------|
| **手指指向与定位** — 指尖指向目标，实时检测 | ![手指指向与定位](report/demo_gifs/%E6%89%8B%E6%8C%87%E6%8C%87%E5%90%91%E4%B8%8E%E5%AE%9A%E4%BD%8D.gif) |
| **目标跟踪与丢失恢复** — Kalman 滤波跟踪，遮挡后自动恢复 | ![目标跟踪](report/demo_gifs/%E7%9B%AE%E6%A0%87%E8%B7%9F%E8%B8%AA.gif) |
| **AR 信息面板** — 竖拇指唤醒，GPT 分析图像内容 | ![AR面板](report/demo_gifs/AR%E5%B1%8F%E5%B9%95%E8%B7%9F%E8%B8%AA.gif) |
| **AR 面板唤醒** — 两步唤醒流程（截图 + GPT 分析） | ![AR唤醒](report/demo_gifs/AR%E5%B1%8F%E5%B9%95%E7%9A%84%E5%94%A4%E9%86%92.gif) |
| **AR 面板退出** — 摇头解锁，面板自动隐藏 | ![AR退出](report/demo_gifs/AR%E5%B1%8F%E5%B9%95%E7%9A%84%E7%BB%93%E6%9D%9F%E6%93%8D%E4%BD%9C.gif) |

---

## 核心功能

- **YOLOv8 目标检测** — 实时检测 80 类物体，置信度阈值 0.35
- **MediaPipe Hand** — 21 个手部关键点，指尖指向计算 + 手势识别（竖拇指等）
- **MediaPipe Pose** — 头部姿态估计，低头点头确认 / 摇头解锁
- **注视停留判定** — 指尖在目标区域停留 1.5 秒触发确认弹窗，带衰减容错
- **三级确认机制** — 指向 → 停留 → 点头，逐级降低误触发
- **Kalman 滤波跟踪** — 对目标点和检测框分别建模，平滑预测
- **ROI 局部检测** — 锁定后只在目标周围 ROI 检测，帧率提升至 35fps+
- **丢失恢复** — 目标消失后保持最后位置，扩大搜索半径自动重捕获
- **AR 信息面板** — 半透明面板叠加 GPT/DeepSeek/Qwen 等 AI 分析结果
- **多模型支持** — ChatGPT / DeepSeek / Qwen / GLM 可选
- **多状态机** — IDLE → DWELL_WAIT → CONFIRMING → TRACKING → UNLOCKING

---

## 快速开始

### 环境要求

- Python 3.8+
- 两个 USB 摄像头（或 1 个 + Azure Kinect）
- GPU（推荐，YOLO 推理需要）

### 安装

```bash
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 检查摄像头编号（默认：前置 CAM_FRONT_ID=0，侧置 CAM_SIDE_ID=1）
#    可在 config.py 中修改
```

### 运行

**基础版本（登录 + AR 面板）：**
```bash
python run_ar.py
```

启动后显示登录页面，按 `1`-`4` 选择 AI 模型（ChatGPT / DeepSeek / Qwen / GLM），输入对应 API Key 后 Enter 登录。

**简洁版本（无登录，直接运行）：**
```bash
python main.py
```

### 操作方式

| 操作 | 动作 | 说明 |
|------|------|------|
| 选择目标 | 手指指向目标物体 | 指尖对准物体，停留 1.5 秒 |
| 确认锁定 | 低头点头 | 触发弹窗后点头确认绑定目标 |
| 取消选择 | 左右摇头 | 弹窗中摇头取消选择 |
| 唤醒 AR | 竖拇指 👍 | 第一次截图，第二次 GPT 分析 |
| 滚动面板 | 手指上下滑动 | AR 面板内容滚动 |
| 解锁退出 | 左右摇头 | 跟踪状态下摇头解锁回到空闲 |

### 配置

所有参数集中在 `config.py`，包括：

- 摄像头 ID、分辨率
- 停留触发时长、超时时间
- 检测阈值（YOLO / MediaPipe）
- 头部姿态角度阈值
- Kalman 滤波器参数

---

## 项目结构

```
├── main.py                  # 主程序（简洁版）
├── run_ar.py                # 主程序（AR 增强版，含登录）
├── config.py                # 全局配置
├── requirements.txt         # 依赖
├── ar_layer/                # AR 模块
│   ├── login_manager.py     # 登录 & 多模型管理
│   ├── gpt_analyzer.py      # AI 视觉分析（多模型）
│   ├── ar_manager.py        # AR 状态管理
│   ├── ar_panel.py          # AR 面板渲染
│   ├── world_anchor.py      # 世界锚点
│   └── button_trigger.py    # 按钮触发
├── hand_module.py           # 手部检测
├── face_module.py           # 人脸检测
├── face_gesture.py          # 头部姿态识别
├── pose_module.py           # 姿态检测
├── object_module.py         # YOLO 目标检测
├── kalman_tracker.py        # Kalman 滤波跟踪
├── background_model.py      # 背景建模
├── utils.py                 # 工具函数
├── hardware_camera.py       # 摄像头接口
└── report/                  # 演示素材
    ├── cv_report.html       # 课程设计展示网页
    └── demo_gifs/           # 功能演示 GIF
```

---

## 演示网页

打开 `report/cv_report.html` 可查看课程设计答辩展示网页，支持键盘 ← → 或滚轮翻页。

---

## 链接

- GitHub: [github.com/JUpi5teR/gesture_pointed_recognition](https://github.com/JUpi5teR/gesture_pointed_recognition)
- 完成者：张英鹏 · 人工智能 2302
