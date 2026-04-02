<!--
 * @Author: Maonan Wang
 * @Date: 2025-04-18 18:22:47
 * @Description: VLMLight
 * @LastEditors: WANG Maonan
 * @LastEditTime: 2025-07-10 21:53:05
-->
# VLMLight: Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2505.19486-b31b1b.svg)](https://www.arxiv.org/abs/2505.19486)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Version](https://img.shields.io/badge/version-1.0.0-green)

![VLMLight Framework](./assets/method/vlmlight_framework.png)

Official implementation of [VLMLight: Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning](https://www.arxiv.org/abs/2505.19486).

## 📌 News
- **[June 2025]** Codebase open-sourced.
- **[May 2025]** Initial preprint released on arXiv, [VLMLight](https://www.arxiv.org/abs/2505.19486).

## 🚀 Overview

VLMLight presents a novel vision-language multimodal framework for adaptive traffic signal control, featuring:

1. The first vision-based traffic control system utilizing visual foundation models for scene understanding;
2. A dual-branch architecture combining fast RL policies with deliberative LLM reasoning
3. Enhanced handling of safety-critical scenarios through multi-agent collaboration

## ✨ Key Features

### Image-Based Traffic Simulation
First multi-view visual traffic simulator enabling context-aware decision making:

| BEV       | North     | East      | South     | West      |
|-----------|-----------|-----------|-----------|-----------|
| ![BEV](./assets/result/bev.gif) | ![North](./assets/result/north.gif) | ![East](./assets/result/east.gif) | ![South](./assets/result/south.gif) | ![West](./assets/result/west.gif) |

### Dual-Branch Architecture
- **Fast RL Policy**: Efficient handling of routine traffic
- **Deliberative Reasoning**: Structured analysis for complex scenarios
- **Meta-Controller**: Dynamic branch selection based on real-time context

### Safety-Critical Event Handling
Specialized pipeline for emergency vehicle prioritization:

<div align=center>
   <img src="./assets/method/special_event.png" width="85%" >
</div>
<p align="center">Deliberative Reasoning policy for complex traffic in Massy.</p>

## 🛠️ Installation

1. Install [TransSimHub](https://github.com/Traffic-Alpha/TransSimHub):
```bash
git clone https://github.com/Traffic-Alpha/TransSimHub.git
cd TransSimHub
pip install -e ".[all]"
```

2. Install [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent):
```bash
pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]"
# Or use `pip install -U qwen-agent` for the minimal requirements.
# The optional requirements, specified in double brackets, are:
#   [gui] for Gradio-based GUI support;
#   [rag] for RAG support;
#   [code_interpreter] for Code Interpreter support;
#   [mcp] for MCP support.
```


Pretrained models available in [rl_tsc/results](./rl_tsc/results/):

| Hongkong YMT | France Massy | SouthKorea Songdo |
|--------------|--------------|-------------------|
| ![YMT](./assets/reward/Hongkong_YMT_reward.png) | ![Massy](./assets/reward/France_Massy_reward.png) | ![Songdo](./assets/reward/SouthKorea_Songdo_reward.png) |

### 3. Run VLMLight
Execute the decision pipeline:
```bash
cd vlm_tsc_en
python vlmlight_decision.py
```



## 🙏 Acknowledgements

We thank our collaborators from SenseTime and Shanghai AI Lab (in alphabetical order):
- Yuheng Kan (阚宇衡)
- Zian Ma (马子安) 
- Chengcheng Xu (徐承成) 

for their contributions to the [TransSimHub](https://github.com/Traffic-Alpha/TransSimHub) simulator development.

## 📫 Contact

If you have any questions, please open an issue in this repository. We will respond as soon as possible.
