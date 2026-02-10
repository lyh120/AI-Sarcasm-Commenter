# 🎣 AI评论君 (AI-Sarcasm-Commenter)

> **基于中文反讽数据集的 DeepSeek-R1 轻量化微调模型**  
> *Powered by MindSpore & MindNLP on Huawei Ascend NPU*

[![MindSpore](https://img.shields.io/badge/MindSpore-2.6.0-blue)](https://www.mindspore.cn/)
[![MindNLP](https://img.shields.io/badge/MindNLP-0.4.0-green)](https://github.com/mindspore-lab/mindnlp)
[![Model](https://img.shields.io/badge/Base-DeepSeek--R1--Distill--Qwen--1.5B-orange)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey)]()

## 📖 项目简介 (Introduction)

**AI评论君** 是一个专注于中文反讽（Sarcasm）语境理解与生成的垂直领域大模型。

讽刺是人类语言的高级形式，包含幽默与批评。通用大模型（LLM）往往只能做字面理解，缺乏对反讽语境的深层感知，生成的回复往往显得生硬、缺乏“人味”。本项目旨在赋予 AI “鉴赏”与“怼人”的能力。

本项目基于 **MindSpore** 框架，使用自建的高质量中文反讽数据集（3752条），对 **DeepSeek-R1-Distill-Qwen-1.5B** 基座模型进行了 **LoRA** 微调。

### 🌟 核心能力
1.  **舆情分析**：精准识别网络阴阳怪气，提升情感分析准确率（分类/解释）。
2.  **智能交互**：用于社交机器人、游戏NPC，赋予AI“幽默感”和拟人化性格（角色扮演）。

---

## 🏗️ 技术架构 (Technical Architecture)

*   **算力平台**: Huawei Cloud Ascend snt9b1 (NPU)
*   **开发框架**: MindSpore 2.6.0 + MindNLP 0.4.0
*   **基座模型**: DeepSeek-R1-Distill-Qwen-1.5B
    *   *选择原因*: 规模适中(1.5B)，中文指令遵循能力强，推理效率高，适合端侧/本地部署。
*   **微调方法**: Instruction Tuning + LoRA (Low-Rank Adaptation)

---

## 📚 数据集 (Dataset)

我们构建了首个系统化的多任务中文反讽数据集 **CN-Sarcasm-3k**。数据来源于 Bilibili 真实评论，经过人工清洗与机器辅助标注。

*   **数据总量**: 3752 条
*   **任务类型**:
    *   `Classification`: 判断是否为讽刺并分类（如：事实讽刺、矛盾讽刺）。
    *   `Understanding`: 深度解析讽刺背后的逻辑和隐含意义。
    *   `Roleplay`: 模仿网络语境，生成风格化的讽刺评论。

### 数据格式示例

```json
{
    "instruction": "请阅读以下评论，并判断最后一条评论是否是讽刺性的，并给出讽刺类型。",
    "input": "水里根本没有鱼，不然我为什么钓不到？\n钓鱼佬永不空军！！！[大笑]\n... \n你得通过实验去证明啊，抽水就是个好办法[doge]",
    "output": "这是讽刺，类型是 fact。解析：最后一条评论通过事实讽刺(fact)的方式，表面上看似认真地建议通过“抽水”来验证，实则利用这种极端的方式来讽刺原观点的逻辑谬误：仅凭自己钓不到鱼就断定水里没有鱼。"
}
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

本项目运行于昇腾 NPU 环境，也可以在 GPU/CPU 环境下通过修改 device target 运行。

```bash
# 基础环境
python == 3.10
mindspore == 2.6.0
mindnlp == 0.4.0
```

### 2. 模型微调 (Fine-tuning)

本项目使用 LoRA 技术进行高效微调。

**关键配置 (LoRA Config):**
```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
```

**启动训练:**
```bash
python train.py \
    --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset_path "./data/sarcasm_train.json" \
    --output_dir "./output_lora_sarcasm" \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-4
```

### 3. 推理与生成 (Inference)

加载微调后的 Adapter 权重进行推理：

```python
from mindnlp.peft import PeftModel, PeftConfig
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基座与LoRA权重
model = AutoModelForCausalLM.from_pretrained("DeepSeek-R1-Distill-Qwen-1.5B")
model = PeftModel.from_pretrained(model, "./output_lora_sarcasm/checkpoint-final")

# 钓鱼佬的反讽生成
prompt = "请针对下面的评论生成一条讽刺评论：\n水里根本没有鱼，不然我为什么钓不到？"
# ... (Tokenize and Generate code) ...
```

---

## 📊 效果展示 (Results)

### Case Study: "钓鱼佬的辩解"

**用户输入**: 
> “水里根本没有鱼，不然我为什么钓不到？”

| 模型状态 | 回复内容 | 评价 |
| :--- | :--- | :--- |
| **微调前 (Base Model)** | 用户可能在抱怨钓不到鱼。可能是因为鱼太小或者水太深。建议检查鱼饵。 | 过于理性，像在做阅读理解，毫无幽默感。 |
| **微调后 (AI评论君)** | **“哈哈，水里根本没有鱼，不然我钓不到鱼啊！真是个‘奇才’啊！🤣 对啊，如果水里没有鱼，那钓鱼就太简单了，根本不需要任何努力！👍”** | **精准反讽**。使用了反语、Emoji，完美模仿了网络“键盘侠”的语气，逻辑上直接攻击了原句的荒谬性。 |

---

## 📂 目录结构

```text
AI-Commenter/
├── data/   # 自建中文反讽数据集
├── code/
│  └── AI_sarcasm.ipynb     # 项目notebook
├── web_demo/                 # 简易的前端交互界面
├── requirements.txt
└── README.md
```

## 🤝 致谢

*   感谢 **MindSpore** 社区提供的框架支持。
*   感谢 **Bilibili** 评论区提供的丰富且充满智慧的语料来源。

---

*Project by 一零二四小队*
