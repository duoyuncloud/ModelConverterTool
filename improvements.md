# 模型转换工具改进计划

为了提升模型转换工具的灵活性和鲁棒性，需要对以下两个核心功能进行重构和优化。

---

## 1. `fake_weight` 功能增强

### 当前状态
`fake_weight` 现在是 `convert` 命令的一个 `--fake-weight` 选项，生成固定维度的零权重，主要用于测试转换流程。

### 设计方案
**保持现有的集成方式，不需要独立的 `init` 指令**

- **理由**: fake_weight 是调试和测试工具，更适合作为现有命令的选项而非独立功能
- **当前用法**: `modelconvert convert <model> <format> --fake-weight`
- **扩展思路**: 在现有 `--fake-weight` 基础上添加自定义维度支持

### 改进目标
- 增强现有 `--fake-weight` 选项，支持自定义维度参数
- 添加 `--fake-weight-config` 选项，允许通过 JSON/YAML 文件指定各层权重的形状
- 保持向后兼容，无配置时使用默认零权重

### 应用场景
- 在没有真实权重文件的情况下，快速初始化模型以进行结构调试或单元测试
- 支持对动态模型架构（例如，改变头数量、层数）的测试

---

## 2. muP 缩放参数转换 (`--mup2llama`)

### 背景说明
muP (μ-Parametrization) 是一种参数初始化和缩放技术，通过特定的缩放因子来确保模型在不同宽度下的训练稳定性。`llama-format` 工具在处理包含 muP 缩放参数的模型时，转换逻辑需要完善。

### muP 参数详解（以 MiniCPM 为例）
- **`scale_emb`** (embedding_scale): 词嵌入层的缩放因子，直接从配置中读取
- **`scale_depth`** (residual_scale): 残差连接的缩放因子，计算公式为 `scale_depth / sqrt(num_hidden_layers)`
- **`dim_model_base` + `hidden_size`** (logit_scale): 输出 logit 的缩放因子，计算公式为 `hidden_size / dim_model_base`

这些参数影响模型的数值稳定性和训练收敛性，特别是在不同模型规模之间进行缩放时。

### 转换示例
```python
# MiniCPM 配置中的 muP 参数
config = {
    "scale_emb": 12.0,           # 词嵌入缩放
    "scale_depth": 1.4,          # 深度缩放  
    "dim_model_base": 256,       # 基础模型维度
    "hidden_size": 2304,         # 当前隐藏层维度
    "num_hidden_layers": 40      # 层数
}

# 转换后需要处理的缩放因子
embedding_scale = config["scale_emb"]                                          # = 12.0
residual_scale = config["scale_depth"] / sqrt(config["num_hidden_layers"])     # = 1.4 / 6.32 ≈ 0.22
logit_scale = config["hidden_size"] / config["dim_model_base"]                 # = 2304 / 256 = 9.0
```

### 设计方案：`convert` 命令的 `--mup2llama` 选项

#### 基本用法
```bash
modelconvert convert openbmb/MiniCPM-2B-sft-bf16 huggingface --mup2llama
```

#### 优势
- **架构一致**: 与现有架构高度一致（参考 `--fake-weight`、`--dtype` 等选项）
- **基础设施复用**: 复用 `convert` 命令的验证、历史记录、错误处理等功能
- **代码先例**: 已有 `custom_quant`、`safetensors` 等特殊处理逻辑的成功先例
- **命名直观**: `--mup2llama` 明确表示从 muP 转换到 LLaMA 格式

#### 实现方案
- 添加 `--mup2llama` 布尔选项启用 muP 参数转换
- 在转换引擎中检测模型是否包含 muP 参数，自动应用相应的缩放逻辑
- 支持与现有输出格式组合使用（`huggingface`、`safetensors`、`gguf` 等）

#### 改进目标
- **正确处理 muP 缩放参数**: 将 muP 相关的缩放参数正确映射或转换为标准 LLaMA 架构中对应的参数形式
- **权重重新缩放**: 根据 muP 缩放因子对相应的权重张量进行适当的数值调整，确保转换后模型的数值行为一致
- **配置参数适配**: 确保转换后的模型配置文件中移除 muP 特有参数，添加标准 LLaMA 所需的配置

---

## 最终目标

将带有 muP 缩放的模型权重转换为标准 LLaMA 格式，确保转换后的模型可以被原生 LLaMA 推理框架正确加载和运行。




##
-y ./install.sh 1. 试试在干净的环境安装是否能一键完成 
-y test_integration+ci 2. 实现一个高coverage的集成测试，用于版本更新前跑
-o3. 加入更精细的量化配置支持（比如desc, group_size, sym等等）
-o try-run-->check*4. 支持更多小功能（to-llama-foramt、try-run、fake-weight等等），从modelconvert变为modeltool

*5. 接入更多实际业务使用的转换（Megatron2hf, hf2mtk, hf2rk, hf2ax, hf2qnn），大多是闭源的
*6. 干点别的，比如基于模型的cli自动补全工具