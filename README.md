# PyLite - AI 编译器语言

PyLite 是一个基于 LLVM 的 AI 编译器编程语言，专为 LLM 工程化设计。

## 特性

- 🧠 **LLM 推理与微调**：内置 OpenAI 兼容 API 调用，支持微调全流程
- ⚡ **CUDA 加速**：算子融合优化 + 自动调优 + FlashAttention 内核生成
- 🐳 **Docker 打包**：一键生成 Dockerfile + 构建镜像 + 推送仓库
- 🔧 **编译器后端**：基于 LLVM，可生成 x86/ARM/NVPTX 多平台代码
- 📦 **轻量级**：运行时库仅 ~2MB，零外部依赖

## 快速开始

```bash
# 编译项目
cd build && cmake .. && ninja -j$(nproc)

# C++ 调用
g++ -std=c++17 -I include -L build/bin \
    -o myapp myapp.cpp \
    -lpylite_runtime -Wl,-rpath,$(pwd)/build/bin

# PyLite 脚本
./build/bin/pylitec my_script.pys
```

## 模块

| 模块 | 函数数 | 说明 |
|------|--------|------|
| Docker | 9 | 容器管理 + Dockerfile 生成 + 镜像构建 |
| LLM | 10 | 推理调用 + 网络结构 + 微调全流程 |
| CUDA | 7 | PTX 编译 + 内核启动 + GPU 内存管理 |
| Fusion | 7 | 算子 IR + 融合规则 + 内核生成 + 自动调优 |
| Inference | 7 | 模型加载 + Token 生成 + KV Cache + FlashAttention |
| Tokenizer | 4 | BPE 编码/解码 |
| Server | 3 | OpenAI 兼容 REST API |
| Quantize | 4 | FP16→AWQ 4-bit 量化 + 多架构模板 |

## 许可证

MIT License
