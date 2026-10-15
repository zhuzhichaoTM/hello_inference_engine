# CLAUDE.md

## 目的与项目概况
- **任务目的**：构建通用的低比特量化框架，一键生成指定低比特gguf模型文件
- **验证模型**：Qwen3.8-27B 
- **任务输入**： Qwen3.8-27B BF16 GGUF模型权重文件
- **任务输出**： 详细参考最终交付文件清单
- **推理引擎**：llama.cpp 
- **量化范围**：支持1bit到16bit的量化，支持imatrix张量权重量化

## 最终交付文件清单

项目生成的文件清单如下，包括但不限于如下文件，最终生成件可以在本机也可以在云端环境，需备注文件路径出处。

```
/../Qwen3.8-27B-quant/
├── bf16_gguf/                         # BF16 GGUF 中间产物
│
├── calibration/                       # 校准数据集
│   ├── wikitext.txt                   # WikiText-2
│   ├── alpaca.txt                     # Alpaca 指令数据
│   ├── custom.txt                     # 自定义业务数据
│   ├── self_generated.txt             # UD 自校准数据
│   └── final.txt                      # 合并后的校准集
│
├── gguf/                         # 量化输出：聚集Q4_K_M低比特模型gguf
│   ├── Qwen3.8-27B-Q4_K_M-imatrix.gguf
│   ├── Qwen3.8-27B-UD-Q4_K_M.gguf    # ~15.5GB ⭐ UD 优化档
│   ├── imatrix-Qwen3.8-27B.gguf      # 重要性矩阵
│   ├── imatrix-UD-Qwen3.8-27B.gguf   # UD 重要性矩阵
│   ├── mtp-Qwen3.8-27B-extracted.safetensors  # MTP 提取
│   └── mmproj-extracted.safetensors  # mmproj 提取
│
├── logs/                              # 过程日志
│   ├── imatrix.log
│   ├── imatrix_ud.log
│   └── self_calibration.log
│
└── quant_framework/                   # 量化框架代码，可以支持量化其他大模型
└── test/                              # 测试代码
├── docs/                       # 方案设计文档
│   ├── 大模型低比特模型量化系统设计方案.md
│   ├── 低比特模型量化详细设计方案.md
│   ├── 低比特模型量化详细使用指导.md
```

## 云端环境信息
- **ssh登陆方式**：ssh -p 23 root@117.50.81.252
- **ssh密码**：r2K5TW1683w7v4aO

## 限制要求

- **要求1**：本项目聚集探究gguf模型量化过程，除BF16 GGUF从网络下载外，其他都需要本地量化生成
- **要求2**：量化过程文档存档备案，包括量化方法，量化过程，量化结果
- **要求3**：可以支持外部超参配置
- **要求4**：可以配置生成制定低比特gguf模型；
- **要求5**：可以配置是否激活imatrix量化
- **要求6**：可以配置imatrix量化策略，如基于外部数据或自生成方式
- **要求n**：检查以上要求是否满足
