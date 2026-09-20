# NVIDIA GPU 全产品线系统分析报告（2016–2026）

> **报告说明**
> - 覆盖范围：数据中心（Tesla/DGX/HGX）、消费级（GeForce）、专业可视化（Quadro/RTX A/RTX PRO）、车载（DRIVE）、机器人与边缘（Jetson）五大产品线，时间跨度 2016–2026 共十年。
> - 数据原则：所有规格数据均来自 NVIDIA 官方 datasheet / 产品页 / 新闻稿、TechPowerUp GPU Database、Wikipedia 微架构条目等公开权威来源，文末附逐条参考 URL。官方未公布的数据（如数据中心卡零售价）标注为"行业估值"或"待核实"，不做杜撰。
> - 时效声明：报告撰写于 2026 年 9 月。云 GPU 租赁价格、Blackwell 中低端型号定价、Rubin 详细规格时效性强，引用时已标注置信度。

---

## 目录

1. [执行摘要](#一执行摘要)
2. [产品线全景与十年架构总览](#二产品线全景与十年架构总览)
3. [数据中心 GPU 详解](#三数据中心-gpu-详解)
4. [消费级 GeForce 详解](#四消费级-geforce-详解)
5. [专业可视化 GPU 详解](#五专业可视化-gpu-详解)
6. [车载 DRIVE 平台详解](#六车载-drive-平台详解)
7. [机器人与边缘 Jetson 平台详解](#七机器人与边缘-jetson-平台详解)
8. [关键选型参数解读](#八关键选型参数解读)
9. [业务选型决策框架](#九业务选型决策框架)
10. [典型应用场景配置案例](#十典型应用场景配置案例)
11. [各领域架构演进趋势分析](#十一各领域架构演进趋势分析)
12. [附录：参考链接与置信度说明](#十二附录参考链接与置信度说明)

---

## 一、执行摘要

过去十年（2016–2026），NVIDIA GPU 完成了从"图形处理器"到"AI 计算基础设施"的彻底转型，其核心脉络可概括为四条主线：

1. **算力精度不断下探**：FP64/FP32（Pascal 时代唯一路径）→ FP16 Tensor Core（Volta，2017）→ INT8/INT4（Turing）→ TF32/BF16/稀疏化（Ampere）→ FP8 + Transformer Engine（Hopper）→ FP4 微缩放（Blackwell）→ HBM4 + 更大 FP4 域（Rubin）。每一代新精度几乎直接翻倍等效算力。
2. **交付形态从"卡"到"柜"**：2016 年 DGX-1 是 8×P100 的单机；2024 年 GB200 NVL72 把 72 颗 GPU 装进一个 NVLink 域机柜（~120 kW）；Rubin 时代进一步到 NVL144/NVL576。采购单位、散热、网络设计范式彻底改变。
3. **市场分层极致化**：数据中心（HBM + NVLink + MIG + 高 TDP）、专业可视化（ECC + 认证驱动 + 显示输出）、消费级（游戏/创作性价比）、车端（ASIL-D 功能安全 + 低功耗）、边缘（5–60W 模组化 SoM）五条产品线共用同一套架构代号，但裁剪维度完全不同。
4. **地缘政治催生特供线**：A800/H800（NVLink 限速）、H20（算力大幅削减）、RTX 5880 Ada、RTX 5090D 等合规型号成为中国市场选型必须单独评估的分支。

**一句话选型结论**：训练选 NVLink 域内大显存旗舰（H100/H200 → B200/GB200 → Rubin），高并发推理选带宽/显存性价比最优卡（H200/L40S/RTX PRO 6000），边缘与车端按功耗档位选 Orin/Thor，创作与入门 AI 选 GeForce/RTX PRO 桌面卡。详细决策路径见第九章。

---

## 二、产品线全景与十年架构总览

### 2.1 五代半架构与产品矩阵映射

| 架构代际 | 发布年份 | 制程 | 数据中心 | 专业可视化 | 消费级 | 车端/边缘 SoC | 关键技术 |
|---|---|---|---|---|---|---|---|
| **Pascal** | 2016 | TSMC 16nm | P100/P40/P4 | Quadro P6000/GP100 | GTX 10 系 | Parker（PX2） | HBM2 首发、NVLink1、GDDR5X |
| **Volta** | 2017 | TSMC 12nm FFN | V100 | GV100/Titan V | — | Xavier | **第1代 Tensor Core**、NVLink2 |
| **Turing** | 2018 | TSMC 12nm | T4 | Quadro RTX 8000 | RTX 20 系 | （Pegasus 用 Turing dGPU） | **RT Core 首发**、INT8/INT4 TC、GDDR6 |
| **Ampere** | 2020 | TSMC N7 / Samsung 8N | A100/A30/A10/A2 | RTX A 系列 | RTX 30 系 | **Orin** | TF32/BF16、**2:4 稀疏**、MIG、NVLink3 |
| **Ada Lovelace** | 2022 | TSMC 4N | L40/L40S/L4 | RTX xxxx Ada | RTX 40 系 | — | DLSS 3 帧生成、FP8 TC、大 L2 |
| **Hopper** | 2022 | TSMC 4N | H100/H200/H20 | — | — | （Thor 早期方案曾用） | **Transformer Engine/FP8**、NVLink4 900GB/s |
| **Blackwell** | 2024 | TSMC 4NP | B200/GB200/GB300 | RTX PRO Blackwell | RTX 50 系 | **Thor/Jetson Thor** | **FP4/FP6**、双 die、NVLink5 1.8TB/s、GDDR7 |
| **Rubin** | 2026 | TSMC 3nm 级 | Vera Rubin NVL144 | — | — | — | HBM4、NVLink6、Vera CPU |
| **Feynman** | 2028（路线图） | — | 待公布 | — | — | — | 接替 Rubin，细节未公开 |

### 2.2 产品线共用架构、裁剪维度不同

同一架构代号下，NVIDIA 对不同市场做四类裁剪：

- **计算精度裁剪**：数据中心旗舰保留完整 FP64（P100/V100/A100/H100 FP64 ≈ FP32 一半）；消费级 FP64 砍至 1/64；推理卡（P4/T4/L4/A2）几乎无 FP64。
- **显存介质裁剪**：数据中心用 HBM2→HBM3e→HBM4（带宽 732GB/s → 8TB/s+）；专业/消费用 GDDR5X→GDDR6X→GDDR7（成本导向）；车端/边缘用 LPDDR4x/LPDDR5（功耗导向）。
- **互联裁剪**：NVLink 是数据中心/部分专业卡专属（P100 160GB/s → B200 1.8TB/s）；消费级自 40 系起完全取消 SLI/NVLink。
- **可靠性/合规裁剪**：ECC 显存、ISV 驱动认证、vGPU 授权、MIG 切分、ASIL-D 车规认证，分别对应专业卡、数据中心卡、车端 SoC。

### 2.3 旗舰算力十年跨度

| 指标 | 2016 P100 SXM2 | 2025 B200 | 倍数 |
|---|---|---|---|
| FP32 | 10.6 TFLOPS | ~80 TFLOPS | ~7.5× |
| AI 峰值（最低精度） | 21.2 TFLOPS (FP16) | 9,000 TFLOPS dense FP4 | **~425×** |
| 显存带宽 | 732 GB/s | 8,000 GB/s | ~11× |
| 显存容量 | 16 GB | 192 GB | 12× |
| 单卡 TDP | 300 W | 1,000 W | 3.3× |
| NVLink | 160 GB/s | 1,800 GB/s | 11× |

> 注：AI 算力 425 倍中，绝大部分来自**精度下探 + 稀疏化 + Tensor Core 规模**，而非传统 FLOPS 提升——这正是理解 NVIDIA 十年演进的关键。

## 三、数据中心 GPU 详解

### 3.1 Pascal（2016）：深度学习基础设施元年

| 型号 | 制程 | FP64 | FP32 | FP16 | 显存 | 带宽 | TDP | 互联 | 参考价 |
|---|---|---|---|---|---|---|---|---|---|
| Tesla P100 SXM2 | TSMC 16nm | 5.3 TF | 10.6 TF | 21.2 TF | 16GB HBM2 | 732 GB/s | 300W | NVLink1 160GB/s | DGX-1 整机 $129K |
| Tesla P100 PCIe | TSMC 16nm | 4.7 TF | 9.3 TF | 18.7 TF | 12/16GB HBM2 | 549/732 GB/s | 250W | PCIe 3.0 | ~$5–6K（估） |
| Tesla P40 | TSMC 16nm | 0.20 TF | 12 TF | — | 24GB GDDR5 | 346 GB/s | 250W | PCIe 3.0 | ~$5,699 |
| Tesla P4 | TSMC 16nm | 0.18 TF | 5.5 TF | INT8 22 TOPS | 8GB GDDR5 | 192 GB/s | 50–75W | PCIe 3.0 LP | ~$2,249 |

**要点**：P100 首次将 HBM2 + NVLink 带入数据中心，DGX-1（8×P100，$129K）成为 AI 超算的原始形态；P40/P4 无 FP16 加速，定位推理与视频转码，T4 之前的主力推理卡。

### 3.2 Volta（2017）：Tensor Core 诞生

| 型号 | FP64 | FP32 | FP16 Tensor | 显存 | 带宽 | TDP | 互联 |
|---|---|---|---|---|---|---|---|
| V100 SXM2 16/32GB | 7.8 TF | 15.7 TF | **125 TF** | 16/32GB HBM2 | 900 GB/s | 300W | NVLink2 300GB/s |
| V100 PCIe 16/32GB | 7.0 TF | 14.0 TF | 112 TF | 16/32GB HBM2 | 900 GB/s | 250W | PCIe 3.0 |
| V100S PCIe 32GB | 8.2 TF | 16.4 TF | 130 TF | 32GB HBM2 | 1,134 GB/s | 250W | PCIe 3.0 |

**要点**：第1代 Tensor Core（FP16 输入/FP32 累加）使深度学习算力较 P100 提升约 6 倍（125 vs 21 TF），是现代 AI 算力革命的起点；DGX-1V $149K、DGX-2（16 卡 + NVSwitch）$399K。

### 3.3 Turing（2018）：推理专用化

| 型号 | FP32 | FP16 TC | INT8/INT4 | 显存 | 带宽 | TDP | 形态 | 参考价 |
|---|---|---|---|---|---|---|---|---|
| Tesla T4 | 8.1 TF | 65 TF | 130/260 TOPS | 16GB GDDR6 | 320 GB/s | 70W | 单槽半高 LP | ~$2,499 |

**要点**：第2代 Tensor Core 支持多精度（FP16/INT8/INT4）；70W 低功耗单槽形态使其成为云厂商推理主力（AWS G4 等），至今仍有存量。

### 3.4 Ampere（2020–2022）：训练推理统一底座

| 型号 | FP64 | FP32 | TF32 dense | BF16 dense | INT8 dense | 显存 | 带宽 | TDP | 互联 | 参考价 |
|---|---|---|---|---|---|---|---|---|---|---|
| A100 SXM4 40GB | 9.7 | 19.5 | 156 | 312 | 624 TOPS | 40GB HBM2e | 1,555 GB/s | 400W | NVLink3 600GB/s | ~$10–15K |
| A100 SXM4 80GB | 9.7 | 19.5 | 156 | 312 | 624 TOPS | 80GB HBM2e | 2,039 GB/s | 400W | NVLink3 600GB/s | ~$20K+ |
| A100 PCIe 40/80GB | 9.7 | 19.5 | 156 | 312 | 624 TOPS | 40/80GB HBM2e | 1,555/1,935 GB/s | 250/300W | PCIe 4.0 | ~$10–20K |
| A30 | 5.2 | 10.3 | 82 | 165 | 330 TOPS | 24GB HBM2 | 933 GB/s | 165W | NVLink 200GB/s | ~$4,500 |
| A10 | 0.97 | 31.2 | 62.5 | 125 | 250 TOPS | 24GB GDDR6 | 600 GB/s | 150W | PCIe 4.0 | ~$3,000 |
| A16 | — | 4×8 | — | — | — | 4×16GB GDDR6 | 4×232 GB/s | 250W | PCIe 4.0 | ~$5,000（VDI） |
| A2 | — | 4.5 | — | 18 | 36 TOPS | 16GB GDDR6 | 200 GB/s | 40–60W | PCIe 4.0 LP | ~$1,500 |
| **A800 SXM4（中国）** | 9.7 | 19.5 | 156 | 312 | 624 TOPS | 80GB HBM2e | 2,039 GB/s | 400W | NVLink3 **限速400GB/s** | ~$12K+ |

> 上表 Tensor 算力为 dense 值；Ampere 起支持 2:4 结构化稀疏，稀疏峰值 = dense × 2。

**要点**：第3代 TC 引入 TF32（FP32 代码零改动加速 8×）、BF16、结构化稀疏；MIG 可将 1 卡切 7 实例（多租户）；A800 为出口管制下首款中国特供（算力不变、NVLink 限速）。

### 3.5 Hopper（2022–2024）：Transformer 引擎与大模型时代

| 型号 | FP64 | FP32 | BF16 dense | FP8 dense | 显存 | 带宽 | TDP | NVLink | 参考价 |
|---|---|---|---|---|---|---|---|---|---|
| **H100 SXM5** | 34 | 67 | 990 | 1,979 | 80GB HBM3 | 3.35 TB/s | 700W | 900 GB/s | ~$25–30K |
| H100 PCIe | 26 | 51 | 756 | 1,513 | 80GB HBM2e | 2.0 TB/s | 350W | 600 GB/s | ~$25K |
| H100 NVL | 34 | 67 | 835 | 1,670 | 94GB HBM3 | 3.9 TB/s | 350–400W | 600 GB/s | ~$30K |
| **H200 SXM** | 34 | 67 | 990 | 1,979 | **141GB HBM3e** | **4.8 TB/s** | 700W | 900 GB/s | ~$30–40K |
| H800 SXM（中国） | 34 | 67 | 990 | 1,979 | 80GB HBM3 | 3.35 TB/s | 700W | **限速400GB/s** | ~$30K+ |
| **H20（中国）** | 1 | 44 | 148 | 296 | **96GB HBM3** | **4.0 TB/s** | 400W | 900 GB/s | ~$12–15K |

**要点**：第4代 TC + Transformer Engine 首次支持 FP8 训练（LLM 训练事实标准）；H100 是 2023–2025 全球大模型训练的绝对主力（Llama 3、Grok 均以其集群训练）；H200 保持算力不变、升级 HBM3e 主打推理带宽；H20 为合规特供——FP8 算力砍至 H100 的 ~15%，但保留 96GB/4TB/s 高带宽显存，适配"算力受限但带宽敏感"的推理场景。

### 3.6 Blackwell（2024–2026）：双 die 与机柜级交付

| 型号 | FP32 | BF16 dense | FP8 dense | FP4 dense | 显存 | 带宽 | TDP | NVLink5 |
|---|---|---|---|---|---|---|---|---|
| B200 SXM | ~80 TF | 2,250 TF | 4,500 TF | 9,000 TF | 192GB HBM3e | 8 TB/s | 1,000W | 1.8 TB/s |
| B100（过渡） | ~80 TF | 2,250 TF | 4,500 TF | 9,000 TF | 192GB HBM3e | 8 TB/s | 700W | 1.8 TB/s |
| **GB300/B300（Blackwell Ultra）** | ~80 TF | ~1.5× B200 | — | ~15,000 TF（稀疏口径） | **288GB HBM3e** | ~8 TB/s | ~1,400W | 1.8 TB/s |
| RTX PRO 6000 Server | ~125 TF | — | ~500 TF | ~2,000 TF | 96GB GDDR7 ECC | 1.8 TB/s | 600W | 无（PCIe 5.0） |

**要点**：B200 由两颗 die 经 10TB/s die-to-die 互联封装（2,080 亿晶体管）；第5代 TC 新增 FP4/FP6 微缩放格式；GB200 = 1×Grace CPU + 2×B200，GB200 NVL72 = 36 Grace + 72 B200 组成单 NVLink 域机柜（FP4 算力 ~1.4 EFLOPS、~$3M+）；B300/GB300 显存增至 288GB 主打超长上下文推理。

### 3.7 Grace 超芯片与 Rubin 路线

| 平台 | 构成 | 亮点 | 交付 |
|---|---|---|---|
| GH200 | 72核 Grace Arm CPU + H100（96/141GB HBM3/e） | NVLink-C2C 900GB/s 一致性互连，CPU 480GB LPDDR5X 可作 GPU 扩展内存 | 单卡/整机 |
| GB200 NVL72 | 36×Grace + 72×B200 | 13.5TB HBM3e + 17TB LPDDR5X，~120kW/柜 | 整柜 ~$3–3.5M |
| GB300 NVL72 | 36×Grace + 72×B300 | 20TB+ HBM3e | 整柜 |
| **Vera Rubin NVL144**（2026 H2） | Vera CPU（88 Arm 核）+ 144×Rubin GPU | HBM4 288GB/GPU、NVLink6、FP4 ~50 PF/GPU（待核实） | 机柜级 |
| **Rubin Ultra NVL576**（2027） | 4-die Rubin GPU | 1TB HBM4e/GPU、Kyber 机架 | 机柜级 |
| Feynman（2028） | — | 路线图已公布，细节未公开 | — |

### 3.8 DGX/HGX 系统形态演进

| 系统 | GPU | 年份 | 算力 | 售价 |
|---|---|---|---|---|
| DGX-1 | 8×P100 | 2016 | 170 TF FP16 | $129,000 |
| DGX-1V | 8×V100 | 2017 | 1 PF FP16 | $149,000 |
| DGX-2 | 16×V100 + NVSwitch | 2018 | 2 PF FP16 | $399,000 |
| DGX A100 | 8×A100 | 2020 | 5 PF FP16 | $199,000 |
| DGX H100 | 8×H100 | 2023 | 32 PF FP8 | ~$400K |
| DGX B200 | 8×B200 | 2025 | 72 PF FP8（稀疏） | ~$500K+ |
| DGX GB200/NVL72 | 72×B200 + 36 Grace | 2025 | 1.4 EF FP4（稀疏） | ~$3M+ |

**形态启示**：9 年间 DGX 单机算力提升 ~8,000 倍，而售价仅涨 ~23 倍——**单位算力成本下降约两个数量级**，这是大模型经济可行性的底层原因。

## 四、消费级 GeForce 详解

### 4.1 Pascal（GTX 10 系，2016–2017）

| 显卡 | 芯片 | CUDA | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|---|
| Titan Xp | GP102 满血 | 3840 | ~12.1 TF | 12GB GDDR5X | 548 GB/s | 250W | $1,200 |
| GTX 1080 Ti | GP102 | 3584 | ~11.3 TF | 11GB GDDR5X | 484 GB/s | 250W | $699 |
| GTX 1080 | GP104 | 2560 | ~8.9 TF | 8GB GDDR5X | 320 GB/s | 180W | $599 |
| GTX 1070 | GP104 | 1920 | ~6.5 TF | 8GB GDDR5 | 256 GB/s | 150W | $379 |
| GTX 1060 6GB | GP106 | 1280 | ~4.4 TF | 6GB GDDR5 | 192 GB/s | 120W | $249 |
| GTX 1050 Ti | GP107 | 768 | ~2.1 TF | 4GB GDDR5 | 112 GB/s | 75W | $139 |

**要点**：无 RT/Tensor Core，不支持 DLSS；16nm FinFET 使能效跨代提升；至今仍是大量存量 AI 开发者的入门卡（二手市场百元级，可跑 7B 量化模型）。

### 4.2 Turing（RTX 20/GTX 16 系，2018–2019）

| 显卡 | 芯片 | CUDA | RT/Tensor | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|---|---|
| Titan RTX | TU102 满血 | 4608 | 72/576 | ~16.3 TF | 24GB GDDR6 | 672 GB/s | 280W | $2,499 |
| RTX 2080 Ti | TU102 | 4352 | 68/544 | ~13.4 TF | 11GB GDDR6 | 616 GB/s | 250W | $999 |
| RTX 2080 | TU104 | 2944 | 46/368 | ~10.1 TF | 8GB GDDR6 | 448 GB/s | 215W | $699 |
| RTX 2070 | TU106 | 2304 | 36/288 | ~7.5 TF | 8GB GDDR6 | 448 GB/s | 175W | $499 |
| RTX 2060 | TU106 | 1920 | 30/240 | ~6.5 TF | 6GB GDDR6 | 336 GB/s | 160W | $349 |
| GTX 1660 Ti | TU116 | 1536 | 无 | ~5.5 TF | 6GB GDDR6 | 288 GB/s | 120W | $279 |
| GTX 1650 | TU117 | 896 | 无 | ~3.0 TF | 4GB GDDR5 | 128 GB/s | 75W | $149 |

**要点**：硬件光追（第1代 RT Core）与 Tensor Core（第2代）首次进入消费级，DLSS 1.0→2.x 开创 AI 超分；GTX 16 系砍 RT/Tensor 守低价。

### 4.3 Ampere（RTX 30 系，2020–2022）

| 显卡 | 芯片 | CUDA | RT/Tensor | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|---|---|
| RTX 3090 Ti | GA102 | 10752 | 84/336 | ~40.0 TF | 24GB GDDR6X | 1,008 GB/s | 450W | $1,999 |
| **RTX 3090** | GA102 | 10496 | 82/328 | ~35.6 TF | **24GB GDDR6X** | 936 GB/s | 350W | $1,499 |
| RTX 3080 Ti | GA102 | 10240 | 80/320 | ~34.1 TF | 12GB GDDR6X | 912 GB/s | 350W | $1,199 |
| RTX 3080 | GA102 | 8704 | 68/272 | ~29.8 TF | 10GB GDDR6X | 760 GB/s | 320W | $699 |
| RTX 3070 | GA104 | 5888 | 46/184 | ~20.3 TF | 8GB GDDR6 | 448 GB/s | 220W | $499 |
| RTX 3060 | GA106 | 3584 | 28/112 | ~12.7 TF | **12GB GDDR6** | 360 GB/s | 170W | $329 |
| RTX 3050 | GA106 | 2560 | 20/80 | ~9.1 TF | 8GB GDDR6 | 224 GB/s | 130W | $249 |

**要点**：FP32 名义翻倍（SM 双发射）；**RTX 3090 的 24GB 大显存使其成为至今性价比最高的个人 LLM 微调/推理卡**（二手 ~$600–800 可跑 70B Q4 量化）；3060 12GB 也是小模型推理甜点。

### 4.4 Ada Lovelace（RTX 40 系，2022–2024）

| 显卡 | 芯片 | CUDA | RT/Tensor | FP32 | AI TOPS | 显存 | 带宽 | TGP | MSRP |
|---|---|---|---|---|---|---|---|---|---|
| **RTX 4090** | AD102 | 16384 | 128/512 | ~82.6 TF | ~1,321 | **24GB GDDR6X** | 1,008 GB/s | 450W | $1,599 |
| RTX 4080 Super | AD103 | 10240 | 80/320 | ~52.2 TF | ~836 | 16GB GDDR6X | 736 GB/s | 320W | $999 |
| RTX 4070 Ti Super | AD103 | 8448 | 66/264 | ~44.1 TF | ~706 | 16GB GDDR6X | 672 GB/s | 285W | $799 |
| RTX 4070 Super | AD104 | 7168 | 56/224 | ~35.5 TF | ~568 | 12GB GDDR6X | 504 GB/s | 220W | $599 |
| RTX 4070 | AD104 | 5888 | 46/184 | ~29.1 TF | ~466 | 12GB GDDR6X | 504 GB/s | 200W | $599 |
| RTX 4060 Ti | AD106 | 4352 | 34/136 | ~22.1 TF | ~353 | 8/16GB GDDR6 | 288 GB/s | 160W | $399/$499 |
| RTX 4060 | AD107 | 3072 | 24/96 | ~15.1 TF | ~242 | 8GB GDDR6 | 272 GB/s | 115W | $299 |

**要点**：第3代 RT + 第4代 Tensor（FP8）；DLSS 3 光流帧生成（Ada 独占）；4N 制程能效比大幅领先 30 系；**4090 是出口管制下中国消费级 AI 算力的事实上限**（另有 4090D 特供版），至今仍被广泛用于 13B–34B 模型微调的低成本方案。

### 4.5 Blackwell（RTX 50 系，2025）

| 显卡 | 芯片 | CUDA | RT/Tensor | FP32 | AI TOPS | 显存 | 带宽 | TGP | MSRP |
|---|---|---|---|---|---|---|---|---|---|
| **RTX 5090** | GB202 | 21760 | 170/680 | ~104.8 TF | 3,352 | **32GB GDDR7** | 1,792 GB/s | 575W | $1,999 |
| RTX 5080 | GB203 | 10752 | 84/336 | ~56.3 TF | 1,801 | 16GB GDDR7 | 960 GB/s | 360W | $999 |
| RTX 5070 Ti | GB203 | 8960 | 70/280 | ~43.9 TF | 1,406 | 16GB GDDR7 | 896 GB/s | 300W | $749 |
| RTX 5070 | GB205 | 6144 | 48/192 | ~30.9 TF | 988 | 12GB GDDR7 | 672 GB/s | 250W | $549 |
| RTX 5060 Ti | GB206 | 4608 | 36/144 | ~23.7 TF | 759 | 8/16GB GDDR7 | 448 GB/s | 180W | $379/$429 |
| RTX 5060 | GB206 | 3840 | 30/120 | ~19.2 TF | 614 | 8GB GDDR7 | 448 GB/s | 145W | $299 |

**要点**：GDDR7（PAM3）、PCIe 5.0、第5代 Tensor（FP4）、DLSS 4 多帧生成；**5090 32GB 显存 + FP4 使其成为消费级最强本地 LLM 卡**（可原生跑 70B FP4/量化模型，中国市场对应 5090D v2 24GB 特供版）。

### 4.6 消费级选型要点

- **显存决定模型上限**：8GB → 7B 级；12–16GB → 13B–34B 量化；24GB → 70B Q4；32GB → 70B 更稳或 FP4 原生。
- **性价比拐点**：二手 3090（24GB）仍是每 GB 显存成本最低的 AI 卡；新卡中 5070 Ti/5060 Ti 16GB 是 AI 入门甜点。
- **无 NVLink**：40 系起消费卡无法互联，多卡训练需靠 PCIe + 框架（FSDP）弥补，效率远低于数据中心方案。

## 五、专业可视化 GPU 详解

### 5.1 命名演变（选型时避免混淆）

- **2000–2020**：Quadro 品牌（P 系列 → RTX 系列）
- **2020-10**：RTX A6000 发布，Quadro 退役 → "NVIDIA RTX Axxxx"
- **2022–2024**：去掉 A，"RTX xxxx Ada"（RTX 6000 Ada）
- **2025-03 GTC**：统一为 **NVIDIA RTX PRO xxxx Blackwell**

### 5.2 各代规格汇总

**Pascal（2016–2017）**

| 型号 | CUDA | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|
| Quadro P6000 | 3840 | ~12 TF | 24GB GDDR5X ECC | 432 GB/s | 250W | $5,999 |
| Quadro GP100 | 3584 | ~10.3 TF（FP64 5.2 TF） | 16GB HBM2 ECC | 720 GB/s | 235W | ~$7,000 |
| Quadro P5000 | 2560 | ~8.9 TF | 16GB GDDR5X ECC | 288 GB/s | 180W | ~$2,000 |

**Volta（2017–2018）**

| 型号 | CUDA/Tensor | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|
| Quadro GV100 | 5120/640 | ~14.8 TF（Tensor ~118.5 TF） | 32GB HBM2 ECC | 870 GB/s | 250W | $8,999 |
| Titan V | 5120/640 | ~14.9 TF | 12GB HBM2 | 652.8 GB/s | 250W | $2,999 |

**Turing（2018–2021）**

| 型号 | CUDA | RT/Tensor | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|---|
| Quadro RTX 8000 | 4608 | 72/576 | ~16.3 TF | 48GB GDDR6 ECC | 672 GB/s | 295W | $10,000 |
| Quadro RTX 6000 | 4608 | 72/576 | ~16.3 TF | 24GB GDDR6 ECC | 672 GB/s | 295W | $6,300 |
| Quadro RTX 5000 | 3072 | 48/384 | ~11.2 TF | 16GB GDDR6 ECC | 448 GB/s | 265W | $2,300 |
| T1000/T600/T400 | 896/640/384 | 无（TU117） | 1.2–2.5 TF | 2–8GB GDDR6 | 80–160 GB/s | 30–50W | $180–450 |

**Ampere（2020–2023）**

| 型号 | CUDA | RT/Tensor | FP32 | 显存 | 带宽 | TDP | 形态 | MSRP |
|---|---|---|---|---|---|---|---|---|
| RTX A6000 | 10752 | 84/336 | ~38.7 TF | 48GB GDDR6 ECC | 768 GB/s | 300W | 双槽 | $4,650 |
| RTX A5500 | 10240 | 80/320 | ~34.1 TF | 24GB GDDR6 ECC | 768 GB/s | 230W | 双槽 | ~$3,600 |
| RTX A5000 | 8192 | 64/256 | ~27.8 TF | 24GB GDDR6 ECC | 768 GB/s | 230W | 双槽 | $2,250 |
| RTX A4000 | 6144 | 48/192 | ~19.2 TF | 16GB GDDR6 ECC | 448 GB/s | 140W | **单槽** | ~$1,000 |
| RTX A2000 | 3328 | 26/104 | ~8.0 TF | 6/12GB GDDR6 ECC | 288 GB/s | 70W | LP | ~$450/$650 |
| RTX A1000/A400 | 2048/768 | 16/64 等 | ~6.7/2.5 TF | 8/4GB | 192/96 GB/s | 50W | LP | ~$450/$130 |
| **L40（服务器）** | 18176 | 142/568 | ~90.5 TF | 48GB GDDR6 ECC | 864 GB/s | 300W | 被动散热 | 随服务器 |
| **L4（服务器）** | 7424 | — | ~30.3 TF | 24GB GDDR6 | 300 GB/s | 72W | 单槽被动 | 随服务器 |

**Ada Lovelace（2022–2024）**

| 型号 | CUDA | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|
| RTX 6000 Ada | 18176 | ~91.1 TF | 48GB GDDR6 ECC | 960 GB/s | 300W | $6,800 |
| RTX 5880 Ada（中国） | 14080 | ~69.3 TF | 48GB GDDR6 ECC | 960 GB/s | 285W | 待核实 |
| RTX 5000 Ada | 12800 | ~65.3 TF | 32GB GDDR6 ECC | 576 GB/s | 250W | $4,000 |
| RTX 4500 Ada | 7680 | ~39.6 TF | 24GB GDDR6 ECC | 432 GB/s | 210W | $2,250 |
| RTX 4000 Ada / 4000 SFF | 6144 | ~26.7/19.2 TF | 20GB GDDR6 ECC | 360/280 GB/s | 130/70W | $1,250 |
| RTX 2000 Ada | 2816 | ~12.0 TF | 16GB GDDR6 ECC | 224 GB/s | 70W | ~$625 |
| **L40S（服务器）** | 18176 | ~91.6 TF（FP8 ~733 TF dense） | 48GB GDDR6 ECC | 864 GB/s | 350W | ~$7–9K（渠道） |

**Blackwell（2025–2026）**

| 型号 | CUDA | RT/Tensor | FP32 | 显存 | 带宽 | TDP | MSRP |
|---|---|---|---|---|---|---|---|
| RTX PRO 6000 工作站版 | 24064 | 188/752 | ~125 TF | **96GB GDDR7 ECC** | 1,792 GB/s | 600W | ~$8.5–10K |
| RTX PRO 6000 Max-Q | 24064 | 188/752 | ~95 TF | 96GB GDDR7 | 1,792 GB/s | 300W | 待核实 |
| RTX PRO 6000 服务器版 | 24064 | 188/752 | ~110 TF | 96GB GDDR7 | 1,792 GB/s | 600W 可调 | 随服务器 |
| RTX PRO 5000 | 14080 | 110/440 | ~70 TF | 48GB GDDR7 | 1,344 GB/s | 300W | ~$4,500 |
| RTX PRO 4500 | ~10496 | 待核实 | ~40+ TF | 32GB GDDR7 | 896 GB/s | 200W | ~$2,600 |
| RTX PRO 4000 | ~8960 | 70/280 | ~40 TF | 24GB GDDR7 | 672 GB/s | 140W 单槽 | ~$1,500 |

### 5.3 专业卡的核心价值（vs 同芯片 GeForce）

| 维度 | 专业卡 | 消费卡同芯片 |
|---|---|---|
| 显存 | ECC、容量更大（48→96GB） | 无 ECC、容量减半 |
| 驱动 | ISV 认证（CAD/CAE/DCC 数百款）、vGPU 授权 | 游戏/创作驱动 |
| 形态 | 单槽/SFF/被动散热（服务器）齐全 | 双槽风冷为主 |
| 定位溢价 | 同芯片贵 2–5 倍 | — |

**选型启示**：纯 AI 负载（不依赖 ISV 认证/vGPU）选 GeForce/数据中心卡性价比更高；CAE 仿真、渲染农场、虚拟桌面（VDI）、边缘推理服务器（L4/L40S/RTX PRO Server）必须走专业线。

## 六、车载 DRIVE 平台详解

### 6.1 全系列规格

| 平台 | 发布/量产 | 制程 | CPU | GPU | AI 算力 | 功耗 | 安全等级 | 量产案例 |
|---|---|---|---|---|---|---|---|---|
| DRIVE PX2 AutoCruise | 2016 | TSMC 16nm | 6核（2 Denver+4 A57） | Pascal 256 CUDA | ~4 DL TOPS | 10–15W | ASIL-B/系统级 D | Tesla HW2.0 |
| DRIVE PX2 AutoChauffeur | 2016 | 16nm | 12核 | 2×Parker + 2×Pascal MXM | 24 DL TOPS | ~250W 水冷 | ASIL-D（配 Aurix） | 沃尔沃 Drive Me、Roborace |
| **DRIVE AGX Xavier** | 2018/2020 | 12nm | 8核 Carmel ARMv8.2 | Volta 512 CUDA + 64 TC + 2×NVDLA | 30 TOPS | 10–30W | **ASIL-D 系统级** | 小鹏 P7 XPILOT 3.0 |
| DRIVE AGX Pegasus | 2018/2019 | 12nm | 16核 | 2×Xavier + 2×Turing dGPU | 320 TOPS | ~500W 水冷 | ASIL-D | 早期 Robotaxi（文远知行等） |
| **DRIVE Orin（Orin X）** | 2019/2022 | Samsung 8nm | 12核 Cortex-A78AE | Ampere 2048 CUDA + 64 TC + 2×NVDLA 2.0 | **254 TOPS** | SoC 45–65W | ASIL-D | 蔚来全系（4×Orin=1016 TOPS）、理想 Max（2×Orin=508 TOPS）、小鹏 G9、小米 SU7 Max |
| **DRIVE Thor** | 2022/2025–26 | TSMC 4nm | Arm Neoverse V3AE | **Blackwell + Transformer Engine** | 1000 TFLOPS FP8 / 2000 FP4 | 150–300W（待量产核实） | ASIL-D | 极氪首发、比亚迪、小鹏、昊铂、文远知行 |

### 6.2 每代演进要点

- **PX2（2016）**：首次把 CNN 感知流水线塞进车里，混合 SoC+dGPU 水冷方案验证可行性。
- **Xavier（2018）**：首款正向自研车规 SoC——Carmel CPU + Volta GPU + NVDLA 专用加速器，30W 实现 30 TOPS，开启 L2+ 高速 NOA 量产时代。
- **Orin（2022）**：行业事实标准。单芯 254 TOPS，支持 1/2/4 芯灵活堆叠（254/508/1016 TOPS），覆盖从城市 NOA 到 Robotaxi；中国新势力全系标配。
- **Thor（2025–）**：**舱驾一体中央计算**——单 SoC 同时跑智能座舱 + 仪表盘 + 端到端自动驾驶大模型；Blackwell GPU + Transformer Engine 原生支持 FP8/FP4 多模态模型，硬件级 MIG/虚拟化安全分区。

## 七、机器人与边缘 Jetson 平台详解

### 7.1 全系列规格

| 模组 | 年份 | GPU | CPU | AI 算力 | 内存/带宽 | 功耗 | 模组价（千片） |
|---|---|---|---|---|---|---|---|
| Jetson TX1 | 2015 | Maxwell 256 | 4×A57 | 1.02 TF FP16 | 4GB LPDDR4 / 25.6GB/s | 10–15W | $299 |
| Jetson TX2 | 2017 | Pascal 256 | 2×Denver2+4×A57 | 1.33 TF FP16 | 8GB LPDDR4 / 59.7GB/s | 7.5–15W | $399 |
| Jetson Nano 4GB | 2019 | Maxwell 128 | 4×A57 | 472 GF FP16 | 4GB / 25.6GB/s | 5–10W | $129（套件 $99） |
| Jetson Nano 2GB | 2020 | Maxwell 128 | 4×A57 | 472 GF | 2GB / 25.6GB/s | 5–10W | 套件 $59 |
| Jetson AGX Xavier 32GB | 2018 | Volta 512+64TC+2×NVDLA | 8×Carmel | 32 TOPS | 32GB LPDDR4x / 136.5GB/s | 10–30W | $999 |
| Jetson Xavier NX 8/16GB | 2020/21 | Volta 384+48TC | 6×Carmel | 21 TOPS | 8/16GB / 51.2GB/s | 10–15W | $399/$499 |
| **Jetson AGX Orin 64GB** | 2022 | Ampere 2048+64TC+2×NVDLA2 | 12×A78AE | **275 TOPS** | 64GB LPDDR5 / 204.8GB/s | 15–60W | $1,599（套件 $1,999） |
| Jetson AGX Orin 32GB | 2022 | Ampere 1792+56TC | 8×A78AE | 200 TOPS | 32GB / 204.8GB/s | 15–40W | $899 |
| Jetson Orin NX 16/8GB | 2023 | Ampere 1024+32TC | 8/6×A78AE | 100/70 TOPS | 16/8GB / 102.4GB/s | 10–25W | $599/$399 |
| Jetson Orin Nano 8/4GB | 2023 | Ampere 1024/512+32/16TC | 6×A78AE | 40/20 TOPS | 8/4GB / 68/34GB/s | 7–15W | $249/$199 |
| **Jetson Thor** | 2024/2025–26 | **Blackwell + Transformer Engine** | Arm Neoverse V3AE | **~800–1000+ TFLOPS FP4/FP8** | 128GB LPDDR5X / 273GB/s | ~100–130W | 套件 $3,499 |

> Jetson Thor 官方公布值：AGX Thor 开发套件 128GB LPDDR5X、~273GB/s 内存带宽、2070 TFLOPS FP4（官方口径上限，标称 1000+ TFLOPS 档位视精度），价格 $3,499（2025-08 发布）。量产模组价待核实。

### 7.2 每代演进要点

- **TX1/TX2（2015–2017）**：嵌入式 AI 模组形态奠基；Denver 异构 CPU + Pascal，开创无人机/巡检机器人边缘视觉市场。
- **Nano（2019）**：$99 开发套件把 CUDA 生态带入教育与 Maker 市场。
- **Xavier 家族（2018–2021）**：Tensor Core + NVDLA 进入机器人；NX 用 Nano 同封装实现 ~40× 性能，成为 AMR/医疗设备首选。
- **Orin 家族（2022–2023）**：同一 Ampere 架构覆盖 7W/20 TOPS → 60W/275 TOPS 全档位，是当前工业机器人、清洁机器人、机器狗的标准硬件。
- **Thor（2025–）**：面向具身智能/人形机器人，本地运行 VLA（Vision-Language-Action）大模型 + 全身实时运控闭环。

### 7.3 Isaac / GR00T 与硬件协同

```
云端（DGX/HGX）           边缘（Jetson）
Isaac Sim / Lab          Isaac ROS（NITROS 零拷贝）
Omniverse 仿真+合成数据 →  TensorRT 量化导出 → GR00T/VLA 端侧推理（50–100Hz 控制闭环）
RL 策略训练
```

- **Isaac Sim/Lab**：Omniverse 物理仿真 + 域随机化合成数据，云端并行训练数万智能体。
- **Isaac ROS**：GPU 零拷贝流水线，把深度相机/IMU/LiDAR 的感知、SLAM、避障直接跑在 CUDA/Tensor Core 上，亚毫秒延迟。
- **GR00T**：人形机器人多模态具身基础模型；Jetson Thor 的 Blackwell FP4/FP8 算力是其在端侧闭环运行的前提（传统 Orin 承载"大模型+实时运控"存在吞吐瓶颈）。

## 八、关键选型参数解读

影响业务选择的"超参数"，按对 AI 负载的重要性排序：

### 8.1 显存容量（第一约束）

**决定"模型能否装下"**，是硬门槛而非性能参数。

- 推理：BF16 每 1B 参数 ≈ 2GB 权重；加 KV cache 与框架开销后经验值 **模型参数量(B) × 2.5GB ≈ 所需显存**（BF16、中等上下文）。量化后近似等比例缩小（FP8 ≈ ×1.2GB/B，FP4/INT4 ≈ ×0.7GB/B）。
- 训练（混合精度 + Adam）：约 **16 × 参数字节**（参数 2B + 梯度 2B + FP32 master 4B + Adam m/v 8B）再叠加激活；ZeRO-3/FSDP 可将参数类状态均摊到 N 卡。

### 8.2 显存带宽（推理吞吐主因）

Decode 阶段 memory-bound：**单 token 吞吐 ≈ 带宽 ÷ 每 token 读取字节数**。同算力下带宽翻倍，推理吞吐近似翻倍——这是 H200（4.8TB/s）相比 H100（3.35TB/s）纯靠显存升级提升推理 ~1.4–1.9× 的原因，也是 H20"砍算力保带宽"的产品逻辑。

### 8.3 算力精度矩阵

| 精度 | 引入架构 | 典型用途 | 代际提升规律 |
|---|---|---|---|
| FP64 | 全代（数据中心旗舰半速） | HPC、CAE、科学计算 | 增长最慢（5.3→40 TF） |
| FP32 | 全代 | 图形、传统训练 | ~7.5×/10年 |
| TF32 | Ampere | FP32 代码零改训练 | 新精度即 8× FP32 |
| FP16/BF16 Tensor | Volta→ | 训练/推理主力 | 125→2,250 TF（18×） |
| FP8 | Hopper | LLM 训练/推理 | H100 1,979 → B200 4,500 TF |
| FP4 | Blackwell | 推理（微缩放） | 9,000 TF dense |
| INT8/INT4 | Turing→ | 推理量化 | 与 Tensor 算力同趋势 |

**选型换算**：比较两代卡时，先确认你的框架实际能用到哪个精度——FP4 数字再高，PyTorch 训练用不上就只与推理引擎（TRT-LLM/vLLM FP4）相关。

### 8.4 互联（多卡扩展的天花板）

| 层级 | 技术 | 带宽 | 影响 |
|---|---|---|---|
| 单卡内 | — | — | — |
| 机内 | NVLink 桥/域 | 160→1,800 GB/s | 张量并行（TP）是否可行；无 NVLink 的多卡训练效率骤降 |
| 机柜级 | NVLink Switch | NVL72 全域 | 70B+ 模型 TP=72 成为可能 |
| 跨机 | InfiniBand/RoCE | 200–800 Gb/s | 数据并行/流水线并行；万卡集群的决定性成本项 |

**经验法则**：TP 必须留在 NVLink 域内；消费卡（无 NVLink，PCIe ~64GB/s）多卡训练仅适合流水线/数据并行或全参数分片（FSDP），通信敏感场景效率远低于数据中心。

### 8.5 功耗与 TCO

- 单卡 TDP：边缘 5–60W → 推理卡 70–400W → 训练卡 400–1,400W → 机柜 120–140kW。
- TCO 构成：CapEx（卡 + 整机溢价）+ 电力（PUE 1.2–1.5）+ 机位 + 网络。粗略估算：**每 1W IT 功耗全年电费 ≈ ¥6–9**（视电价/PUE），一张 700W H100 年电费约 ¥5K–8K，远低于其购置成本——**训练卡的瓶颈在 CapEx 与利用率，而非电费**；但推理业务 7×24 满载下电费占比显著上升，能效比（TOPS/W）成为关键指标。

### 8.6 价格量级速查（行业参考，波动大，待逐案核实）

| 层级 | 代表 | 单价区间 |
|---|---|---|
| 消费级 | RTX 5060–5090 | $299–$1,999 |
| 专业工作站 | RTX PRO 4000–6000 | $1,500–$10,000 |
| 数据中心推理 | L4/T4/L40S | $2K–$9K |
| 数据中心训练 | A100→H100→B200 | $10K–$40K |
| 整机 | DGX 系列 | $129K–$500K+ |
| 机柜 | NVL72 | ~$3M+ |
| 云租赁 | H100 按需 | ~$2–4/GPU·h（现货，2025 水位） |

## 九、业务选型决策框架

### 9.1 第一步：按负载类型分流

```
负载是什么？
├─ LLM/生成式 AI 训练 → 9.2
├─ LLM/生成式 AI 推理 → 9.3
├─ 传统 CV/推荐/搜推广训练 → 9.4
├─ 科学计算/HPC → 9.5
├─ 图形渲染/仿真/数字孪生 → 9.6
├─ 车载自动驾驶 → 9.7
└─ 机器人/边缘推理 → 9.8
```

### 9.2 LLM 训练

**算式**：`所需 GPU 数 ≈ 6·N·D / (单卡峰值 FLOPS × MFU × 目标时长秒)`，MFU 大集群现实值 30–50%。

**显式门槛**：单卡显存必须放得下"分片后参数 + 激活"。70B BF16 训练在 8×H100（640GB）上需 ZeRO-3/FSDP + activation checkpointing；405B 级别必须跨节点（参考：Llama-3-405B = 16,384×H100，MFU 38–43%）。

**选型梯度**：

| 规模 | 推荐 | 理由 |
|---|---|---|
| ≤1.5B 微调/实验 | 1–2× RTX 4090/5090 或 A100 | 显存够、成本低；LoRA 更省 |
| 7B–13B 全参微调 | 4–8× A100/H100 | 需 NVLink 域内 TP |
| 30B–70B 预训练/微调 | 8×H100/H200 起步，B200 更优 | 显存+带宽双约束 |
| 100B–400B+ 预训练 | HGX H100/H200 多节点 或 GB200 NVL72 | 必须 NVLink 域 + IB 组网 |
| 中国合规约束 | H20 集群（推理强）/A800（存量）或国产替代评估 | 训练需大规模 H20 堆叠，互联未限速是优势 |

**避坑**：消费卡堆叠训练 70B+ 不可行（无 NVLink、PCIe P2P 受限、无 ECC）；B200 单卡虽强但训练性价比的胜负手在 NVL72 机柜级 NVLink 域，单卡/8 卡 B200 对 H100 8 卡训练的提升主要来自显存与 FP8 吞吐。

### 9.3 LLM 推理

**算式**：
- 显存下限 ≈ `参数量 × 字节/参数 × 1.15`（权重 + 开销），KV cache 另算：`2 × 层数 × KV头数 × 头维 × 字节 × batch × seq_len`。
- 吞吐近似 ∝ 显存带宽 ÷ 每 token 读取字节（decode）；prefill ∝ Tensor 算力。

**选型梯度**：

| 场景 | 推荐 | 理由 |
|---|---|---|
| 个人/小团队 7B–14B | RTX 4060 Ti 16GB / 5060 Ti 16GB / 4070 Ti S | 16GB 显存甜点，桌面即可 |
| 个人 30B–70B（Q4/FP8） | RTX 3090/4090/5090（24–32GB） | 每 GB 显存成本最低 |
| 企业在线服务 8B | L4 / L40S / RTX PRO 6000 Server | 72W 单槽 L4 每机架密度最高；L40S FP8 强 |
| 企业 70B 高并发 | 4–8× H100/H200 或 8×L40S | H200 141GB/4.8TB/s 带宽最优 |
| 200B–700B（DeepSeek/Qwen-MoE） | 8×H200（1.1TB）或 NVL72 | 大 NVLink 域做 TP，FP8/FP4 |
| 中国合规 | H20 集群（96GB/4TB/s 保带宽） | 算力弱但带宽足，恰好匹配推理 |
| 边缘/端侧 | Jetson Orin NX/Nano、Thor | 见 9.8 |

**关键权衡**：延迟敏感（小 batch 在线）→ 优先带宽；吞吐敏感（离线批量）→ 优先算力×并发密度；长上下文 → 显存与 KV cache 池优先（H200/B300/RTX PRO 6000 的大显存价值在此）。

### 9.4 传统 CV / 推荐 / 搜推广

- **CV 训练**（检测/分割）：显存需求中等（8–24GB），算力敏感 → RTX 4090/5090、A10/A40、L40S 性价比高；A100/H100 只在超大 batch 或 3D/视频模型时必要。
- **推荐系统**：特征是 embedding 巨大 + 显存带宽敏感 → NVIDIA Merlin 官方推荐 H100/A100 + HugeCTR；单机 embedding 放不下时用 A100 80GB/H200 141GB 的大显存版本比堆小卡更优。
- **推理服务化**：T4/A2/L4 低功耗卡在每机架 TOPS/W 上长期占优（视频结构化、OCR 流水线的经典配置）。

### 9.5 科学计算 / HPC

唯一看 FP64 的领域：P100（5.3）→ V100（7.8）→ A100（9.7）→ H100（34）→ B200（40 TF）。分子动力学、CFD、天气（GROMACS/ANSYS/WRF）选 H100/B200 或 GH200（CPU-GPU 一致性内存利好前处理）；双精度不敏感的 AI4Science 可降级用推理卡省钱。

### 9.6 图形渲染 / 仿真 / 数字孪生 / VDI

- **渲染农场/Omniverse**：L40/L40S、RTX 6000 Ada、RTX PRO 6000（96GB 装超大场景）。
- **CAE/EDA 工作站**：RTX A5000→RTX 5000 Ada→RTX PRO 5000（ISV 认证 + ECC 是硬要求）。
- **VDI/vGPU**：A16（4 芯分片）、L4、RTX PRO Server + vGPU 授权；按每用户 1–4GB 显存切片。
- **轻量 CAD**：T400→RTX A400→RTX 2000 Ada 低功耗 LP 形态。

### 9.7 车载自动驾驶

按整车 E/E 架构与智驾等级：

| 目标 | 平台 | 配置 |
|---|---|---|
| L2+ 高速 NOA | 单 Orin（254 TOPS） | 11V 视觉 + 前向雷达，BEV+Transformer |
| 城市 NOA | 双 Orin（508 TOPS） | +激光雷达/4D 毫米波 |
| 准 L3/Robotaxi 冗余 | 4×Orin（1016 TOPS）或 Thor | 全冗余感知+域控热备 |
| 舱驾一体中央计算（2025+ 新平台） | 单/双 Thor | 座舱 + 仪表 + 智驾单芯片分区运行 |

约束维度：ASIL-D 功能安全、车规温度（-40~105°C）、整板功耗预算（风冷 ≤65W，更高需水冷）、供给周期（Orin 成熟、Thor 2025–26 爬坡）。

### 9.8 机器人 / 边缘 AI

| 场景 | 选型 | 说明 |
|---|---|---|
| 教育/创客/轻量 IoT | Orin Nano（$199–249 模组） | 20–40 TOPS，跑 YOLO/小 VLM |
| AMR/清洁机器人/机器狗 | Orin NX 8/16GB | 70–100 TOPS，多路相机+LiDAR SLAM |
| 工业质检/多相机网关 | AGX Orin 32/64GB | 200–275 TOPS，JetPack 全栈 |
| 人形机器人/具身智能 | Jetson Thor | 本地 VLA 大模型 + 实时全身运控 |
| 固定边缘服务器（有市电） | L4 / RTX PRO 4000/6000 Server | 半高单槽，塞 2U 边缘机 |

### 9.9 采购方式决策

| 方式 | 适用 | 备注 |
|---|---|---|
| 自建裸机 | 年利用率 >60%、数据合规要求高 | H100 级需机房改造（700W/卡） |
| 云按需 | 弹性/短期/POC | H100 ~$2–4/GPU·h 现货水位（2025，波动大） |
| 云预留/裸金属长租 | 稳定训练负载 | 通常 6–12 月锁定，折扣 30–50% |
| DGX Cloud / NCP（CoreWeave 等） | 无运维团队但要集群规模 | 含 InfiniBand 组网与调度栈 |
| 二手/存量 | 预算受限的推理/微调 | 3090/A100 二手市场成熟，注意矿卡与保修 |

## 十、典型应用场景配置案例

### 10.1 大模型预训练集群：Llama-3-405B 级别

**公开配置**（Meta，2024）：16,384× H100 SXM（8 卡 HGX × 2,048 节点）+ RoCE/InfiniBand 双平面，训练 54 天、15.6T tokens，MFU 38–43%，4D 并行（TP×CP×PP×DP）。

**为什么这样选**：
- H100 而非 A100：FP8 Transformer Engine + 900GB/s NVLink4，等效训练吞吐 ~3–4×。
- 大规模单一 NVLink+IB 域：405B BF16 权重 ~810GB + 优化器状态，TP 必须留在 8 卡 NVLink 域内，跨节点走 PP/DP。
- 效果：单次训练成本千万美元级；Meta 双 24,576 卡集群同时支撑训练与推理弹性调度。

**中小团队平替**：7B–13B 预训练用 8×H100 单节点（约 $30–40 万 CapEx 或 ~$20–30/GPU·h 云长租），MFU 可达 35–45%；70B 全参训练建议 ≥2 节点 16 卡或租整柜。

### 10.2 LLM 在线推理服务：70B 模型高并发

**典型配置**：8× H200（或 2×4 H100 NVL）单节点，TensorRT-LLM/vLLM + FP8 权重 + FP8 KV cache。

**为什么**：70B FP8 权重 ~70GB，8 卡余下 ~800GB 全做 KV cache 池 → 支撑高并发长上下文；H200 的 4.8TB/s 带宽直接提升 decode 吞吐 ~1.4–1.9× vs H100。实测口径下 H200 跑 Llama-2-70B 吞吐较 H100 提升约 45%（NVIDIA 官方数据）。

**效果**：单节点可服务 70B 模型数百并发、P99 延迟 <2s（视输出长度）；对比用 8×A100 的方案，同等并发需 2 节点，TCO 反而更高。

**低成本版**：4×RTX 4090/5090（96–128GB）跑 70B Q4/FP8 + vLLM，吞吐为 H100 节点的 ~1/5–1/8，但 CapEx 仅 1/15——适合内部工具、低 QPS 场景。

### 10.3 超大规模 MoE 推理：DeepSeek-V3/R1（671B，37B 激活）

**公开建议配置**：8× H200（1.1TB 显存）或 16× H100 双节点，SGLang/vLLM + FP8；生产环境推荐 NVL72。

**为什么**：MoE 全量权重必须驻留显存（671B FP8 ≈ 700GB）→ 显存是唯一硬门槛；专家路由带来不规则访问 → 高带宽 + 大 NVLink 域收益最大；GB200 NVL72 用 72 卡 NVLink 域 + FP4 把单 token 成本再压低一个量级。

**效果**：H200 8 卡可跑 FP8 全量模型做中高并发服务；NVL72 机柜在同等功耗下吞吐为 H100 集群数倍（NVIDIA 官方宣称推理性能 30×，含 FP4 与机柜级带宽红利）。

### 10.4 搜推广/推荐系统

**典型配置**：训练侧 8×A100 80GB 或 H100 + HugeCTR/Merlin；推理侧 T4/L4 集群。

**为什么**：推荐模型瓶颈是 TB 级 embedding 表与稀疏访存——大显存版本卡（80GB/141GB）单机即可容纳更大 embedding 分片，减少跨节点 all-to-all；推理侧 QPS 极高但单请求计算小，低功耗高密度的 T4/L4 每机架 TOPS/W 最优。

**效果**：Meta/字节公开实践均显示 GPU 推荐训练较 CPU 集群提速 10×+，推理成本随 INT8 量化进一步下降。

### 10.5 自动驾驶量产方案：城市 NOA 车型

**典型配置**：双 Orin（508 TOPS）+ 11 摄像头 + 1 激光雷达 + 5 毫米波（理想 Max / 蔚来 NAD / 小鹏 XNGP 方案族）。

**为什么**：城市 NOA 的 BEV+Transformer+占用网络感知栈实测需 ~200–400 TOPS 冗余预算；双 Orin 提供算力冗余与功能安全备份；ASIL-D 认证与 -40°C 车规是数据中心卡无法替代的硬约束。

**下一代**：Thor 单芯片 1000+ TOPS，舱驾一体（座舱域 + 智驾域硬件隔离共芯片），整车 BOM 与线束复杂度下降，2025–26 起在极氪/比亚迪等量产。

### 10.6 人形机器人：具身智能整机

**典型配置**：Jetson Thor（128GB，~100–130W）+ Isaac ROS + GR00T N 系列 VLA 模型；云端 Isaac Sim/Lab 做 Sim2Real 训练。

**为什么**：人形机器人需在 100W 级功耗内同时跑多路相机感知、VLA 大模型推理（几十亿参数级）、50–100Hz 全身控制闭环——只有 Blackwell FP4/FP8 Tensor Core + 128GB 统一内存的 Thor 能在端侧满足；传统 Orin 方案需"大模型上云"牺牲实时性。

**效果**：Figure、Agility、宇树、傅利叶等头部人形机器人厂商均采用 NVIDIA 训练 + 端侧推理链路；Thor 使本地闭环摆脱对云端网络的依赖。

### 10.7 科学计算：分子动力学/气候模拟

**典型配置**：8×H100 HGX 节点或 GH200 超芯片节点 + InfiniBand。

**为什么**：FP64 是硬指标——H100 FP64 34 TFLOPS ≈ 3.5× A100；GH200 的 480GB CPU 内存与 GPU 一致性互连利好大规模网格前处理与数据调度；Frontier/Aurora 级超算之外，H100/H200 集群是商业 HPC 主流。

**效果**：GROMACS、NAMD 等主流代码在 H100 上较 V100 提速 4–6×（官方 ISV 数据），单次千万原子级模拟周期从周缩至天。

### 10.8 中小企业"一房机柜"混合负载

**典型配置**：2U 服务器 ×2，每台 4×L40S（或 2×RTX PRO 6000 Server），风冷即可。

**为什么**：L40S 48GB + FP8 733 TFLOPS dense，单卡可服务 70B FP8 推理、同时兼顾渲染/视频/轻量训练；相比 H100 节点省一半 CapEx 且无需液冷；是"既要推理服务、又要偶尔微调、还要图形能力"的混合负载甜点。

### 10.9 个人开发者/科研桌面

| 预算 | 配置 | 能做什么 |
|---|---|---|
| ~$500 | 二手 RTX 3090 24GB | 70B Q4 推理、7B LoRA、Stable Diffusion 全套 |
| ~$1,600 | RTX 4090 24GB | +13B 全参微调、ComfyUI 大工作流 |
| ~$2,000 | RTX 5090 32GB | +70B FP4 原生推理、DLSS4 游戏 |
| ~$3,500 | Jetson Thor 开发套件 | 具身智能/VLA 模型端侧原型 |

## 十一、各领域架构演进趋势分析

### 11.1 总路线图：一年一代的 AI 算力节奏

```
Pascal(2016) → Volta(2017) → Turing(2018) → Ampere(2020) → Hopper(2022)
→ Blackwell(2024) → Blackwell Ultra(2025) → Rubin(2026H2) → Rubin Ultra(2027) → Feynman(2028)
```

自 2024 年起 NVIDIA 宣布**年度迭代节奏**（此前 2–3 年一代），GTC keynote 逐年确认：GTC 2024 公布 Blackwell + Rubin 预研，Computex 2024 公布 Rubin 命名，GTC 2025 公布 Vera Rubin NVL144 / Rubin Ultra NVL576 / Feynman 2028。

### 11.2 数据中心：五个不可逆趋势

1. **精度下探即算力**：每代新精度 ≈ 2× 等效算力（FP16→FP8→FP4）。FP4 微缩放格式（NVFP4/MXFP4）在推理侧精度损失可控，Rubin 时代 FP4 将成默认推理精度；训练仍以 BF16/FP8 为主。
2. **从卡到柜再到"AI 工厂"**：NVLink 域 8 卡（HGX）→ 72 卡（NVL72 铜缆背板）→ 144/576（Rubin NVL144/576、Kyber 机架）。**单卡参数表的时代结束了**——采购、散热（液冷强制）、供电（120kW+）、组网都以机柜为单位设计。
3. **CPU-GPU 一体化**：Grace Arm CPU + GPU 经 NVLink-C2C 一致性互连，CPU 内存成为 GPU 的慢速扩展池（GH200→GB200→Vera Rubin）；x86 主机 CPU 地位被边缘化。
4. **显存比算力涨得更快**：HBM2 16GB/732GB/s → HBM3e 192–288GB/8TB/s → HBM4 288GB–1TB/13TB/s+。推理时代"带宽即吞吐"使 HBM 堆叠数成为成本大头（B200 BOM 中 HBM3e 占比过半）。
5. **合规特供常态化**：A800→H800→H20 的演进逻辑是"先限互联、再限总算力"；中国市场份额与合规线之间的博弈将长期存在，选型必须预留替代路径。

### 11.3 消费级：游戏卡变 AI 入口

- **渲染 AI 化**：DLSS 超分（Turing）→ 帧生成（Ada DLSS3）→ 多帧生成 + transformer 超分模型（Blackwell DLSS4，下放 40/30/20 系）→ 神经渲染/材质压缩方向。
- **AI PC 叙事**：NVIDIA 用 RTX AI PC 统一宣传口径（官方 AI TOPS：4090 ~1321 → 5090 3352），本地小模型推理（RTX AI Toolkit、NIM microservice）成为 40/50 系第二卖点。
- **显存军备竞赛迟缓**：8GB 守门（4060/5060）被社区诟病多年，5090 32GB 是消费级首次突破 24GB——反映 NVIDIA 刻意维持与专业/数据中心线的显存区隔。
- **供电/散热极限**：TGP 250W→575W，12V-2×6 接口与三槽+散热成为机箱设计约束。

### 11.4 专业可视化：与数据中心线合流

- **命名与定位重构**：Quadro（图形认证）→ RTX A（图形+AI 工作站）→ RTX PRO（图形+AI+服务器版三位一体）。RTX PRO 6000 的 96GB 显存、600W、服务器变体，本质是"单 die 的推理数据中心卡"。
- **趋势判断**：专业线旗舰的 AI 推理属性将持续增强（FP4、大显存），与 L40S 的边界进一步模糊；传统 ISV 认证价值集中在 CAE/EDA/医疗影像等长周期软件行业。

### 11.5 车端：舱驾一体 + 大模型上车

- **路线**：分布式 ECU → 域集中（Orin 智驾域）→ **中央计算（Thor）**：单 SoC 跑座舱+智驾+车身，硬件级虚拟化分区满足功能安全。
- **算力跃迁**：PX2 24 TOPS → Xavier 30 → Orin 254 → Thor 1000–2000 TOPS（十年 ~80×）。
- **算法牵引硬件**：端到端 Transformer/VLA 智驾栈取代分模块 CNN 流水线 → Thor 原生 Transformer Engine/FP8 正是为此设计；车端也开始跑多模态 LLM（语音助手、场景理解）。
- **中国变量**：Orin 曾垄断中国高阶智驾，Thor 时代面临地平线 J6、华为昇腾、自研芯片的正面竞争，NVIDIA 以"Thor + DRIVE OS + 全栈算法授权"对冲。

### 11.6 机器人/边缘：从 CV 模组到具身智能大脑

- **算力曲线**：TX2 1.3 TF → AGX Xavier 32 TOPS → AGX Orin 275 TOPS → Thor 1000+ TFLOPS（FP4），功耗带始终锚定 7–130W。
- **软件栈即壁垒**：JetPack + Isaac ROS + Isaac Sim 构成"云端训练—仿真合成数据—端侧部署"闭环，开发者迁移成本远高于硬件差价。
- **趋势**：Orin 解决"感知+运动"，Thor 解决"认知（VLA 大模型）+控制"；人形机器人浪潮将使 Jetson 从年百万级模组市场走向千万级。

### 11.7 十年演进规律总结

| 维度 | 规律 | 对选型的含义 |
|---|---|---|
| 算力 | 靠新精度而非 FLOPS 增长 | 买新不买旧，但确认软件栈能用上新精度 |
| 显存 | HBM 容量/带宽增速 > 算力增速 | 推理选型"带宽优先"，训练"容量门槛优先" |
| 互联 | NVLink 域每代翻倍 | 大模型训练必须为互联付费（SXM/NVL 而非 PCIe） |
| 形态 | 卡 → 节点 → 机柜 → AI 工厂 | 采购评估单位从 $/卡变为 $/PFLOPS·年（含电力与组网） |
| 软件 | CUDA→TensorRT-LLM→NIM→Isaac/GR00T | 硬件差距的 30–50% 由软件栈决定，生态锁定是真实成本 |

## 十二、附录：参考链接与置信度说明

### 12.1 官方资料（一级来源）

**数据中心**
- NVIDIA Data Center 产品总览：https://www.nvidia.com/en-us/data-center/
- Tesla P100 datasheet：https://images.nvidia.com/content/pdf/tesla/Tesla-P100-PCIe-datasheet.pdf
- Volta 架构白皮书：https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
- T4 datasheet：https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
- A100 datasheet：https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- H100/H200 产品页：https://www.nvidia.com/en-us/data-center/h100/ 、/h200/
- Blackwell 平台：https://www.nvidia.com/en-us/data-center/blackwell/ ；GB200 NVL72：https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- Grace Hopper：https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/
- DGX 系列：https://www.nvidia.com/en-us/data-center/dgx-h100/
- Blackwell 发布新闻稿：https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing
- Vera Rubin/Blackwell Ultra 新闻稿：https://nvidianews.nvidia.com/news/nvidia-announces-blackwell-ultra-vera-rubin-platforms

**消费级**
- GeForce 各代产品页：https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/ 、/40-series/ 、/30-series/
- RTX AI PC 官方 AI TOPS 口径：https://www.nvidia.com/en-us/geforce/rtx-ai-pcs/

**专业可视化**
- RTX PRO 6000 Blackwell：https://www.nvidia.com/en-us/design-visualization/products/rtx-pro-6000-blackwell/
- RTX 6000 Ada：https://www.nvidia.com/en-us/design-visualization/rtx-6000/
- L40/L40S/L4：https://www.nvidia.com/en-us/data-center/l40s/ 、/l4/
- 历代 datasheet：https://resources.nvidia.com/en-us-design-viz-techbriefs

**车端/边缘**
- DRIVE 平台：https://www.nvidia.com/en-us/self-driving-cars/drive-platform/hardware/
- DRIVE AGX：https://developer.nvidia.com/drive/agx
- DRIVE Thor 新闻稿：https://nvidianews.nvidia.com/news/nvidia-unveils-drive-thor-centralized-car-computer-unifying-safe-intelligent-automated-driving
- Jetson 模组对比：https://developer.nvidia.com/embedded/jetson-modules
- Jetson AGX Thor：https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
- Isaac 平台：https://www.nvidia.com/en-us/autonomous-machines/robotics/
- GR00T 新闻稿：https://nvidianews.nvidia.com/news/foundation-model-isaac-robotics-platform

### 12.2 第三方数据库与媒体（二级来源）

- TechPowerUp GPU Database（逐卡规格）：https://www.techpowerup.com/gpu-specs/
- Wikipedia 微架构条目：Nvidia_Tesla、Pascal/Volta/Turing/Ampere/Hopper/Blackwell/Rubin_(microarchitecture)、GeForce_10/16/20/30/40/50_series
- SemiAnalysis（GB200/机柜/云价格分析）：https://www.semianalysis.com/

### 12.3 部署案例与方法论文献

- Meta GenAI 基础设施：https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/
- Llama 3 herd paper：https://arxiv.org/abs/2407.21783 ；Llama 3.1 405B 博客：https://ai.meta.com/blog/llama-3-1-405b/
- xAI Colossus：https://x.ai/news/colossus ；NVIDIA 报道：https://blogs.nvidia.com/blog/xai-colossus/
- ZeRO 论文：https://arxiv.org/abs/1910.02054 ；DeepSpeed ZeRO：https://www.deepspeed.ai/tutorials/zero/
- Scaling laws：https://arxiv.org/abs/2001.08361 ；Chinchilla：https://arxiv.org/abs/2203.15556 ；PaLM MFU：https://arxiv.org/abs/2204.02311
- Transformer 推理算数（Kipply）：https://kipp.ly/transformer-inference-arithmetic/
- FP8 格式论文：https://arxiv.org/abs/2209.05433 ；SmoothQuant：https://arxiv.org/abs/2211.10438 ；AWQ：https://arxiv.org/abs/2306.00978 ；GPTQ：https://arxiv.org/abs/2210.17323
- DeepSeek-V3 部署：https://github.com/deepseek-ai/DeepSeek-V3
- vLLM：https://docs.vllm.ai/ ；SGLang：https://github.com/sgl-project/sglang ；TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM
- 云价格：AWS P5/P4 https://aws.amazon.com/ec2/instance-types/p5/ ；Lambda https://lambdalabs.com/service/gpu-cloud/pricing ；CoreWeave https://www.coreweave.com/pricing ；Vantage 实例价 https://instances.vantage.sh/

### 12.4 置信度分级

- **高置信度**（官方 datasheet/发布稿长期稳定值）：所有架构代际、CUDA 核心、显存容量/类型/位宽/带宽、TDP、接口、发布年份、消费级 MSRP。
- **中置信度**：数据中心卡单价（NVIDIA 不公开零售定价，为行业普遍引用区间）；A800/H800/H20/5880 Ada/5090D 等特供型号价格；Ada 系 Super 型号官方 AI TOPS。
- **待核实**（撰写时点数据仍在变化）：Rubin/Rubin Ultra 详细算力与量产时间表、RTX PRO 4500/4000 Blackwell 精确核心数与定价、Jetson Thor 量产模组价、2025–2026 云 GPU 现货价格、B300/GB300 部分稀疏 FP4 口径（不同资料 ±10%）。
- **价格波动警示**：MSRP 为首发价；矿潮期（2020–2022）、AI 缺货期（2023–2024）、出口管制变动期实际成交价可偏离 30–200%。

### 12.5 使用建议

本报告数据截至 2026 年 9 月。用于采购决策前建议：(1) 用 12.1 官方 URL 复核目标型号最新 datasheet；(2) 价格项以当期报价为准；(3) Blackwell/Rubin 世代参数仍在快速更新，优先核对官方新闻稿。

---

*报告完 · 生成于 2026-09-20*
