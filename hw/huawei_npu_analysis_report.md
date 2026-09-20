# 华为昇腾 NPU 全栈产品体系系统分析报告

> **报告日期**：2026-09-20
> **覆盖范围**：数据中心 / 消费侧 / 车端 / 边缘与机器人侧，2016–2026（约十年）
> **数据原则**：所有关键参数均附参考来源链接；华为官方未公布的数据一律标注"未公开"或"第三方估计"，不做编造。价格为非官方渠道/分析师估计时单独注明。
> **主要参考入口**：昇腾社区硬件中心 https://www.hiascend.com/hardware 、华为企业产品文档 https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909?submodel=doc

---

## 0. 执行摘要

- 华为 NPU 战略自 2018 年"达芬奇（Da Vinci）架构 + 全栈全场景 AI"发布以来，形成了**一套架构（Da Vinci 3D Cube）、四大战场（云/边/端/车）**的产品体系。昇腾 Ascend 芯片是数据中心的底座，麒麟 Kirin NPU 是端侧底座，MDC 平台基于昇腾芯片面向车规，Atlas 200/300/500 系列覆盖边缘与机器人。
- 2025-09 华为全联接大会首次公开长周期芯片路线图：**昇腾 950PR（2026Q1）→ 950DT（2026Q4）→ 960PR/DT（2027）→ 970（2028）→ 980（2029）**，转入"一年一代"节奏，并自研 HBM（HiBL/HiZQ），片间互联升级到灵衢 UB 2.0（2 TB/s 双向）([Tom's Hardware, 2026-09-18](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-details-ai-accelerator-roadmap-pulls-in-next-generation-ascend-npus-by-quarters-fp4-performance-of-the-ascend-960pr-doubles-expectations))。
- 系统级打法是"超节点（SuperPoD）+ 灵衢总线"：Atlas 950 SuperPoD 最大支持 1024 颗 950DT，总算力 1 EFLOPS（mxFP8）/2 EFLOPS（mxFP4）、1.72 PB/s 互联总带宽、256 TB 全局统一内存编址（[昇腾官网产品页](https://www.hiascend.com/hardware/cluster)）。更早的 CloudMatrix 384（384×910C）以"以规模换单芯片差距"的思路在聚合指标上超过 NVIDIA GB200 NVL72（300 vs 180 PFLOPS BF16），代价是功耗约为对方 3.9 倍（[Tom's Hardware 转述 SemiAnalysis](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power)）。
- 选型核心结论：**单芯片代差约 1–1.5 代（910C ≈ 60% H100 推理性能），但华为靠超节点规模与电力冗余弥补；选型的关键不在"单卡 TOPS"，而在模型参数量×并发量决定的显存容量与互联带宽需求，以及 CANN/CUDA 迁移成本**。

---

## 1. 十年产品谱系总览（2016–2026）

| 时间 | 数据中心 | 消费侧 | 车端 | 边缘/机器人 |
|---|---|---|---|---|
| 2016-2017 | — | 麒麟 970（首款手机 NPU，寒武纪 1A IP，2017-09 IFA 发布，Mate 10 首发） | — | 海思安防 SoC（Hi35xx NNIE 引擎） |
| 2018 | Ascend 310（2018-10 全联接大会，达芬奇首发）、Ascend 910（同期发布） | 麒麟 980（双核 NPU） | — | Atlas 200/300/500 系列规划随昇腾发布 |
| 2019 | Ascend 910 量产上市（256 TFLOPS FP16，[STH/Hot Chips 31](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/)）；Atlas 900 集群（1024×910）、Atlas 800/300 上市 | 麒麟 810/990（自研达芬奇 NPU，大核+微核） | MDC 300F（2019 HC 发布，64 TOPS INT8 稠密，[媒体规格汇总](https://www.toutiao.com/article/7679656110160167460/)） | Atlas 200 DK 开发套件 |
| 2020 | Atlas 800 训练服务器（9000）、Atlas 800 推理服务器（3000） | 麒麟 9000（5nm，达芬奇 2.0 NPU，Mate 40） | MDC 210/610 发布（2020-09-25 生态论坛，[媒体报道](https://www.toutiao.com/article/7657390733850182187/)） | Atlas 500 智能小站 |
| 2021 | Atlas 900 PoD | 麒麟芯片因制裁断供（库存期） | MDC 810 量产（2021-04，400+ TOPS，极狐阿尔法 S HI 版首发） | Atlas 500 Pro |
| 2022 | 受制裁转向国产供应链；Atlas 300I Pro/Duo（Ascend 310P）上市 | 麒麟旗舰停更 | MDC 610 规模上车（问界/阿维塔 ADS 1.0） | Atlas 200I A2（Ascend 310B） |
| 2023 | Ascend 910B 量产（SMIC N+1/N+2，~25 Da Vinci 核，[Tom's Hardware 拆解](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910)）；Atlas 900 A2 PoD | 麒麟 9000S 回归（Mate 60，SMIC N+2 7nm） | ADS 2.0（MDC 610/810）城区 NCA | Atlas 200I DK A2 开发者套件 |
| 2024 | 910B2/B3/B4 系列迭代；CANN 8.x；MindIE | 麒麟 9010（Pura 70）、9020（Mate 70） | ADS 3.0（车位到车位） | 昇腾模组进机器人本体 |
| 2025 | Ascend 910C 规模出货（双 die 封装，780 BF16 TFLOPS，[Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power)）；CloudMatrix 384 超节点；HC2025 公布 950/960/970 路线图与自研 HBM；CANN 宣布开源 | 麒麟 X90（鸿蒙 PC，MateBook Pro/Fold 2025）、麒麟 9020/9030（Mate 70/80） | 乾崑 ADS 4（MDC 810 Pro 等新一代域控） | 乐聚"夸父"人形机器人（盘古+鸿蒙，HDC 2024 亮相） |
| 2026 | 昇腾 950PR 上市（Atlas 350 卡、Atlas 650E 服务器、Atlas 950 SuperPoD 1024 卡，[昇腾官网](https://www.hiascend.com/hardware)）；960DT 提前至 2027Q1 | 麒麟 9030（Mate 80 系列，2025-11 起） | ADS 4 规模铺开 | — |

---

## 2. 达芬奇（Da Vinci）架构：一套架构打全场

达芬奇架构 2018 年发布，核心特征（[华为 HC2019 官方解读、Hot Chips 31 报告](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/)）：

- **3D Cube 矩阵计算单元**：一个 Da Vinci 核 = Cube（矩阵乘）+ Vector（向量）+ Scalar（标量）三单元 + 片上 buffer，Cube 以 16×16×16 矩阵乘为基本粒度。
- **同构缩放**：同一架构 IP 从 IoT 端（毫瓦级）一路缩放到云端（数百瓦），端/边/云共享指令集与 CANN 软件栈——这是华为与 NVIDIA（GPU 架构不下放手机）最大的路线差异。
- **演进脉络**：Da Vinci（2018，Ascend 310/910、麒麟 810/990）→ Da Vinci 2.0（2020，麒麟 9000）→ "New Da Vinci" 核（2023，910B/910C，die shot 显示约 25 核、支持 INT8/FP16/BF16，[Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910)）→ 950 代（2026，SIMD+SIMT 混合编程，新增 mxFP8/mxFP4/HiF8 低精度格式，[昇腾官网 950PR 页](https://www.hiascend.com/hardware/processor)）。

---

## 3. 数据中心产品线详解

### 3.1 芯片级（NPU 处理器）

| 芯片 | 发布/量产 | 制程 | 架构 | Dense 算力 | 片上内存 | 显存带宽 | 片间互联 | TDP | 来源 |
|---|---|---|---|---|---|---|---|---|---|
| **Ascend 310** | 2018-10 发布 | 12nm（未官方确认） | Da Vinci，约 1-2 核 | 16 TOPS INT8 / 8 TFLOPS FP16 | LPDDR4X | 未公开 | PCIe | 8 W | [华为 2018 发布报道](https://www.huawei.com/en/news/2018/10/huawei-launches-ascend-ai-processor)、[昇腾社区](https://www.hiascend.com) |
| **Ascend 310P** | 2021-22 | 未公开 | Da Vinci | 140 TOPS INT8 / 70 TFLOPS FP16 | 集成于 300I Pro（24 GB LPDDR4X） | 204.8 GB/s（整卡） | PCIe 4.0 | ≤72 W（整卡） | [Atlas 300I Pro 文档入口](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-300i-pro-pid-251052354) |
| **Ascend 310B** | 2022-23 | 未公开 | Da Vinci | 20 TOPS INT8 / 10 TFLOPS FP16 | LPDDR4X 12/8/4 GB（模组） | 未单列 | SerDes/PCIe 3.0 | 21–25 W（模组） | [Atlas 200I A2 官网规格（已核实）](https://www.hiascend.com/hardware/accelerator-module-A2) |
| **Ascend 910 / 910A** | 2019-08 上市 | TSMC N7+（7nm EUV） | 32×Da Vinci 核 | 256 TFLOPS FP16 / 512 TOPS INT8 | 4×HBM2 共 32 GB | 未单列（官方页口径随卡型） | HCCS 720 Gbps + 2×100G RoCE | 310–350 W | [STH Hot Chips 31](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/)、[华为官方新闻](https://www.huawei.com/en/news/2019/8/huawei-ascend-910-most-powerful-ai-processor) |
| **Ascend 910B** | 2023 | SMIC N+1/N+2 7nm 级 | ~25×"New Da Vinci"核（die shot） | ~320–400 TFLOPS FP16（第三方口径不一，官方未公开） | 64 GB HBM2e（第三方拆解/渠道口径） | ~1.6 TB/s（第三方） | HCCS（392 GB/s 级，第三方） | ~350 W（第三方） | [Tom's Hardware 拆解](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910)；注：官方未公布 910B 规格 |
| **Ascend 910C** | 2025Q1 起出货 | SMIC N+2 + 存量 TSMC die | 双 chiplet（2×910B die） | 780 BF16 TFLOPS | 128 GB HBM2e（8 模组） | 3.2 TB/s | 784 GB/s（灵衢前身 HCCS/UB） | ~550–600 W（第三方估计） | [Tom's Hardware/SemiAnalysis](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power) |
| **Ascend 910D** | 2025 送样测试 | 未公开 | 未公开 | 目标对标 H100（2000 BF16 TFLOPS） | 未公开 | 未公开 | 未公开 | 未公开 | [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-ascend-ai-910d-processor-designed-to-take-on-nvidias-blackwell-and-rubin-gpus) |
| **Ascend 950PR** | 2026Q1 | 未公开 | SIMD+SIMT | 1 PFLOPS FP8 / 2 PFLOPS FP4（官方路线图）；整卡 Atlas 350 实测口径 1561 TFLOPS mxFP4 / 804 TFLOPS mxFP8·HiF8·INT8 / 425 TFLOPS FP16·BF16 | 128 GB HiBL 1.0 自研 HBM | 1.6 TB/s（芯片）；1.4 TB/s（Atlas 350 卡） | 灵衢 2.0，2 TB/s 双向（芯片）；卡间 318–424 GB/s | ≤600 W（Atlas 350 卡） | [昇腾官网 950PR](https://www.hiascend.com/hardware/processor)、[Atlas 350 官网规格（已核实）](https://www.hiascend.com/hardware/accelerator-card)、[TH 路线图](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-details-ai-accelerator-roadmap-pulls-in-next-generation-ascend-npus-by-quarters-fp4-performance-of-the-ascend-960pr-doubles-expectations) |
| **Ascend 950DT** | 2026Q4 | 未公开 | SIMD+SIMT | 1 PFLOPS FP8 / 2 PFLOPS FP4（路线图）；Atlas 650E 单芯片口径 1.56 PFLOPS mxFP4 / 0.80 PFLOPS FP8·HiF8 / 0.425 PFLOPS FP16·BF16 | 144 GB HiZQ 2.0 自研 HBM（路线图）；Atlas 650E 为 96 GB/颗 | 4.0 TB/s | 灵衢 2.0 2 TB/s；整卡级 784 GB/s（650E 内） | 见整机 | [TH 路线图](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-details-ai-accelerator-roadmap-pulls-in-next-generation-ascend-npus-by-quarters-fp4-performance-of-the-ascend-960pr-doubles-expectations)、[Atlas 650E 官网（已核实）](https://www.hiascend.com/hardware/ai-server) |
| **Ascend 960DT** | 2027Q1（较原计划提前 3 个季度） | 未公开 | SIMD+SIMT | 2 PFLOPS FP8 / 4 PFLOPS FP4 | 288 GB | 9.6 TB/s | 2.2 TB/s | 未公开 | 同上 TH |
| **Ascend 960PR** | 2027Q3（提前 1 季度） | 未公开 | SIMD+SIMT | 2 PFLOPS FP8 / 8 PFLOPS FP4（FP4 较原规划翻倍） | 192 GB | 2.4 TB/s | 2.2 TB/s | 未公开 | 同上 TH |
| **Ascend 970** | 2028 | 未公开 | SIMD+SIMT | 3.6 PFLOPS FP8 / 14 PFLOPS FP4 | 288 GB | 14.4 TB/s | 4.4 TB/s | 未公开 | 同上 TH |
| **Ascend 980** | 2029（初步指标） | 未公开 | SIMD+SIMT | 7.2 PFLOPS FP8 / 28 PFLOPS FP4 | 384 GB | 38.4 TB/s | 8 TB/s | 未公开 | 同上 TH |

> 注：950 系列起华为引入自有低精度格式 **HiF8/HiF4** 与 mxFP8/mxFP4；950PR 面向 **Prefill 与推荐**（高带宽取向），950DT 面向**训练与全场景**。华为轮值董事长在 HC2026 宣布"一年一代"，并引用 τ Scaling Law 解释互联带宽与算力同步翻倍的路线（TH 报道同上）。

### 3.2 加速卡

| 卡型 | 芯片 | 算力 | 显存 | 功耗 | 形态/接口 | 定位 | 来源 |
|---|---|---|---|---|---|---|---|
| Atlas 300I 推理卡（3000） | 4×Ascend 310 | 64 TOPS INT8 / 32 TFLOPS FP16 | LPDDR4X | ~67 W | HHHL PCIe | 早期中心推理 | [华为企业文档入口](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909?submodel=doc) |
| Atlas 300V 视频解析卡 | 1×Ascend 310 | 16 TOPS INT8，100 路 1080P 解码 | 8 GB LPDDR4X | ~38 W | HHHL PCIe | 视频结构化 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-300v-pid-256209253) |
| Atlas 300V Pro | 1×Ascend 310P | 140 TOPS INT8 / 70 TFLOPS FP16，100 路 1080P | 24 GB LPDDR4X | 72 W | HHHL PCIe 4.0 | 视频+推理融合 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-300v-pro-pid-253542321) |
| Atlas 300I Pro | 1×Ascend 310P | 140 TOPS INT8 / 70 TFLOPS FP16 | 24 GB LPDDR4X，204.8 GB/s | 72 W | HHHL PCIe 4.0 | 中心/边缘推理主力卡 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-300i-pro-pid-251052354) |
| Atlas 300I Duo | 2×Ascend 310P | 280 TOPS INT8 / 140 TFLOPS FP16 | 48 GB LPDDR4X | 150 W | FHHL PCIe 4.0 | 高密推理 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-300i-duo-pid-252823107) |
| Atlas 300I A2 推理卡 | Ascend 310B 系 | 88–140 TOPS INT8（不同 SKU） | LPDDR4X | ≤150 W | PCIe | 新一代推理卡 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/a300i-a2-pid-260323393) |
| Atlas 300T 训练卡 | 1×Ascend 910 | 256 TFLOPS FP16 / 512 TOPS INT8 | 32 GB HBM2 | ~300 W | PCIe/自研形态 | 训练卡（910 时代） | [STH](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/)、华为企业文档 |
| Atlas 150 加速卡 | 950 系（入门级） | 未公开 | 未公开 | 未公开 | PCIe | 新一代入门推理卡 | [昇腾硬件中心](https://www.hiascend.com/hardware/accelerator-card) |
| **Atlas 350 加速卡** | 1×Ascend 950PR | **1561 TFLOPS mxFP4 / 804 TFLOPS mxFP8·HiF8·mxFP6·INT8 / 425 TFLOPS FP16·BF16** | **112 GB HBM，1.4 TB/s，ECC** | **≤600 W，被动散热** | 全高全长双宽 PCIe 5.0 x16（双向 128 GB/s）；卡间灵衢 4 卡全互连 318 GB/s / 2 卡 424 GB/s | LLM 推理/推荐/多模态生成 | [昇腾官网（已核实）](https://www.hiascend.com/hardware/accelerator-card) |

### 3.3 服务器 / 超节点 / 集群

| 产品 | 配置 | 算力 | 内存/带宽 | 功耗/散热 | 定位 | 来源 |
|---|---|---|---|---|---|---|
| Atlas 800 推理服务器（3000） | 最多 8×Atlas 300I/V Pro + 鲲鹏 920 | 最大 1.12 POPS INT8 | — | 风冷 4U | 中心推理/视频分析 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/a800-3000-pid-250743608) |
| Atlas 800 训练服务器（9000/9010） | 8×Ascend 910 + 鲲鹏 920 | 2 PFLOPS FP16 | 8×32 GB HBM2 | ~10 kW | 训练 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9000-pid-250702818) |
| Atlas 800I A2 / 800T A2 | 8×Ascend 910B 系 | 910B 系整机（官方未单列总算力） | 8×64 GB | 风冷/液冷 | 训练+推理一体机 | [800T A2](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-800t-a2-pid-254184887)、[800I A2](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-800i-a2-pid-261457531) |
| Atlas 800I A3 / 800T A3 超节点 | 910C 系超节点 | 未公开 | 未公开 | 风冷/液冷 | 中规模超节点 | [800T A3](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-800t-a3-pid-261716443)、[800I A3](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-800i-a3-pid-264117745) |
| Atlas 900 AI 集群（2019） | 1024×Ascend 910 | 256–512 PFLOPS FP16 | — | 液冷机柜 | 当时全球最快 AI 训练集群（ResNet-50 59.8s，官方宣称） | [华为 2019 发布](https://www.huawei.com/en/news/2019/9/huawei-atlas-900-ai-cluster) |
| Atlas 900 A2 PoD | 数百卡 910B 系，HCCS 全互联 | 未公开 | — | 液冷 | 训练集群基础单元 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-a2-pod-pid-254184911) |
| Atlas 900 A3 SuperPoD | 384×910C（= CloudMatrix 384 同源） | 300 PFLOPS BF16 dense | 49.2 TB HBM，1229 TB/s | ~559 kW，16 柜（12 算+4 网），6912×800G LPO 全光 | 大模型训练/推理超节点 | [TH/SemiAnalysis](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power)、[产品页](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-a3-superpod-pid-261207247) |
| Atlas 800 A3 风冷超节点 | 910C 系，风冷版 | 未公开 | 未公开 | 风冷 | 普通机房可部署超节点 | [昇腾硬件中心](https://www.hiascend.com/hardware/cluster) |
| **Atlas 850E 风冷超节点** | 950DT 系 | 未公开 | 未公开 | 风冷 | 950 代风冷超节点 | [昇腾硬件中心](https://www.hiascend.com/hardware/cluster) |
| **Atlas 950 SuperPoD 液冷超节点** | 最大 1024×昇腾950DT + 256×鲲鹏950；16 计算柜+4 灵衢互联柜（各 44OU） | **1 EFLOPS mxFP8/FP8/HiF8，2 EFLOPS mxFP4** | **1024×96 GB HBM（98 TB），4.0 TB/s/颗；256 TB 全局统一内存编址** | 液冷，单柜 100 kW；单柜互联 64×1.68 TB/s（双向），整机 1.72 PB/s | 万亿参数模型训练/中心推理 | [昇腾官网（已核实）](https://www.hiascend.com/hardware/cluster) |
| Atlas 950/960 SuperCluster（集群级） | 8192 卡（950）/15488 卡（960）规模 | 8 EFLOPS FP8 / 16 EFLOPS FP4（950 SuperPoD 最大配置，HC2025 口径：2048 鲲鹏 950 + 8192 950DT + 160 柜） | — | — | 国家级智算中心 | [TH 路线图报道](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-details-ai-accelerator-roadmap-pulls-in-next-generation-ascend-npus-by-quarters-fp4-performance-of-the-ascend-960pr-doubles-expectations)（注：2026-07 华为展示的实配为 256 CPU+1024 NPU） |
| **Atlas 650E 服务器** | 8×昇腾950DT + 2×鲲鹏950，14U | **12.48 PFLOPS mxFP4 / 6.43 PFLOPS FP8·HiF8 / 3.40 PFLOPS FP16·BF16** | **8×96 GB HBM（768 GB），4.0 TB/s；双机 16 NPU full-mesh 13.4 TB/s** | **14.5 kW，风冷**，每 NPU 400 Gbps UBoE/RoCE 直出 | 通用风冷机房的 950DT 推理/训推服务器 | [昇腾官网（已核实）](https://www.hiascend.com/hardware/ai-server) |

### 3.4 数据中心侧关键对比：CloudMatrix 384 vs NVIDIA GB200 NVL72

（SemiAnalysis 数据，经 [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power) 转述）

| 指标 | GB200 NVL72 | CloudMatrix 384 | 倍数 |
|---|---|---|---|
| BF16 dense 算力 | 180 PFLOPS | 300 PFLOPS | 1.7× |
| HBM 容量 | 13.8 TB | 49.2 TB | 3.6× |
| HBM 总带宽 | 576 TB/s | 1229 TB/s | 2.1× |
| Scale-Up 规模 | 72 GPU | 384 NPU | 5.3× |
| Scale-Up 带宽 | 518.4 Tb/s | 1075.2 Tb/s | 2.1× |
| Scale-Out 带宽 | 28.8 Tb/s | 153.6 Tb/s | 5.3× |
| 整机功耗 | 145 kW | 559 kW | 3.9× |
| 每 BF16 FLOP 功耗 | 0.81 W/TFLOP | 1.87 W/TFLOP | 2.3× 劣 |
| 每 TB/s 显存带宽功耗 | 251.7 W | 455.2 W | 1.8× 劣 |

**解读**：单芯片 910C（780 BF16 TFLOPS）仅为 B200（2500 TFLOPS）的 ~31%；DeepSeek 实测推理约为 H100 的 60%（[TH](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-research-suggests-huaweis-ascend-910c-delivers-60-percent-nvidia-h100-inference-performance)）。华为用 5.3× 芯片数 + 全光互联把系统级指标反超 1.7–3.6×，但付出 3.9× 功耗——在中国电价（部分地区 2025 年约 $56/MWh，TH 引述）下是可接受的成本结构。

### 3.5 价格（全部为第三方口径，华为官方不公开报价）

- Ascend 910B 加速卡：媒体/渠道口径约 ¥11–14 万元/卡（约 $15–20k），Reuters/TrendForce 均报道过区间；910C 模组级价格第三方估计 $20k+。**均为估计，供预算参考**。
- Atlas 300I Pro：渠道报价约 ¥3–5 万元/卡（估计）。
- Atlas 800 训练服务器（8×910）：第三方估计 ¥100 万+/台；Atlas 900/CloudMatrix/SuperPoD 为项目制报价，未公开。
- 参考入口：[TrendForce 新闻](https://www.trendforce.com/news/)、[SemiAnalysis](https://semianalysis.com/2025/04/16/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72/)、昇腾 AI 市场询价入口 [hiascend.com/marketplace](https://www.hiascend.com/marketplace/product)。

### 3.6 软件栈（对选型的实际影响）

- **CANN**（对标 CUDA）：当前 CANN 8.x 商用，2026 年 CANN 9.1.0 完成商用/社区版归一（[昇腾公告](https://www.hiascend.com/productbulletins/detail/809)）；HC2025 宣布 CANN 开源开放。
- **框架支持**：MindSpore（昇思）、torch_npu（PyTorch 原生插件）、vLLM-Ascend、MindIE（推理引擎）、MindSpeed（训练加速库）、Ascend C 算子开发、CATLASS 模板库（[昇腾文档中心](https://www.hiascend.com/document)）。
- **迁移成本**：PyTorch 模型经 torch_npu 可迁移，主流开源模型（DeepSeek、Qwen、GLM、LLaMA 系）官方均有适配；自定义 CUDA kernel 需重写为 Ascend C——这是与 NVIDIA 选型最大的隐性成本。

---

## 4. 消费侧产品线（麒麟 NPU）

| 芯片 | 年份 | 制程 | NPU 配置 | 公开算力 | 代表机型 | 来源 |
|---|---|---|---|---|---|---|
| 麒麟 970 | 2017 | TSMC 10nm | 单核 NPU（**寒武纪 Cambricon-1A IP**） | 官方口径约 1.92 TFLOPS FP16；图像识别约 2005 张/分钟 | Mate 10 / P20 | [PChome 报道](https://article.pchome.net/)、[华为 Mate 10 发布](https://consumer.huawei.com/cn/phones/mate10/) |
| 麒麟 980 | 2018 | TSMC 7nm | 双核 NPU（寒武纪 1H） | 官方未公布 TOPS；宣传"AI 算力翻倍" | Mate 20 / P30 | [华为官方新闻](https://consumer.huawei.com/cn/press/news/2018/hw-457614/) |
| 麒麟 810 | 2019 | TSMC 7nm | **自研达芬奇 NPU**（首次弃用寒武纪） | ETH AI Benchmark 当时第一（具体分数未官方沉淀） | Nova 5 / 荣耀 9X | [凤凰网报道](https://tech.ifeng.com/) |
| 麒麟 990 / 990 5G | 2019 | TSMC 7nm+ EUV | 达芬奇 NPU **双大核+微核** | 未公开 TOPS；微核专攻超低功耗场景 | Mate 30 / P40 | [知乎/官方发布报道](https://zhuanlan.zhihu.com/p/81974395) |
| 麒麟 9000 / 9000E | 2020 | TSMC 5nm | 达芬奇 2.0 NPU 双大核+微核 | 未公开 | Mate 40（最后一代 TSMC 麒麟旗舰） | [站长之家等](https://www.chinaz.com/) |
| 麒麟 9000S | 2023 | **SMIC N+2（7nm 级）** | 达芬奇 NPU（规格未公开） | 未公开 | Mate 60 系列（回归之作） | TechInsights 拆解（techinsights.com） |
| 麒麟 9010 | 2024 | SMIC N+2 改进 | 同上 | 未公开 | Pura 70 系列 | 同上 |
| 麒麟 9020 | 2024 | SMIC | 同上 | 未公开 | Mate 70 系列 | 同上 |
| 麒麟 9030 | 2025 | SMIC | 同上 | 未公开 | Mate 80 系列 | 同上 |
| **麒麟 X90** | 2025 | SMIC | NPU+达芬奇架构 | 未公开 | **MateBook Pro / MateBook Fold 鸿蒙 PC**（华为 PC 处理器首作） | [华为 MateBook 产品页](https://consumer.huawei.com/cn/laptops/)、TH/媒体拆解报道 |

**消费侧要点**：
1. 2017–2019 从"购买寒武纪 IP"切换到"自研达芬奇"，与云侧同构——这是华为全栈战略的真正起点。
2. 麒麟 NPU 的 TOPS 从未官方披露，端侧选型不看峰值算力而看**端侧大模型部署能力**：HarmonyOS NEXT/5/6 的小艺智能体、盘古端侧模型（约 3B 级量化）依赖 NPU 常驻推理，功耗预算通常 <2 W。
3. 2025 年麒麟 X90 把 NPU 路线延伸到 PC：鸿蒙 PC 的 AI 能力（文档摘要、图像处理、小艺）由片上 NPU 承担，定位对标苹果 Neural Engine + 高通 X Elite NPU。
4. 海思另有安防/IPC SoC 线（Hi3516/Hi3519/Hi3559 等，内置 NNIE AI 引擎，约 1–4 TOPS 级），广泛用于摄像头端侧推理——资料分散于海思官网与渠道文档，官方未集中公开规格。

---

## 5. 车端产品线（MDC 智能驾驶计算平台）

| 平台 | 年份 | 芯片 | 算力 | 功耗 | 定位/搭载 | 来源 |
|---|---|---|---|---|---|---|
| MDC 300F | 2019 | 昇腾 310（Atlas 系加速） | **64 TOPS INT8 稠密** | ~60 W 级（域控整机） | 商用车/作业车、L3 试点 | [媒体规格汇总](https://www.toutiao.com/article/7679656110160167460/)、[华为 2019 HC 发布](https://www.huawei.com/cn/news/2019/9/huawei-connect-2019) |
| MDC 210 | 2020-09-25 发布 | 昇腾系 | 48 TOPS（媒体口径） | ~30 W | L2+ 乘用车 | [媒体报道](https://www.toutiao.com/article/7657390733850182187/) |
| MDC 610 | 2020 发布/2022 上车 | **昇腾 610 车规芯片** | **200 TOPS** | ~80–120 W（域控） | ADS 1.0/2.0 主力：问界 M5/M7、阿维塔 11 部分配置 | 同上；[阿维塔 11 配置报道](https://www.toutiao.com/article/7652676347978695183/) |
| MDC 810 | 2021-04 量产 | 昇腾 610（多芯组合） | **400+ TOPS** | ~200 W 级 | ADS 高配：极狐阿尔法 S HI、阿维塔 11/12（3 激光雷达版） | [量产报道](https://www.toutiao.com/article/6952520032152814084/)、阿维塔 11 400TOPS MDC810 配置 |
| MDC 810 Pro / 新一代域控 | 2024-25 | 未公开（推测昇腾新代） | 未公开（ADS 4 乾崑方案） | 未公开 | 乾崑 ADS 4：问界 M8/M9 新款、享界、尊界等 | [华为智能汽车解决方案官网](https://www.huawei.com/cn/intelligent-automotive) |

**要点**：
- MDC 本质是"车规化的昇腾推理芯片 + 鸿蒙车机 OS + ADS 算法栈"，与数据中心同源架构，便于模型云-车协同训练/下发。
- 竞品对标：MDC 610（200 TOPS）对英伟达 Orin-X（254 TOPS）、地平线征程 5（128 TOPS）；MDC 810（400 TOPS）对双 Orin 方案。2024 后行业转向舱驾一体/更高算力（Orin Thor、征程 6），华为新一代域控具体芯片未公开。
- 选型注意：车端不单独卖芯片，MDC 随 ADS 整体方案授权；消费者业务（智选车/HI 模式）由车企集成。

---

## 6. 边缘与机器人产品线

### 6.1 边缘推理

| 产品 | 芯片 | 算力 | 内存 | 功耗 | 形态 | 场景 | 来源 |
|---|---|---|---|---|---|---|---|
| Atlas 200 AI 加速模块 | Ascend 310 | 16 TOPS INT8 / 8 TFLOPS FP16 | 8 GB LPDDR4X | ~9.5 W | 半信用卡模组 | 边端设备嵌入 | [昇腾社区/企业文档](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909?submodel=doc) |
| **Atlas 200I A2 加速模块** | Ascend 310B | **20 TOPS INT8 / 10 TFLOPS FP16**（另有 8 TOPS SKU） | LPDDR4X 12/8/4 GB ECC | **21–25 W** | MXM 82×60×7mm，-20~+103℃ | 机器人、无人机、边端智能设备（官网原文："集成于边端智能设备、机器人、无人机中"）；40 路 1080P 视频分析 | [昇腾官网（已核实）](https://www.hiascend.com/hardware/accelerator-module-A2) |
| Atlas 200I DK A2 开发者套件 | 同上 | 同上 | 8 GB | ~25 W | 开发板 | 开发者/高校教学 | [文档](https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/index/index.html) |
| Atlas 500 智能小站 | 1×Ascend 310 | 16/22 TOPS INT8 | — | ~25–40 W | 室外防尘防水小站 | 智慧交通路口、园区 | [华为企业文档](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909?submodel=doc) |
| Atlas 500 A2 智能小站 | Ascend 310B | 20 TOPS 级 | — | 低功耗 | 室外小站 | 同上 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-500-a2-pid-254412305) |
| Atlas 500 Pro 智能边缘服务器 | 最多 4×Ascend 310P（300I Pro） | 最大 560 TOPS INT8 | — | ~600 W 短机箱 | 2U 短深边缘服务器 | 区县机房、油站/电站 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/a500-pro-3000-pid-251168992) |
| Atlas 800 推理服务器（3000） | 8×300I/V Pro | 1.12 POPS INT8 | — | ~2 kW | 4U | 中心推理/视频汇聚 | [产品页](https://support.huawei.com/enterprise/zh/ascend-computing/a800-3000-pid-250743608) |

### 6.2 机器人侧

- 华为未单独发售"机器人芯片"，路线为 **"昇腾模组（Atlas 200I A2）本体算力 + 鸿蒙 OS + 盘古具身智能大模型（云）"** 三层结构。官方已将 Atlas 200I A2 标注为机器人/无人机用算力（见上）。
- 合作落地：乐聚机器人"夸父 Kuavo"人形机器人搭载鸿蒙系统并接入盘古大模型，HDC 2024 亮相（[The Robot Report](https://www.therobotreport.com/)、华为 HDC 报道）；华为云具身智能平台（CloudRobo 类服务）面向机器人训练/仿真。
- 机器人选型实质是"边缘推理选型"：本体实时控制用 Atlas 200I A2/200 模组（20 TOPS 足够跑视觉+规划量化模型），大模型推理上云（华为云昇腾服务）。

---

## 7. 影响业务选型的关键超参对比框架

| 维度 | 权重说明 | 数据中心训练 | 数据中心推理 | 边缘推理 | 车端 | 端侧 |
|---|---|---|---|---|---|---|
| 精度算力 | FP8/FP4 决定性价比；BF16/FP16 决定兼容性 | ★★★（FP8→FP4 演进中） | ★★★（mxFP4 是 950 杀手锏） | ★★ INT8 | ★★ INT8 | ★ INT8/FP16 |
| 显存容量 | 决定能装多大模型/多大 KV-Cache | ★★★（训练需模型+优化器） | ★★★（并发=显存） | ★★ | ★ | ★ |
| 显存带宽 | 决定 decode 吞吐（Token/s） | ★★ | ★★★（950DT 4.0 TB/s 是卖点） | ★★ | ★ | ★★ |
| 互联带宽 | 决定多卡扩展效率（TP/EP/PP） | ★★★（灵衢 2 TB/s vs NVLink） | ★★★（EP 并行依赖） | ★ | ★ | — |
| 功耗 | 决定 TCO 与部署形态 | ★★（中国电价下权重较低） | ★★ | ★★★（无风扇场景） | ★★★（车规功耗墙） | ★★★（电池） |
| 软件生态 | CANN vs CUDA 迁移成本 | ★★★ | ★★★ | ★★ | ★★（车规认证链） | ★★（HiAI/Foundation） |
| 供货/合规 | 制裁与产能 | ★★★ | ★★★ | ★★ | ★★ | ★ |

**换算公式（估算用）**：
- 推理并发 ≈ 显存可用容量 ÷（模型权重量化后大小 + 单并发 KV-Cache）
- Decode 单并发吞吐 ≈ 显存带宽 ÷ 模型权重字节数（下界估计，实际含算力/调度损耗）
- 训练卡数 ≈ 总参数量×（2 bytes 权重 + 优化器 ~8–16 bytes）÷ 单卡显存 → 决定 TP/PP 切分下限；再用互联带宽评估并行效率。

---

## 8. 典型应用场景配置方案

### 场景 A：互联网大模型在线推理（如 DeepSeek-R1/Qwen3-235B 级 MoE）

- **推荐**：Atlas 950 SuperPoD（1024×950DT 液冷）或 Atlas 650E（8×950DT 风冷起步）。
- **理由**：MoE 大 EP 并行需要全互联大带宽——950 SuperPoD 单柜 64×1.68 TB/s 双向、全局 256 TB 统一内存编址，EP 路由可全局放置专家；mxFP4 使 235B 模型权重 <150 GB，单节点即可多副本。650E 双机 16 NPU full-mesh（13.4 TB/s）可覆盖中型 EP 域。
- **效果**：官方定位"万亿模型训练和中心推理最佳选择"；对比 910C 时代需 CloudMatrix 384 级规模，950 代单节点即可完成同等 EP 推理，功耗与占地显著下降。
- **预算级替代**：Atlas 800I A3（910C 系）风冷超节点——存量方案，生态成熟。

### 场景 B：企业私有化大模型微调 + 推理（70B 级）

- **推荐**：Atlas 800T A2（8×910B）训推一体；或 Atlas 650E 一步到位。
- **理由**：70B BF16 权重 ~140 GB，910B 单卡 64 GB→8 卡 TP8 可训可推；昇腾原生 MindSpeed/LLaMA-Factory 适配成熟。
- **效果**：单台覆盖"微调+在线推理"全生命周期，采购/运维成本低于超节点。

### 场景 C：智慧城市视频分析（千路级摄像头）

- **推荐**：中心侧 Atlas 800 推理服务器（8×300I Pro = 1.12 POPS INT8，800 路 1080P 分析）；路侧 Atlas 500/500 A2 小站。
- **理由**：300I Pro/300V Pro 的硬件编解码单元是关键——100 路 1080P 解码/卡，CPU 不参与视频管线；边缘小站做"先过滤、后上传"降低回传带宽 90%+。
- **效果**：经典云边协同架构，也是昇腾出货量最大的场景之一。

### 场景 D：高阶智能驾驶量产车

- **推荐**：MDC 610（200 TOPS，ADS 中配）或 MDC 810（400 TOPS，3 激光雷达高配）。
- **理由**：城区 NCA 端到端模型需要 ≥200 TOPS INT8 稠密算力支撑 BEV+Occupancy 网络实时推理；MDC 610 功耗 ~100 W 内可由域控风冷/液冷承载。
- **效果**：ADS 2.0/3.0 已在问界、阿维塔、深蓝等规模量产验证"车位到车位"全场景。

### 场景 E：人形机器人/具身智能

- **推荐**：本体 Atlas 200I A2（20 TOPS，25 W，-20~103℃ 宽温）+ 云端盘古具身大模型（华为云昇腾）。
- **理由**：本体算力承担实时感知-控制闭环（<10 ms 延迟要求），大脑推理走云端；鸿蒙提供分布式软总线。
- **效果**：乐聚夸父等样机验证该架构可行；代价是本体算力不足以跑 >3B 模型本地推理，需要云边分级。

### 场景 F：手机/PC 端侧 AI

- **推荐**：麒麟 90x0（手机）/ 麒麟 X90（PC），NPU 常驻跑量化小模型（≤3B）。
- **理由**：功耗预算 <2 W，隐私数据不出端；达芬奇 NPU + HiAI Foundation 支持 INT8/INT4 量化部署。
- **效果**：小艺智能体、AI 修图、通话翻译等鸿蒙原生 AI 能力的硬件底座。

---

## 9. 华为 NPU 架构演进趋势分析

1. **精度下探路线明确**：FP16/BF16（910）→ FP8（910C 实测支持、950 官方 mxFP8/HiF8）→ mxFP4/HiF4（950→970，FP4 算力为 FP8 的 4 倍）——与 NVIDIA FP4 路线同向，但华为把 FP4 做成"主打精度"以弥补单芯片绝对算力差距。
2. **从"芯片战"转向"总线战/系统战"**：HCCS（720 Gbps，2019）→ UB 灵衢（784 GB/s 芯片级/单柜 1.68 TB/s，910C/950）→ UB 2.0（2 TB/s，950）→ UB-Mesh 论文级架构瞄准百万芯片超节点（[STH Hot Chips 2025](https://www.servethehome.com/huawei-presents-ub-mesh-interconnect-for-large-ai-supernodes-at-hot-chips-2025/)）。超节点 = 华为的"NVLink 域"对标物，且规模（1024 卡）远大于 NVL72。
3. **HBM 自主化**：910C 依赖外购 HBM2e（经 CoAsia 等代理自三星渠道，TH/SemiAnalysis 报道）→ 950PR 搭载自研 HiBL 1.0（128 GB/1.6 TB/s）→ 950DT HiZQ 2.0（144 GB/4 TB/s）→ 970 达 14.4 TB/s。HBM 是当前最大瓶颈（SemiAnalysis 2025-09 报告标题即"HBM is The Bottleneck"），也是路线图最激进的部分。
4. **制程迂回**：TSMC N7+（910）→ 断供 → SMIC N+1/N+2（910B/910C、麒麟 9000S 系）→ 叠加先进封装（双 die chiplet）。910C 的"双 910B die 封装"即典型案例——用封装补制程。
5. **SIMD→SIMT 编程模型扩展**：950 代引入 SIMT（类 GPU 线程模型），降低 CUDA 开发者的迁移门槛——软件生态补课的信号。
6. **端侧**：NPU 从"拍照加速协处理器"演进为"常驻智能体引擎"；麒麟回归后 NPU 规格不再公开，但鸿蒙端侧盘古模型规模（3B→更大）会拉动 NPU 算力升级。
7. **节奏**：HC2026 明确"一年一代"（950→960→970→980），回应 NVIDIA 的年度节奏；能否兑现取决于 SMIC 产能与 HBM 自产良率。

**风险提示**：910B/910C 官方规格未公开（第三方拆解口径）；SMIC 7nm 产能与良率（第三方估计月产能数万片级）限制放量；CANN 生态成熟度仍落后 CUDA 约 2–3 年（主流框架可用，长尾算子需自研）；车端新一代域控规格未公开。

---

## 10. 参考来源汇总（可点击查阅）

**官方（昇腾/华为）**
- 昇腾硬件中心（当前在售全部型号与规格）：https://www.hiascend.com/hardware
- 昇腾 NPU 处理器页（950PR/950DT）：https://www.hiascend.com/hardware/processor
- Atlas 350 加速卡规格：https://www.hiascend.com/hardware/accelerator-card
- Atlas 650E 服务器规格：https://www.hiascend.com/hardware/ai-server
- Atlas 950 SuperPoD 规格：https://www.hiascend.com/hardware/cluster
- Atlas 200I A2 加速模块：https://www.hiascend.com/hardware/accelerator-module-A2
- 昇腾文档中心（含各产品文档入口）：https://www.hiascend.com/document
- 华为企业产品库（Atlas 全系文档）：https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909?submodel=doc
- 华为智能汽车解决方案：https://www.huawei.com/cn/intelligent-automotive
- 华为消费者产品：https://consumer.huawei.com/cn/

**第三方分析/媒体**
- Tom's Hardware 昇腾路线图（2026-09，含 910C–980 全表）：https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-details-ai-accelerator-roadmap-pulls-in-next-generation-ascend-npus-by-quarters-fp4-performance-of-the-ascend-960pr-doubles-expectations
- CloudMatrix 384 vs GB200 NVL72（SemiAnalysis 数据）：https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power
- SemiAnalysis 原文：https://semianalysis.com/2025/04/16/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72/
- SemiAnalysis《Huawei Ascend Production Ramp: Die Banks, TSMC Continued Production, HBM is The Bottleneck》：https://semianalysis.com/2025/09/08/huawei-ascend-production-ramp/
- 910C vs H100（DeepSeek 研究）：https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-research-suggests-huaweis-ascend-910c-delivers-60-percent-nvidia-h100-inference-performance
- 910B SMIC vs 910 TSMC 拆解：https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910
- 910D 报道：https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-ascend-ai-910d-processor-designed-to-take-on-nvidias-blackwell-and-rubin-gpus
- Ascend 910（Hot Chips 31）：https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/
- UB-Mesh（Hot Chips 2025）：https://www.servethehome.com/huawei-presents-ub-mesh-interconnect-for-large-ai-supernodes-at-hot-chips-2025/
- MDC 系列规格汇总（媒体）：https://www.toutiao.com/article/7679656110160167460/
- 麒麟 970 NPU（寒武纪 1A）报道：https://article.pchome.net/（2017-09 报道）

**说明**：本报告于 2026-09-20 编制；标注"第三方/媒体口径"的参数建议采购前以华为官方报价与最新产品手册复核。
