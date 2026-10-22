#!/usr/bin/env python3
"""从四份推理优化报告提取全部技术点，生成分类学习表 Excel。"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = Workbook()

# ── 样式 ──
F_TITLE = Font(name='Arial', size=14, bold=True, color='FFFFFF')
F_HDR   = Font(name='Arial', size=10.5, bold=True, color='FFFFFF')
F_BODY  = Font(name='Arial', size=10)
F_CAT   = Font(name='Arial', size=11, bold=True)
FILL_TITLE = PatternFill('solid', start_color='1F3864')
FILL_HDR   = PatternFill('solid', start_color='4472C4')
FILL_CAT   = {
    'speculative': PatternFill('solid', start_color='E2EFDA'),
    'quantization': PatternFill('solid', start_color='FCE4D6'),
    'kv_cache':    PatternFill('solid', start_color='DDEBF7'),
    'kernel':      PatternFill('solid', start_color='FFF2CC'),
    'multimodal':  PatternFill('solid', start_color='E4DFEC'),
    'scheduling':  PatternFill('solid', start_color='D9E1F2'),
    'model':       PatternFill('solid', start_color='F2F2F2'),
    'hardware':    PatternFill('solid', start_color='D6DCE4'),
}
THIN = Border(*[Side(style='thin', color='B0B0B0')]*4)
WRAP = Alignment(wrap_text=True, vertical='top')

def style_sheet(ws, title, headers, widths):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    c = ws.cell(1, 1, title); c.font = F_TITLE; c.fill = FILL_TITLE
    c.alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 30
    for j, h in enumerate(headers, 1):
        c = ws.cell(2, j, h); c.font = F_HDR; c.fill = FILL_HDR
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        c.border = THIN
        ws.column_dimensions[get_column_letter(j)].width = widths[j-1]
    ws.row_dimensions[2].height = 32
    ws.freeze_panes = 'A3'

def add_rows(ws, rows, start=3):
    for i, r in enumerate(rows):
        for j, v in enumerate(r, 1):
            c = ws.cell(start+i, j, v)
            c.font = F_BODY; c.border = THIN; c.alignment = WRAP

# ══════════════════════════ Sheet 1: 全量技术点总表 ══════════════════════════
ws = wb.active; ws.title = '技术点总表'
headers = ['大类','技术点（含 llama.cpp 参数）','原理一句话','实测收益','代价/风险','是否采用','报告出处']
widths  = [12, 34, 46, 30, 32, 12, 20]
style_sheet(ws, 'Qwen3.8-27B 推理优化·关键技术点学习总表（全量 30 项）', headers, widths)

rows = [
 # ── speculative（投机解码）──
 ('投机解码','MTP 草稿：-md + --spec-type draft-mtp',
  '用第 64 层的 MTP 预测块做 Draft：一次权重读产出 mean_len≈3 个 token，分母从"每 token 读一遍权重"变为"每 3 token 读一遍"',
  'Decode 49.2→87.3 t/s（+77%）','Draft 权重 +1.28 GiB；接受率随任务波动','✅ 采用','decode-opt §4'),
 ('投机解码','草稿深度 n-max：--spec-draft-n-max N',
  '步耗时随 N 线性涨（每草稿+3.4ms），mean_len 在 N=4 后饱和 → 乘积在 4 处取峰值',
  'N=4 时 94.4 t/s（比 N=2 +8%）；N=8 反降到 69.8','N 越大显存越多（验证 batch 变宽）','✅ 采用 N=4','uplimit §3.11'),
 ('投机解码','草稿深度随位宽移动：（随位宽变化）',
  '位宽下降后步耗时对草稿数不敏感（KV 读占比上升），最优 N 会右移',
  'Q4 上 N=4 最优；Q2 上 N=5 仍略优（161 vs 157）','换位宽必须重扫草稿深度','◐ 观察','newarch §3.11.4'),
 ('投机解码','草稿 KV 量化：--spec-draft-type-k/v q4_0',
  'Draft 只负责提案、由主模型验证 → 草稿精度损失不进输出，"小"比"准"重要',
  '省 ~400 MiB（噪声内），接受率不变','实测在 ±500 MiB 峰值噪声内','✅ 采用','multi-mode §4'),
 ('投机解码','n-gram 草稿：--spec-type ngram-*',
  '从 prompt 里检索 n-gram 匹配做零显存 Draft',
  '文献 1.15×（零显存代价）','自由生成接受率可能 <10% 反而拖慢','❌ 未采用','decode-opt §4.3'),
 ('投机解码','提高草稿精度（Q8 Draft）：Q8_0 Draft（3.16 GB）',
  '接受率略升（0.52→0.55），但每步多读 1.8 GB → 净效果为负',
  '实测 vis Decode -17%（123→103）','"更大"的代价 > "更准"的收益','❌ 证伪','newarch §5.7.2'),

 # ── quantization（权重量化）──
 ('权重量化','UD 动态量化：（模型文件本身）',
  'imatrix 驱动的混合精度：FFN 降到 IQ4_XS/Q4_K/Q3_K，注意力升 Q5_K/Q6_K',
  '有效权重 14.997 GiB；PPL 相对 Q6_K 仅 +0.27%','全局质量损失由 PPL 守门','✅ 基准档','decode-opt §4'),
 ('权重量化','主权重 Q4→Q3：（更低位宽）',
  '体积 -18% → 步耗时几乎不变（block 效率相近）',
  'vis 115.8（比 Q4 慢）；PPL +4.6%','质量超红线且无速度收益','❌ 证伪','newarch §3.11.3'),
 ('权重量化','主权重 Q4→Q2：（2-bit）',
  '体积 -34% → 步耗时 25.3ms（达到 150 t/s 的必要条件）',
  'vis 157.2 t/s（唯一过 150 的档）','PPL +23.1%（超红线 11.6 倍）','❌ 不可交付','newarch §3.11.3'),
 ('权重量化','混合位宽 MIX：--tensor-type（自定义）',
  '按张量类型指定精度：注意力/SSM/输出头升，FFN 降',
  'vis 133.5 t/s，PPL +12.2%','仍超红线；需自建量化','❌ 实验用','newarch §3.11.3'),

 # ── kv_cache（KV Cache）──
 ('KV Cache','混合量化 q8/q4：--cache-type-k q8_0 -ctv q4_0',
  'K 保 8.5bit（检索精度）+ V 降到 4.5bit（对精度更鲁棒）',
  '省显存（F16 KV 装不下 64K 双槽）','本构建无 FA 向量 kernel → 转换开销','◐ ①轮用','decode-opt §4'),
 ('KV Cache','全 q4_0：--cache-type-k/v q4_0',
  'q4_0-q4_0 在 stock 构建有原生 FA 向量 kernel；q8q4 反而没有',
  '比 q8q4 更快（+18%）且省 512 MiB（A/B 实测=几何预测）','K 精度下降需 24K 检索守门（3/3 通过）','✅ 最终采用','multi-mode §5.7'),
 ('KV Cache','全 q8_0：--cache-type-k/v q8_0',
  '两侧都保 8.5bit；有 FA 向量 kernel',
  '速度与 q8q4 相当，显存多 0.5 GiB','占用冗余','◐ 备选','decode-opt §4'),
 ('KV Cache','量化补 FA kernel：GGML_CUDA_FA_QUANTS 编译',
  '把缺失的量化组合补进 FA 向量 kernel 的编译期开关',
  '只回收 4.2%（与噪声同量级）；改 q4q4 后该问题不存在','需维护自定义构建','❌ 证伪','multi-mode §5.7'),

 # ── kernel（算子/编译）──
 ('Kernel','Flash Attention：--flash-attn on',
  '在线注意力：O(N) 显存而非 O(N²)；新版必须写 on 而非裸 flag',
  'Prefill +10.8%，Decode 持平','低风险','✅ 采用','decode-opt §4'),
 ('Kernel','CUDA Graph：（默认启用）',
  '捕获整层 kernel 序列消 launch 开销；日志看 graphs reused',
  '已含在基线里，无额外空间','对 slot 数与 batch 形状敏感','◐ 已含','decode-opt §4.3'),
 ('Kernel','QKV 融合：（改 ggml 图）',
  '合并 Q/K/V 投影为一次 GEMM',
  '理论上限仅 1.02×','改动大、收益最小','❌ 未采用','decode-opt §4.3'),

 # ── multimodal（多模态专属）──
 ('多模态','mmproj 量化：--mmproj mmproj-Q8_0.gguf',
  'ViT 投影从 F16 降到 Q8：视觉路径不走文本头，PPL 不受影响',
  '省 288 MiB；8 项视觉题与 F16 零差异','低风险','✅ 采用','multi-mode §4'),
 ('多模态','图像 token 上限：--image-max-tokens N',
  '图像 token = (边长/32)² 平方增长；cap 同时是性能与质量杠杆',
  '2048px TTFT 3.9s→1.0s（-74%）；cap 768 让视觉题 7/7（1024 只有 6/7）','cap 太低会丢细节（512 网格题出错）','✅ 采用 768','multi-mode §5.7'),
 ('多模态','视觉 batch 上限：--mtmd-batch-max-tokens 256',
  '限制 ViT 编码的每批 token 数，压视觉阶段峰值',
  '省 ~48 MiB；ViT 阶段 <5% 影响','低风险','✅ 采用','multi-mode §4'),
 ('多模态','视觉塔 CPU 卸载：--no-mmproj-offload',
  'ViT 跑 CPU 省显存，代价是编码慢 ~50×',
  '省 ~600 MiB；但 TTFT 0.9s→17.9s','不可接受的延迟','❌ 证伪','multi-mode N1'),

 # ── scheduling（调度/并发）──
 ('调度','双槽并发：--parallel 2',
  '两个 slot 交替推进，掩盖单槽验证停顿',
  '系统吞吐 23→113 t/s（纯文本）；每槽上下文减半','p2+nmax4 峰值越线；视觉并发抢算力','◐ 视场景','decode-opt §4'),
 ('调度','单槽长上下文：--parallel 1',
  '单请求可用满 64K（p2 时每槽只有 32K）',
  '单请求 ctx ×2；峰值降 684 MiB','并发系统吞吐 -10.7%（受控 A/B）','✅ 最终采用','uplimit §5.7'),
 ('调度','batch/ubatch：-b 2048 -ub 512',
  '逻辑/物理 batch；ub 决定 prefill 工作区',
  'b=2048 让 28K Decode +12%；ub=1024 多 300 MiB 无稳定收益','ub 太大峰值越线','✅ 采用','uplimit §4'),

 # ── model（模型层）──
 ('模型层','分辨率→token 杠杆：（图像前处理）',
  '图像 token 数既吃 prefill 也吃 KV，是多模态独有的自由度',
  '视觉 Decode 121→157（cap 768→512，+30%）','2048px 图反降；难题掉一项','◐ 场景档','newarch §5.7.1'),

 # ── hardware（硬件层）──
 ('硬件层','全层 GPU 卸载：-ngl 99 -ngld 99',
  '任何一层留 CPU 都会引入 PCIe 往返（单层 17ms = 预算的 86%）',
  '达成带宽 81.4% 反证无回退','PCIe 带宽 26 GB/s 是硬限制','✅ 硬约束','decode-opt §3.8'),
]
add_rows(ws, rows)

# 分类底色
CAT_ORDER = ['speculative','quantization','kv_cache','kernel','multimodal','scheduling','model','hardware']
for i, r in enumerate(rows):
    cat = r[0]
    fill = FILL_CAT.get({'投机解码':'speculative','权重量化':'quantization','KV Cache':'kv_cache',
                          'Kernel':'kernel','多模态':'multimodal','调度':'scheduling',
                          '模型层':'model','硬件层':'hardware'}.get(cat,''), PatternFill())
    for j in range(1, len(headers)+1):
        ws.cell(3+i, j).fill = fill

# ══════════════════════════ Sheet 2: 按大类汇总 ══════════════════════════
ws2 = wb.create_sheet('按大类汇总')
headers2 = ['大类','技术点数','已采用','证伪/未采用','核心收益','关键教训']
widths2  = [14, 10, 10, 14, 48, 60]
style_sheet(ws2, '八大类技术点汇总（ adopted vs 证伪 ）', headers2, widths2)
rows2 = [
 ('投机解码', 6, 3, 3, 'Decode +77%（主引擎）；n-max=4 是拐点', 'Draft"小"比"准"重要：Q8 Draft -17%；最优 n-max 随位宽移动'),
 ('权重量化', 4, 1, 3, 'Q4_K_M 是唯一过 PPL 红线的档', '质量代价超线性：Q3 +4.6%、Q2 +23%——交点就在 Q4'),
 ('KV Cache', 4, 1, 3, 'q4q4 比 q8q4 更快且省 512 MiB', '有无 FA 向量 kernel 决定速度，不只看位宽'),
 ('Kernel', 3, 1, 2, 'FA 对 Prefill +10.8%', '带宽已 92.9% 后 kernel 层只剩个位数空间'),
 ('多模态', 4, 3, 1, 'TTFT -74%；难题 6/7→7/7', '图像 token 是平方关系，不设 cap 等于交出控制权'),
 ('调度', 3, 2, 1, '系统吞吐 23→113 或 单请求 64K', '并发与单请求体验在单卡上零和'),
 ('模型层', 1, 0, 1, '分辨率是唯一不碰权重的杠杆', '同一杠杆在不同位宽上方向可能相反'),
 ('硬件层', 1, 1, 0, '零 CPU 回退（硬约束）', '单层 PCIe 往返 = 86% 预算'),
]
add_rows(ws2, rows2)
for i in range(len(rows2)):
    for j in range(1, 7):
        ws2.cell(3+i, j).fill = FILL_CAT.get(CAT_ORDER[i], PatternFill())

# ══════════════════════════ Sheet 3: 步耗时模型 ══════════════════════════
ws3 = wb.create_sheet('步耗时模型')
headers3 = ['组成','Q4_K_M (ms)','占比','与位宽有关？','怎么压','压到极限后']
widths3  = [16, 16, 10, 14, 40, 36]
style_sheet(ws3, 'Decode 单步耗时三项分解（理解"为什么不能再快"的钥匙）', headers3, widths3)
rows3 = [
 ('权重读取', 18.2, '56%', '是（正比体积）', '降位宽：Q4→Q2 可到 12.3ms（但 PPL 爆炸）','3.2 BPW 以下'),
 ('验证成本', 11.9, '37%', '否', '换 Draft 结构（EAGLE 类一次验证多 token）','结构性开销'),
 ('固定开销', 2.3, '7%', '否', 'kernel launch / 采样 / 调度——已接近下限','不可压'),
 ('合计', 32.4, '100%', '—', '红线内极限 = 3.06÷32.42ms = 94.4 t/s（重复型 121）','123 t/s'),
]
add_rows(ws3, rows3)
for j in range(1, 7):
    ws3.cell(6, j).font = Font(name='Arial', size=10, bold=True)
    ws3.cell(6, j).fill = PatternFill('solid', start_color='E2EFDA')

# ══════════════════════════ Sheet 4: 四轮演化 ══════════════════════════
ws4 = wb.create_sheet('四轮演化与指标')
headers4 = ['指标','基线','①纯文本','②多模态','③极限交付','变化','口径说明']
widths4  = [22, 12, 12, 12, 14, 12, 50]
style_sheet(ws4, '四轮优化·关键指标演化（同会话 A/B 口径）', headers4, widths4)
rows4 = [
 ('Decode 纯文本 (t/s)', 49.2, 87.3, 92.65, 94.4, '+92%', '首 token 到末 token 斜率'),
 ('Decode 视觉 (t/s)', '—', '—', 96.1, 121.1, '+154%', '重复型提问口径（自由描述约 100）'),
 ('32K Decode (t/s)', 44.1, 65.2, 67.1, 78.1, '+77%', '长上下文 CV 15.8%，读作趋势'),
 ('视觉 TTFT 2048px', '—', '—', '—', '1.0s', '-74%', '基线 3.9s（不限 cap）'),
 ('峰值显存 (MiB)', 19563, 19871, 20452, 19857, '≈持', 'nvidia-smi 生命周期最大'),
 ('安全冗余 (GiB)', 4.88, 4.58, 4.02, 4.60, '✅', '全程 ≥4.0 红线'),
 ('单请求上下文', 65536, 32768, 32768, 65536, '恢复', 'p2 时每槽减半'),
 ('视觉难题', '—', '—', '6/7', '7/7', '↑', 'cap 768 修复注意力偏移'),
 ('PPL', 5.8216, 5.8216, 5.8229, 5.8229, '+0.02%', '红线 ≤+2%'),
]
add_rows(ws4, rows4)

wb.save('output/推理优化技术点学习表.xlsx')
print('saved')
