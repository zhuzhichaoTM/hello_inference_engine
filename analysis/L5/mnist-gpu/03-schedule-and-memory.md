# 03 · 调度与显存：split 决策、offset 规划、原地复用

> 对应源码：`src/ggml-backend.cpp:1066`（`split_graph`）、`:1591`（`alloc_splits`）、`:1646`（`compute_splits`）、
> `src/ggml-alloc.c:718`（`alloc_graph_impl`）、`:202`（`dyn_tallocr_alloc`）、`:623`（`allocate_node`）、`:691`（`free_node`）
>
> 本篇回答三个问题：**每个节点归谁执行？每个张量放在哪个 offset？什么时候可以复用？**

---

## 0 业务侧：为什么训练需要调度

一个物理 batch 的训练图有 28~32 个节点，每个节点都要回答：

```text
① 在哪个 backend 上执行？          → split 决策
② 输出的张量放在哪块显存的哪个位置？  → gallocr 分配
③ 前一个张量的内存什么时候能还回来？  → 生命周期分析
④ 跨 backend 的输入怎么搬？         → copy 节点 + event 同步
```

mnist 的情况简单（单 GPU、单 split），但流程一步不少。真实 LLM 的 `n_splits` 可能是几十个（部分层 offload 到 CPU），`n_inputs` 可能是几百个，这套机制完全一样。

---

## 1 split 决策：五趟分析

### 1.1 断点

```gdb
break ggml_backend_sched_alloc_splits
commands
silent
printf "SCHED_ALLOC graph_nodes=%d n_splits=%d is_reset=%d is_alloc=%d cur_copy=%d n_copies=%d\n",
  sched->graph.n_nodes, sched->n_splits, sched->is_reset, sched->is_alloc, sched->cur_copy, sched->n_copies
set $i = 0
while $i < sched->n_splits
printf "  split[%d] backend=%s nodes=[%d,%d) n_inputs=%d\n", $i,
  ggml_backend_name(sched->backends[sched->splits[$i].backend_id]),
  sched->splits[$i].i_start, sched->splits[$i].i_end, sched->splits[$i].n_inputs
set $i = $i + 1
end
continue
end
```

### 1.2 原始输出

```text
SCHED_ALLOC#1 graph_nodes=28 n_splits=1
  split[0] backend=CUDA0 nodes=[0,28) n_inputs=0

SCHED_ALLOC#2 graph_nodes=34 n_splits=1
  split[0] backend=CUDA0 nodes=[0,32) n_inputs=1
    input[0] name=adamw_params buffer=0x... data=0x55b5...
    graph_copy[0]=adamw_params (view)
    graph_copy[1]=CUDA0#adamw_params#0

SCHED_ALLOC#3 graph_nodes=28 n_splits=1
  split[0] backend=CUDA0 nodes=[0,28) n_inputs=0
```

### 1.3 五趟分析在做什么

`ggml_backend_sched_split_graph` 的结构：

```text
pass 1  给 leaf / node 分配 backend_id（看已有 buffer、INPUT 标记、权重所在后端）
pass 2  把 GPU backend 的分配向相邻节点扩展（CPU 不扩展，作为兜底）
pass 3  处理仍未分配的节点（按能力 + 优先级）
pass 4  处理 view、跨后端 src
pass 5  按 backend_id 切分 splits[]，标记需要跨后端拷贝的 input
```

**pass 1 的核心**：

```cpp
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // ① 张量已经在某个 buffer 上 → 就归那个 backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) return cur_backend_id;

    // ② view → 跟 view_src 同后端
    if (tensor->view_src != NULL) { ... }

    // ③ 图输入 → 最后一个后端（CPU）
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        return sched->n_backends - 1;
    }

    // ④ 有权重 → 跟权重同后端（除非 GPU 想 offload）
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) &&
                        ggml_backend_offload_op(sched->backends[b], tensor)) {
                        return b;                 // GPU 主动接管
                    }
                }
            }
            return src_backend_id;
        }
    }
    return -1;
}
```

本例的实际情况：

| 节点 | 命中哪条规则 | 结果 |
|---|---|---|
| `fc1.weight`（op=NONE） | ① 已在 CUDA0 buffer | CUDA0 |
| `images`（INPUT） | ③ 图输入 | 但 `images` 已在 CUDA0 buffer，规则①优先 → CUDA0 |
| `node_2`（MUL_MAT） | ④ 权重在 CUDA0 | CUDA0 |
| `logits`（ADD） | ④ 源在 CUDA0 | CUDA0 |
| AdamW 节点 | ④ `src[0]` 是参数，在 CUDA0 | CUDA0 |

**`images` 是 INPUT，为什么没有落到 CPU？** 因为规则①优先：`images` 已经被 `ggml_backend_alloc_ctx_tensors` 绑定到 CUDA0 buffer。`ggml_set_input` 只是打标记，不改变已有归属。这正是“静态 buffer 地址固定”带来的效果——用户可以用 `ggml_backend_sched_set_tensor_backend` 显式指定，否则“已在哪就在哪”。

### 1.4 为什么只有一个 split

pass 5 的切分条件是“相邻节点的 backend_id 不同”或“权重在别的不兼容后端”。本例所有节点都是 CUDA0，所以：

```text
i_split = 0
split[0].backend_id = 0
split[0].i_start = 0
split[0].i_end = 28（或 32）
sched->n_splits = 1
```

**CPU 一次都没用到**。`backends[2] = CPU` 只是满足 `ggml_backend_sched_new` 的断言，并在 CUDA 不支持某算子时兜底。本例所有算子 CUDA 都支持（`supports_op` 对 `MUL_MAT / ADD / UNARY / CROSS_ENTROPY_LOSS / OPT_STEP_ADAMW / ARGMAX / COUNT_EQUAL / OUT_PROD / REPEAT_BACK / SCALE` 全部返回 true）。

### 1.5 为什么 `gb_opt` 有 `n_inputs=1`

`adamw_params` 在 CPU buffer（`opt_ctx->ctx_cpu`），AdamW 节点在 CUDA0。pass 5 检测到：

```cpp
if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
    // 创建跨后端拷贝
    struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
    ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
    ...
    split->inputs[n_inputs++] = src;
    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
}
```

于是 `gb_opt` 的 `graph_copy` 里多了两个节点：

```text
graph_copy[0] = adamw_params (view)       ← ggml_view_tensor(sched->ctx, input)，依赖源
graph_copy[1] = CUDA0#adamw_params#0      ← 拷贝目标，在 CUDA0 上
```

**为什么需要 `view` 节点？** 源码注释：

```cpp
// add a dependency to the input source so that it is not freed before the copy is done
```

`sched->ctx` 里的 `adamw_params (view)` 的 `src[0]` 指向真正的 CPU 张量，这样 gallocr 在做生命周期分析时会把它算作“消费者”，不会在拷贝完成前释放 CPU 侧的 `adamw_params`。

**实际拷贝在什么时候发生？** 不在 split 阶段，而在 `compute_splits`：

```cpp
for (int input_id = 0; input_id < split->n_inputs; input_id++) {
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
    struct ggml_tensor * input = split->inputs[input_id];
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        ggml_backend_synchronize(split_backend);
        ggml_backend_tensor_copy(input, input_cpy);       // CPU → CUDA
    } else {
        ggml_backend_event_wait(split_backend, ...);
        if (!split_backend->iface.cpy_tensor_async || !cpy_tensor_async(...)) {
            ggml_backend_synchronize(input_backend);
            ggml_backend_synchronize(split_backend);
            ggml_backend_tensor_copy(input, input_cpy);
        }
    }
}
```

`adamw_params` 打了 `INPUT` 标记，所以走**同步拷贝**路径，每个 opt batch 一次。7 个 float = 28 B，实测 400 B 的 H2D+sync 是 5.41 µs，可忽略。

---

### 1.6 源码逐行：五趟分析的完整代码

`ggml_backend_sched_split_graph` 在 `src/ggml-backend.cpp:1066-1589`，约 520 行。下面按 pass 拆开。

#### pass 1：从已有归属推导 backend（1087-1122）

```cpp
1087      // pass 1: assign backends to ops with pre-allocated inputs
1088      for (int i = 0; i < graph->n_leafs; i++) {
1089          struct ggml_tensor * leaf = graph->leafs[i];
1090          int * leaf_backend_id = &tensor_backend_id(leaf);
1091          // do not overwrite user assignments
1092          if (*leaf_backend_id == -1) {
1093              *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
1094          }
1095      }
1096
1097      for (int i = 0; i < graph->n_nodes; i++) {
1098          struct ggml_tensor * node = graph->nodes[i];
1099          int * node_backend_id = &tensor_backend_id(node);
1100          if (*node_backend_id == -1) {
1101              *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
1102          }
1103      }
```

**`tensor_backend_id` 是宏**（`src/ggml-backend.cpp:756`）：

```cpp
#define tensor_backend_id(t) sched->hv_tensor_backend_ids[hash_id(t)]
```

**`hash_id(t)`**：

```cpp
#define hash_id(t) ggml_hash_find(&sched->hash_set, t)
```

所以 backend_id 存在**按哈希槽位索引的数组**里，不是张量结构体的字段。**为什么要这样**：同一个张量可能在不同图里出现（`gb_grad` / `gb_opt` 共享前 14 个节点），用哈希索引可以在 `sched_reset` 时一次 `memset(-1)` 清空所有归属，不需要遍历张量。

**`-1` 的含义**：未分配。`ggml_backend_sched_reset` 会：

```cpp
memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(int));
```

**用户显式指定的归属**（`ggml_backend_sched_set_tensor_backend`）不会被覆盖，这就是 1092/1100 的 `if (*id == -1)`。

#### pass 2：GPU 后端向上下扩展（1124-1202）

pass 2 有四个子循环，方向是 `down → up → down → up`：

```cpp
1129      // expand gpu down
1130      {
1131          int cur_backend_id = -1;
1132          for (int i = 0; i < graph->n_nodes; i++) {
1133              struct ggml_tensor * node = graph->nodes[i];
1134              if (ggml_is_view_op(node->op)) {
1135                  continue;                              // ← 视图跳过
1136              }
1137              int * node_backend_id = &tensor_backend_id(node);
1138              if (*node_backend_id != -1) {
1139                  if (*node_backend_id == sched->n_backends - 1) {
1140                      // skip cpu (lowest prio backend)
1141                      cur_backend_id = -1;               // ← CPU 不扩展
1142                  } else {
1143                      cur_backend_id = *node_backend_id;
1144                  }
1145              } else if (cur_backend_id != -1) {
1146                  ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
1147              }
1148          }
1149      }
```

**为什么是 4 个方向**：图是 DAG，一个已分配的 GPU 节点可能影响它前面（上游）和后面（下游）的节点。先 down（前→后）再 up（后→前），再重复一次，是为了让“隔一个未分配节点”的情况也能覆盖到。

**为什么 CPU 不扩展**（1139-1141）：CPU 是 fallback，只有“权重在 CPU”或“GPU 不支持这个 op”时才应该用。如果 CPU 也扩展，整个图会滑向 CPU。

**`ggml_backend_sched_set_if_supported`**（1058-1063）：

```cpp
static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}
```

**`supports_op` 是唯一的门槛**：CUDA 后端对 `MUL_MAT/ADD/UNARY/...` 都返回 true，所以 pass 2 会把所有节点都染成 CUDA0。

#### pass 3：未分配节点的兜底 + 升级（1204-1263）

```cpp
1212      for (int i = 0; i < graph->n_nodes; i++) {
1213          struct ggml_tensor * node = graph->nodes[i];
1214          if (ggml_is_view_op(node->op)) continue;
1215          int * node_backend_id = &tensor_backend_id(node);
1216          if (*node_backend_id == -1) {
1217              // unassigned node: find the backend with the most supported inputs
1218              int n_supported_best = -1;
1219              for (int b = 0; b < sched->n_backends; b++) {
1220                  if (ggml_backend_supports_op(sched->backends[b], node)) {
1221                      int n_supported = 0;
1222                      for (int j = 0; j < GGML_MAX_SRC; j++) {
1223                          struct ggml_tensor * src = node->src[j];
1224                          if (src == NULL) continue;
1225                          if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) &&
1226                              ggml_backend_sched_buffer_supported(sched, src, b)) {
1227                              n_supported++;
1228                          }
1229                      }
1230                      if (n_supported > n_supported_best) {
1231                          n_supported_best = n_supported;
1232                          *node_backend_id = b;
1233                          SET_CAUSE(node, "3.best");
1234                      }
1235                  }
1236              }
1237          } else {
1238              // assigned node: upgrade to higher prio backend if possible
1239              for (int b = 0; b < *node_backend_id; b++) {
1240                  if (sched->bufts[b] == sched->bufts[*node_backend_id] &&
1241                      ggml_backend_supports_op(sched->backends[b], node)) {
1242                      ...
1243                      *node_backend_id = b;
1244                      SET_CAUSE(node, "3.upg");
1245                      break;
1246                  }
1247              }
1248          }
1249      }
```

**两个分支**：

1. **仍未分配**（1216）：选“支持最多 src 的后端”。这是最后的兜底，本例不会触发（pass 2 已经全分配）。
2. **已分配但可以升级**（1237）：如果有一个更高优先级（`b < current`）的后端，且 **buft 相同**、支持这个 op、所有 src 的 buffer 也兼容，就升级过去。

**`bufts[b] == bufts[current]` 的意义**：只在“同一块内存类型”的后端之间升级（比如 CUDA0 和 CUDA0 的第二个实例），跨 buft 升级会导致数据拷贝，得不偿失。

#### pass 4：view 和 src 的兜底（1265-1295）

```cpp
1266      for (int i = 0; i < graph->n_nodes; i++) {
1267          struct ggml_tensor * node = graph->nodes[i];
1268          int * cur_backend_id = &tensor_backend_id(node);
1269          if (node->view_src != NULL && *cur_backend_id == -1) {
1270              *cur_backend_id = tensor_backend_id(node->view_src);   // ← view 跟源
1271              SET_CAUSE(node, "4.vsrc");
1272          }
1273          for (int j = 0; j < GGML_MAX_SRC; j++) {
1274              struct ggml_tensor * src = node->src[j];
1275              if (src == NULL) continue;
1276              int * src_backend_id = &tensor_backend_id(src);
1277              if (*src_backend_id == -1) {
1278                  if (src->view_src != NULL) {
1279                      *src_backend_id = tensor_backend_id(src->view_src);
1280                      SET_CAUSE(src, "4.vsrc");
1281                  } else {
1282                      *src_backend_id = *cur_backend_id;             // ← src 跟 dst
1283                      SET_CAUSE(src, "4.cur");
1284                  }
1285              }
1286          }
1287          // if the node is still unassigned, assign it to the first backend that supports it
1288          for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
1289              ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
1290          }
1291          GGML_ASSERT(*cur_backend_id != -1);
1292      }
```

**三条兜底规则**：

| 情况 | 归属 |
|---|---|
| 节点是 view 且未分配 | 跟 `view_src` |
| src 是 view 且未分配 | 跟 `view_src` |
| src 不是 view 且未分配 | 跟当前节点（`4.cur`） |

**`4.cur` 的危险性**：它假设“src 和 dst 在同一个后端”。如果 src 真的在另一个后端，pass 5 会检测到并插入拷贝。所以 `4.cur` 只是“乐观假设”，不是“强制绑定”。

**`GGML_ASSERT(*cur_backend_id != -1)`（1291）**：所有节点必须被分配。如果走到这里还是 -1，说明没有任何后端支持这个 op——程序会 abort。

#### pass 5：切分 split + 标记拷贝（1297-1425）

```cpp
1297      // pass 5: split graph, find tensors that need to be copied
1298      {
1299          int i_split = 0;
1300          struct ggml_backend_sched_split * split = &sched->splits[0];
1301          // find the backend of the first split, skipping view ops
1302          int i = 0;
1303          for (; i < graph->n_nodes; i++) {
1304              struct ggml_tensor * node = graph->nodes[i];
1305              if (!ggml_is_view_op(node->op)) {
1306                  split->backend_id = tensor_backend_id(node);
1307                  break;
1308              }
1309          }
1310          split->i_start = 0;
1311          split->n_inputs = 0;
1312          int cur_backend_id = split->backend_id;
1313          for (; i < graph->n_nodes; i++) {
1314              struct ggml_tensor * node = graph->nodes[i];
1315
1316              if (ggml_is_view_op(node->op)) {
1317                  continue;
1318              }
1319
1320              const int node_backend_id = tensor_backend_id(node);
1321              GGML_ASSERT(node_backend_id != -1);
1322
1323              // check if we should start a new split based on the sources of the current node
1324              bool need_new_split = false;
1325              if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
1326                  for (int j = 0; j < GGML_MAX_SRC; j++) {
1327                      struct ggml_tensor * src = node->src[j];
1328                      if (src == NULL) continue;
1329                      if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
1330                          int src_backend_id = tensor_backend_id(src);
1331                          if (src_backend_id != cur_backend_id &&
1332                              !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
1333                              need_new_split = true;
1334                              break;
1335                          }
1336                      }
1337                  }
1338              }
1339
1340              if (node_backend_id != cur_backend_id || need_new_split) {
1341                  split->i_end = i;
1342                  i_split++;
1343                  ...（扩容 splits 数组）...
1344                  split = &sched->splits[i_split];
1345                  split->backend_id = node_backend_id;
1346                  split->i_start = i;
1347                  split->n_inputs = 0;
1348                  cur_backend_id = node_backend_id;
1349              }
1350
1351              // find inputs that are not on the same backend
1352              for (int j = 0; j < GGML_MAX_SRC; j++) {
1353                  struct ggml_tensor * src = node->src[j];
1354                  if (src == NULL) continue;
1355
1356                  size_t src_id = hash_id(src);
1357                  const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
1358                  GGML_ASSERT(src_backend_id != -1);
1359
1360                  if (src_backend_id != cur_backend_id &&
1361                      !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
1362                      // create a copy of the input in the split's backend
1363                      if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
1364                          ...
1365                          split->inputs[split->n_inputs++] = src;
1366                      }
1367                      node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
1368                  }
1369              }
1370          }
1371          split->i_end = graph->n_nodes;
1372          sched->n_splits = i_split + 1;
1373      }
```

**三个切分条件**（1340）：

1. `node_backend_id != cur_backend_id`：相邻节点的后端不同；
2. `need_new_split`：权重在别的不兼容后端（1331-1332）；
3. （隐含）图结束。

**`need_new_split` 的场景**：真实 LLM 的 CPU offload。如果 `MUL_MAT` 在 CUDA0，但权重在 CPU，且 CUDA0 不支持这个 buft，就必须新开一个 split，让 CPU 算这个节点。

**本例两个条件都不触发**，所以 `n_splits=1`。

**1360-1368 是跨后端拷贝的核心**：

```cpp
if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
        // 第一次遇到：创建拷贝张量 + 记录到 split->inputs
        ...
        split->inputs[split->n_inputs++] = src;
    }
    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);   // ← 改写 src！
}
```

**`node->src[j] = ...` 是原地修改图**。这就是为什么 scheduler 必须维护 `graph_copy`（§2）——如果不复制，`gb_opt` 的 `src` 会被永久改写，下次复用就错了。

**`tensor_id_copy(src_id, backend_id, copy)` 的存储**：

```cpp
#define tensor_id_copy(id, backend_id, copy_id)     sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
```

**三维索引**：`[哈希槽位][后端][拷贝编号]`。本例 `n_copies=1`，所以只有 `[id][backend][0]`。

**本例唯一触发的是 `adamw_params`**：

```text
src = adamw_params（CPU buffer）
src_backend_id = 2（CPU）
cur_backend_id = 0（CUDA0）
buffer_supported(CUDA0, CPU buft) = false
→ 创建 CUDA0#adamw_params#0，split->inputs[0] = adamw_params
```

---

## 2 graph_copy 与 compute_splits：调度器的“私有图”和执行

### 2.1 为什么要复制一张图

split 阶段会**修改 `node->src[]`**：

```cpp
node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
```

如果不复制，原始图（`gb_grad` / `gb_opt`）的 `src` 会被永久改写，下一次复用就错了。所以 scheduler 维护自己的 `sched->graph`（`graph_copy`），把 split 的节点和拷贝节点拼进去：

```cpp
// 每个 split 的输入
for (int j = 0; j < split->n_inputs; j++) {
    graph_copy->nodes[graph_copy->n_nodes++] = input_dep;   // 依赖
    graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;   // 拷贝目标
}
// split 的节点
for (int j = split->i_start; j < split->i_end; j++) {
    graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
}
```

### 2.2 节点数变化

| 图 | 原始节点 | `graph_copy` 节点 | 差额 |
|---|---|---|---|
| `gb_grad` | 28 | 28 | 0 |
| `gb_opt` | 32 | 34 | +2（input_dep + input_cpy） |
| 推理 `gf` | 13 | 13 | 0 |

### 2.3 `n_copies=1` 的含义

```cpp
backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(),
                                       GGML_DEFAULT_GRAPH_SIZE, /*parallel=*/false, /*op_offload=*/true);
```

`parallel=false` → `sched->n_copies = 1`。断点：

```text
SCHED_ALLOC n_splits=1 ... cur_copy=0 n_copies=1
```

`n_copies > 1` 时，scheduler 会为每个 input 创建多份拷贝，用 event 做流水线重叠（真实 LLM 的 batch 推理会开）。本例单线程训练，不需要。

---

### 2.4 源码逐行：`ggml_backend_sched_compute_splits`（`src/ggml-backend.cpp:1646-1846`）

这是“执行”阶段：遍历 split，处理跨后端拷贝，提交 backend。

```cpp
1646  static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
1647      GGML_ASSERT(sched);
1648      struct ggml_backend_sched_split * splits = sched->splits;
1649
1650      ggml_tensor * prev_ids_tensor = nullptr;
1651      std::vector<int32_t> ids;
1652      std::vector<ggml_bitset_t> used_ids;
1653
1654      int prev_backend_id = -1;
1655
1656      for (int split_id = 0; split_id < sched->n_splits; split_id++) {
1657          struct ggml_backend_sched_split * split = &splits[split_id];
1658          int split_backend_id = split->backend_id;
1659          ggml_backend_t split_backend = sched->backends[split_backend_id];
1660
1661          // ensure the previous split's async work has completed before we start
1662          // this split, the allocator may have reused buffer regions across splits
1663          if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
1664              if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
1665                  ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
1666              } else {
1667                  ggml_backend_synchronize(sched->backends[prev_backend_id]);
1668              }
1669          }
1670
1671          // copy the input tensors to the split backend
1672          for (int input_id = 0; input_id < split->n_inputs; input_id++) {
1673              ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
1674              struct ggml_tensor * input = split->inputs[input_id];
1675              struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
1676
1677              if (input->flags & GGML_TENSOR_FLAG_INPUT) {
1678                  // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
1679                  if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
1680                      ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
1681                  } else {
1682                      ggml_backend_synchronize(split_backend);
1683                  }
1684                  ggml_backend_tensor_copy(input, input_cpy);
1685              } else {
1686                  // wait for the split backend to finish using the input before overwriting it
1687                  if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
1688                      ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
1689                  } else {
1690                      ggml_backend_synchronize(split_backend);
1691                  }
1692
1693                  // （MoE 专家局部拷贝，本例不触发，省略 1693-1795）
...
1796              }
1797          }
1798
1799          if (!sched->callback_eval) {
1800              enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
1801              if (ec != GGML_STATUS_SUCCESS) {
1802                  return ec;
1803              }
1804          } else {
1805              // （逐节点 callback 模式，本例不触发）
1805          }
1836
1837          // record the event of this split
1838          if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
1839              ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
1840          }
1841
1842          prev_backend_id = split_backend_id;
1843      }
1844
1845      return GGML_STATUS_SUCCESS;
1846  }
```

#### 2.4.1 第 1663-1669 行：split 间同步

```cpp
if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
    // 同步前一个 split 的后端
}
```

**三个条件**：

1. `split->n_inputs == 0`：当前 split 没有输入拷贝（如果有，会在 1677 的循环里同步）；
2. `prev_backend_id >= 0`：不是第一个 split；
3. `prev_backend_id != split_backend_id`：前后 split 在不同后端。

**为什么需要同步**：因为 gallocr 可能把前后两个 split 的张量放在同一块显存的相同 offset（生命周期分析是全局的）。如果前一个 split 的异步 kernel 还没跑完，后一个 split 就会覆盖数据。

**本例 `n_splits=1`**，这个分支永远不触发。

#### 2.4.2 第 1677-1684 行：用户输入的同步拷贝

```cpp
if (input->flags & GGML_TENSOR_FLAG_INPUT) {
    // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
    if (sched->events[...] != NULL) {
        ggml_backend_event_synchronize(...);
    } else {
        ggml_backend_synchronize(split_backend);
    }
    ggml_backend_tensor_copy(input, input_cpy);
}
```

**为什么 INPUT 要同步**：用户可能在 `compute_splits` 返回后立刻改写 `input` 的 host 内存。如果拷贝是异步的，用户改写会污染正在进行的 DMA。所以必须**同步拷贝**。

**本例**：`adamw_params` 是 INPUT，走这条路径。7 个 float = 28 B，实测 400 B 的 H2D+sync 是 5.41 µs。

#### 2.4.3 第 1685-1696 行：普通张量的异步拷贝

```cpp
} else {
    // wait for the split backend to finish using the input before overwriting it
    if (sched->events[...] != NULL) {
        ggml_backend_event_wait(split_backend, sched->events[...]);
    } else {
        ggml_backend_synchronize(split_backend);
    }
    // 尝试 async copy，失败则同步 copy
    if (!split_backend->iface.cpy_tensor_async || !cpy_tensor_async(...)) {
        ggml_backend_synchronize(input_backend);
        ...
        ggml_backend_tensor_copy(input, input_cpy);
    }
}
```

**与 INPUT 的区别**：

| | INPUT | 普通张量 |
|---|---|---|
| 同步对象 | `split_backend` | `split_backend`（等它用完） |
| 拷贝方式 | 强制同步 | 优先 async |
| 原因 | 用户可能改写 host 数据 | 数据在另一个后端，不会被用户碰 |

**本例不触发**：`gb_grad` 的 `n_inputs=0`；`gb_opt` 的 `adamw_params` 是 INPUT。

#### 2.4.4 第 1799-1803 行：提交 split

```cpp
if (!sched->callback_eval) {
    enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
    if (ec != GGML_STATUS_SUCCESS) {
        return ec;
    }
}
```

**`split->graph` 是 `ggml_graph_view`**（在 `split_graph` 的 1460 行创建）：

```cpp
split->graph = ggml_graph_view(graph, split->i_start, split->i_end);
```

**`ggml_graph_view` 不复制节点**，只是把 `nodes` 指针指向原图的 `[i_start, i_end)` 区间。所以 backend 看到的是一段“子图”。

**`callback_eval` 模式**（1804-1835）：如果用户设置了回调（用于逐层调试），scheduler 会把 split 再切成更小的段，每段执行后同步、回调。本例没有回调。

#### 2.4.5 第 1837-1840 行：记录 event

```cpp
if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
    ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
}
```

**`events` 只在 `n_copies > 1` 时创建**（`sched_new` 的 1905-1909 行）。本例 `n_copies=1`，所以 `events[...] == NULL`，不记录。

**event 的用途**：多 copy 流水线时，下一个 copy 的拷贝要等当前 copy 的 compute 完成。用 event 而不是 `cudaStreamSynchronize` 是为了**不阻塞 host**。

---

## 3 gallocr：从图到 offset

### 3.1 调用链

```text
ggml_backend_sched_alloc_graph(sched, graph)
  ├── split_graph()                  → n_splits, graph_copy
  └── alloc_splits()
        ├── backend_ids_changed?     → 判断是否需要重新 reserve
        └── ggml_gallocr_alloc_graph(galloc, &sched->graph)
              ├── needs_realloc()    → 节点数/叶子数/地址是否变化
              │     ├── 否 → init_tensor() 把旧 offset 写回
              │     └── 是 → reserve() → 重新规划 → init_tensor()
              └── ...
```

### 3.2 第一次分配：28 个节点

```text
NEEDS_REALLOC galloc_nodes=0 graph_nodes=28 galloc_leafs=0 graph_leafs=7
ggml_gallocr_needs_realloc: graph has different number of nodes
ggml_gallocr_alloc_graph: cannot reallocate multi buffer graph automatically, call reserve
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
```

**为什么“cannot reallocate multi buffer graph automatically”？**

```cpp
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
            ggml_gallocr_reserve(galloc, graph);       // 单 buffer 可以自动 realloc
        } else {
            return false;                              // 多 buffer 需要调用方显式 reserve
        }
    }
    ...
}
```

本例 `galloc->n_buffers = 3`（CUDA0、CUDA0、CPU），所以自动路径不可用，必须由 `alloc_splits` 捕获失败后调用 `ggml_gallocr_reserve_n`。

**为什么多 buffer 不能自动 realloc？** 因为跨 buffer 的拷贝节点地址会变，scheduler 需要先同步所有 backend、再重新 reserve、再重新分配，这套流程只有 scheduler 知道（它持有 `events` 和 `splits`）。gallocr 自己做不到。

### 3.3 reserve 阶段的分配日志

```gdb
break ggml-alloc.c:264          # dyn_tallocr_alloc 的 block->offset += size 之后
break ggml-alloc.c:710          # free_node 的 free_bytes 之前
```

```text
ALLOC node_2                        size= 1000064 chunk=0 off=       0
ALLOC node_5                        size= 1000064 chunk=0 off= 1000064
ALLOC node_6                        size=   20096 chunk=0 off= 2000128
ALLOC grad for loss_cross_entropy   size=     128 chunk=0 off= 2020224
ALLOC grad for node_6               size=   20096 chunk=0 off= 2020352
FREE  grad for loss_cross_entropy   off= 2020224 size=       4
ALLOC node_16                       size=     128 chunk=0 off= 2020224   ← 复用
FREE  node_16                       off= 2020224 size=      40
ALLOC node_18                       size=   20096 chunk=0 off= 2040448
FREE  node_5                        off= 1000064 size= 1000000
FREE  node_18                       off= 2040448 size=   20000
ALLOC grad for node_5               size= 1000064 chunk=0 off= 1000064   ← 复用
FREE  grad for node_6               off= 2020352 size=   20000
FREE  grad for node_5               off= 1000064 size= 1000000
ALLOC node_24                       size=    2048 chunk=0 off= 1000064   ← 复用
FREE  node_24                       off= 1000064 size=    2000
ALLOC node_26                       size= 1568000 chunk=0 off= 2020224   ← 复用
FREE  grad for node_2               off=       0 size= 1000000
FREE  node_26                       off= 2020224 size= 1568000
RESERVE buffer_id=0 new_size=0
RESERVE buffer_id=1 new_size=0
```

**注意 `size` 与 `FREE ... size` 的差异**：`ALLOC` 打印的是对齐后的分配大小（`1,000,064`），`FREE` 打印的是 `ggml_backend_buft_get_alloc_size` 返回的**未对齐**大小（`1,000,000`）。这是 gdb 断点打在两个不同位置造成的：

```cpp
// allocate_node → dyn_tallocr_alloc（ggml-alloc.c:203）
size = aligned_offset(NULL, size, alloc->alignment);   // 1,000,000 → 1,000,064
...
block->offset += size;      // ← 断点在这里，size 已对齐
block->size   -= size;

// free_node → dyn_tallocr_free_bytes（ggml-alloc.c:312）
size_t size = ggml_backend_buft_get_alloc_size(buft, node);   // 1,000,000
ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);           // ← 断点在调用前，size 未对齐
```

**这不漏字节。** `free_bytes` 内部也会对齐：

```cpp
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);   // 1,000,000 → 1,000,064
    ...
}
```

验算：`1,000,000 % 128 = 64`，对齐后 `1,000,000 + (128-64) = 1,000,064`，与分配时完全一致。所以 free list 里不会残留对齐碎片——**alloc 和 free 用的是同一个 `aligned_offset` 函数**，只是 gdb 观察点选在了对齐之前/之后。

### 3.4 offset 布局的可视化

```text
offset (B)     0        1,000,064   2,000,128  2,020,224  2,040,448   3,588,352
               │            │            │           │           │            │
node_2         ├────────────┤            │           │           │            │
node_5         │            ├────────────┤           │           │            │
node_6         │            │            ├── 20,096 ┤           │            │
grad loss      │            │            │           ├── 128 ────┤            │
grad node_6    │            │            │           │  ├─20,096─┤            │
node_16        │            │            │           │  ↑复用   │            │
node_18        │            │            │           │      ├─20,096─┤        │
grad node_5    │            ├── 1,000,064（复用 node_5）─────────┤            │
node_24        │            ├─ 2,048（复用）│           │           │        │
node_26        │            │            │           ├─1,568,000─┤            │
               └────────────┴────────────┴───────────┴───────────┴────────────┘
```

### 3.5 复用链的源码依据

```cpp
// ggml-alloc.c:764
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];

    // 分配父节点（此时只剩 leaf 需要分配）
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        ggml_gallocr_allocate_node(galloc, node->src[j], buffer_id);
    }

    // 分配本节点
    ggml_gallocr_allocate_node(galloc, node, buffer_id);

    // 更新父节点：消费计数减一，归零则释放
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
        p_hn->n_children -= 1;
        if (p_hn->n_children == 0 && p_hn->n_views == 0) {
            ggml_gallocr_free_node(galloc, parent);
        }
    }
}
```

**关键：分配与释放都发生在“遍历 `nodes[]`”的过程中，而不是等到整张图算完。** 所以 offset 规划是**顺序的、单遍的**：

```text
访问 node_i  → 分配它的输出 → 减少它所有 src 的计数 → 归零的 src 立即 free
```

这要求 `nodes[]` 是拓扑序（后序 DFS 保证），否则会在父节点还没消费完时就释放。

### 3.6 复用发生的三个条件

```cpp
if (p_hn->n_children == 0 && p_hn->n_views == 0) {
    ggml_gallocr_free_node(galloc, parent);
}
```

1. **所有消费者都已执行**（`n_children == 0`）
2. **没有存活的视图**（`n_views == 0`）
3. **不是图输出**（`free_node` 里检查 `GGML_TENSOR_FLAG_OUTPUT`）

断点验证：

- `node_5` 在 `node_18`（OUT_PROD 的反向）执行后被释放——`node_5` 的消费者是 `node_6`（前向 MUL_MAT）和 `node_18`（反向 OUT_PROD），两个都执行完才 `n_children=0`。
- `logits` 永远不会被释放——它打了 `OUTPUT` 标记。
- `grad for node_6 (transposed)` 是视图，它会让 `grad for node_6` 的 `n_views=1`，所以 `grad for node_6` 要等转置视图也不用了才释放。

---

### 3.7 源码逐行：`ggml_gallocr_alloc_graph_impl`（`src/ggml-alloc.c:718-823`）

这是 gallocr 的“规划主循环”：一次遍历 `nodes[]`，完成分配、释放、复用。

```cpp
718  static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
719      // clear hash tables
720      ggml_hash_set_reset(&galloc->hash_set);
721      memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);
722
723      // allocate leafs
724      for (int i = 0; i < graph->n_leafs; i++) {
725          struct ggml_tensor * leaf = graph->leafs[i];
726          ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
727      }
728
729      // count number of children and views
730      for (int i = 0; i < graph->n_nodes; i++) {
731          struct ggml_tensor * node = graph->nodes[i];
732
733          if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
734              struct ggml_tensor * view_src = node->view_src;
735              ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;      // ← 统计视图
736          }
737
738          if (node->flags & GGML_TENSOR_FLAG_INPUT) {
739              ggml_gallocr_allocate_node(galloc, node, get_node_buffer_id(node_buffer_ids, i));
740          }
741
742          for (int j = 0; j < GGML_MAX_SRC; j++) {
743              struct ggml_tensor * src = node->src[j];
744              if (src == NULL) continue;
745              ggml_gallocr_hash_get(galloc, src)->n_children += 1;         // ← 统计消费者
746
747              if (src->flags & GGML_TENSOR_FLAG_INPUT) {
748                  ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
749              }
750          }
751      }
752
753      // allocate tensors
754      for (int i = 0; i < graph->n_nodes; i++) {
755          struct ggml_tensor * node = graph->nodes[i];
756          int buffer_id = get_node_buffer_id(node_buffer_ids, i);
757
758          // allocate parents (only leafs need to be allocated at this point)
759          for (int j = 0; j < GGML_MAX_SRC; j++) {
760              struct ggml_tensor * parent = node->src[j];
761              if (parent == NULL) continue;
762              ggml_gallocr_allocate_node(galloc, parent, buffer_id);
763          }
764
765          // allocate node
766          ggml_gallocr_allocate_node(galloc, node, buffer_id);
767
768          // update parents
769          for (int j = 0; j < GGML_MAX_SRC; j++) {
770              struct ggml_tensor * parent = node->src[j];
771              if (parent == NULL) continue;
772              struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
773              p_hn->n_children -= 1;                                       // ← 消费一次
774
775              if (p_hn->n_children == 0 && p_hn->n_views == 0) {           // ← 归零释放
776                  if (ggml_impl_is_view(parent)) {
777                      struct ggml_tensor * view_src = parent->view_src;
778                      struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
779                      view_src_hn->n_views -= 1;
780                      if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
781                          ggml_gallocr_free_node(galloc, view_src);
782                      }
783                  }
784                  else if (p_hn->allocated) {
785                      ggml_gallocr_free_node(galloc, parent);
786                  }
787              }
788          }
789      }
790  }
```

#### 3.7.1 第 719-721 行：清空哈希

```cpp
ggml_hash_set_reset(&galloc->hash_set);
memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);
```

**`hash_values` 是 `struct hash_node` 数组**，按哈希槽位索引：

```cpp
struct hash_node {
    int n_children;      // 还有多少个节点会消费它
    int n_views;         // 有多少个存活的视图
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};
```

**`ggml_hash_set_reset` 只清 `used` bitset**，`keys` 数组保留（因为张量指针不变）。`memset(hash_values, 0)` 把 `n_children/n_views/allocated` 全部归零，但 `buffer_id` 和 `addr` 也被清零——这是故意的，因为它们会在 `allocate_node` 里重新赋值。

**`ggml_gallocr_hash_get`（584-587）**：

```cpp
static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    return &galloc->hash_values[i];
}
```

**`ggml_hash_find_or_insert`**：如果张量不在表里，插入并返回新槽位。所以第一次访问某个张量时，它的 `hash_node` 是零初始化的。

#### 3.7.2 第 730-751 行：两遍统计

**第一遍只统计，不分配**（除了 leaf 和 INPUT）：

```cpp
if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
    ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
}
...
ggml_gallocr_hash_get(galloc, src)->n_children += 1;
```

**为什么先统计**：因为 `allocate_node` 里的原地复用需要知道“父节点有几个消费者”。如果边分配边统计，早分配的节点看到的是不完整的计数。

**`node->op != GGML_OP_NONE` 的过滤**：scheduler 插入的 `input_dep`（`op=NONE` 的 view）不应该被算作视图，因为它只是“依赖占位符”，不消费源张量的内存。

**INPUT 张量提前分配**（738-740、747-749）：

```cpp
if (node->flags & GGML_TENSOR_FLAG_INPUT) {
    ggml_gallocr_allocate_node(galloc, node, ...);
}
```

**为什么 INPUT 要提前**：`images` 是图输入，它的数据由 `tensor_set` 写入，必须在所有计算之前有确定的地址。如果等到它的消费者才分配，`images` 可能在 `node_2` 执行时才拿到地址，而 `tensor_set` 已经发生过了。

**本例的 INPUT**：`images`（`leafs[0]`）、`labels`（`leafs[1]`）、`adamw_params`（`leafs[9]`）。它们都在静态 buffer 里，`allocate_node` 会直接返回（`ggml_gallocr_is_allocated` 为 true，因为 `data != NULL`）。

#### 3.7.3 第 754-789 行：分配主循环

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    int buffer_id = get_node_buffer_id(node_buffer_ids, i);

    // ① 分配父节点
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        ggml_gallocr_allocate_node(galloc, node->src[j], buffer_id);
    }

    // ② 分配自己
    ggml_gallocr_allocate_node(galloc, node, buffer_id);

    // ③ 更新父节点计数
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
        p_hn->n_children -= 1;
        if (p_hn->n_children == 0 && p_hn->n_views == 0) {
            ggml_gallocr_free_node(galloc, parent);
        }
    }
}
```

**为什么 ① 要在 ② 之前**：因为 `allocate_node(node)` 可能尝试“原地复用父节点”（§4）。如果父节点还没分配，就没有地址可复用。

**为什么 ① 对每个 node 都做**：因为父节点可能是“还没分配过的中间张量”。拓扑序保证父节点在 `nodes[]` 里出现在子节点之前，所以正常情况下父节点已经分配过；但 `ggml_view_tensor` 创建的视图可能没有自己的分配，需要在这里触发 `view_init`。

**`get_node_buffer_id`**：

```cpp
static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}
```

**`node_buffer_ids` 来自 scheduler**（`sched->node_backend_ids`），告诉 gallocr 每个节点应该在哪个 buffer 里。本例全部是 0（CUDA0）。

#### 3.7.4 断点对照：`gb_grad` 的执行序列

用本文 §3.3 的 ALLOC/FREE 日志对照源码：

| 源码行 | 日志 | 解释 |
|---|---|---|
| 754 `i=2`（node_2） | `ALLOC node_2 size=1000064 off=0` | 第一次分配，free list 空，取 chunk 尾部 |
| 762 分配 `node_2` 的父节点 | （无日志） | `fc1.weight` 和 `images` 已分配（静态 buffer） |
| 766 分配 `node_2` 自己 | `ALLOC node_2` | 同上 |
| 773 更新父节点 | （无 FREE） | `fc1.weight` 的 `n_children` 从 1 减到 0，但它是 PARAM/静态，`allocated=false`，不释放 |
| 754 `i=4`（node_4） | （无 ALLOC） | `node_4` 原地复用 `node_2`（§4） |
| 773 更新 `node_2` | （无 FREE） | `node_2` 的 `n_children` 从 1 减到 0，但 `node_4` 已经复用了它的地址，`allocated=false` |
| 754 `i=5`（node_5） | `ALLOC node_5 off=1000064` | 新分配 |
| 754 `i=9`（loss） | `ALLOC grad for loss... off=2020224` | 新分配 |
| 773 更新 `grad for loss` | `FREE grad for loss off=2020224 size=4` | `n_children=0`，释放 |
| 754 `i=16`（node_16） | `ALLOC node_16 off=2020224` | best-fit 复用刚释放的块 |
| 773 更新 `node_5`（在 `i=18` 之后） | `FREE node_5 off=1000064 size=1000000` | `node_5` 的两个消费者（`node_6`、`node_18`）都执行完 |
| 754 `i=21`（grad node_5） | `ALLOC grad for node_5 off=1000064` | 复用 `node_5` 的块 |

**注意 `node_4` 没有 ALLOC 也没有 FREE**：它复用了 `node_2` 的地址（`p_hn->allocated = false`），所以既不占新空间，也不释放。

---

## 4 原地复用（in-place）

### 4.1 断点：同一地址链

```text
[ 2] node_2   data=0x7f049d600000
[ 4] node_4   data=0x7f049d600000   ← ADD 复用 node_2
[22] node_22  data=0x7f049d600000   ← STEP 复用 node_4
[23] grad node_2 data=0x7f049d600000 ← MUL 复用 node_22

[ 5] node_5   data=0x7f049d6f4280
[21] grad node_5 data=0x7f049d6f4280 ← OUT_PROD 复用 node_5

[ 6] node_6   data=0x7f049d7e8500
[ 8] logits   data=0x7f049d7e8500   ← ADD 复用 node_6

[14] grad loss data=0x7f049d7ed380
[16] node_16   data=0x7f049d7ed380  ← REPEAT_BACK 复用 grad loss
```

### 4.2 原地复用的判定

```cpp
// ggml-alloc.c:623
static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (!ggml_gallocr_is_own(galloc, parent)) continue;              // 外部内存不能复用
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT) continue;           // 输出不能复用
                if (!ggml_are_same_layout(node, parent)) continue;               // 布局必须一致

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {               // 唯一消费者
                    hn->buffer_id = p_hn->buffer_id;
                    hn->addr = p_hn->addr;
                    p_hn->allocated = false;   // 避免 free
                    return;
                }
            }
        }

        // 否则从 tallocr 分配
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
    }
}
```

### 4.3 `ggml_op_can_inplace` 的白名单

```cpp
bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FILL:
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_CLAMP:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
            return true;
        default:
            return false;
    }
}
```

**注意 `MUL_MAT` 和 `OUT_PROD` 不在列表里。** 它们是 GEMM，输入输出形状不同，无法原地。但实测 `grad for node_5 = OUT_PROD(...)` 复用了 `node_5` 的地址——这不是原地复用，而是 **`node_5` 被释放后，free list 里的同一块被重新分配**。

两者的区别：

| | 原地复用 | free list 复用 |
|---|---|---|
| 发生时机 | `allocate_node` 里，节点分配前 | `dyn_tallocr_alloc` 里，best-fit 搜索 |
| 地址关系 | 与某个父节点**相同** | 与某个**已释放**节点相同 |
| 判定 | `op_can_inplace` + 唯一消费者 | free block 大小匹配 |
| 本例 | `node_4`、`node_22`、`logits`、`node_16` | `grad for node_5`、`node_24`、`node_26` |

### 4.4 原地复用为什么安全

以 `node_4 = ADD(node_2, fc1.bias)` 为例：

```text
node_2 是 [500,500] 的 MUL_MAT 结果，唯一的消费者是 node_4
ADD 逐元素：dst[i] = src0[i] + src1[i]
如果 dst 和 src0 同地址，就是 dst[i] = dst[i] + src1[i]，仍然正确
```

前提是**逐元素且读写同一位置**。对于 `MUL_MAT`（读 A 的所有行、写 C 的一行），原地会破坏数据，所以不在白名单。

### 4.5 原地复用带来的收益

如果不用原地复用，28 个节点需要的峰值显存：

```text
node_2 (1,000,064) + node_4 (1,000,064) + node_5 (1,000,064)
+ node_6 (20,096) + logits (20,096) + grad loss (128) + node_16 (128)
+ node_22 (1,000,064) + grad node_2 (1,000,064) + ...
```

粗略估算约 **6.6 MiB**。实际只用 **3.42 MiB**（3,588,352 B），节省 48%。

---

### 4.6 源码逐行：`ggml_gallocr_allocate_node`（`src/ggml-alloc.c:623-689`）

这是“原地复用”的唯一实现点。

```cpp
623  static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
624      GGML_ASSERT(buffer_id >= 0);
625      struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
626
627      if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
628          hn->allocated = true;
629          assert(hn->addr.offset == 0);
630
631          // try to reuse a parent's buffer (inplace)
632          if (ggml_op_can_inplace(node->op)) {
633              for (int i = 0; i < GGML_MAX_SRC; i++) {
634                  struct ggml_tensor * parent = node->src[i];
635                  if (parent == NULL) continue;
636
637                  // if the node's data is external, then we cannot re-use it
638                  if (!ggml_gallocr_is_own(galloc, parent)) {
639                      continue;
640                  }
641
642                  // outputs cannot be reused
643                  if (parent->flags & GGML_TENSOR_FLAG_OUTPUT ||
644                      (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
645                      continue;
646                  }
647
648                  if (!ggml_are_same_layout(node, parent)) {
649                      continue;
650                  }
651
652                  struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
653                  if (p_hn->n_children == 1 && p_hn->n_views == 0) {
654                      if (ggml_impl_is_view(parent)) {
655                          struct ggml_tensor * view_src = parent->view_src;
656                          struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
657                          if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
658                              hn->buffer_id = p_hn->buffer_id;
659                              hn->addr = p_hn->addr;
660                              p_hn->allocated = false;
661                              view_src_hn->allocated = false;
662                              ggml_gallocr_free_extra_space(galloc, node, view_src);
663                              return;
664                          }
665                      } else {
666                          hn->buffer_id = p_hn->buffer_id;
667                          hn->addr = p_hn->addr;
668                          p_hn->allocated = false;
669                          ggml_gallocr_free_extra_space(galloc, node, parent);
670                          return;
671                      }
672                  }
673              }
674          }
675
676          // allocate tensor from the buffer
677          struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
678          ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
679          size_t size = ggml_backend_buft_get_alloc_size(buft, node);
680          hn->buffer_id = buffer_id;
681          hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
682      }
683  }
```

#### 4.6.1 第 627 行：什么时候跳过

```cpp
if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
```

**两个条件**：

1. `!is_allocated`：张量还没地址（`data == NULL` 且 `buffer == NULL` 且 `hash_node.allocated == false`）；
2. `!is_view`：不是视图。

**视图为什么跳过**：视图的 `data` 由 `view_src->data + view_offs` 决定，不需要分配。`ggml_gallocr_init_tensor` 会处理：

```cpp
if (tensor->view_src != NULL) {
    if (tensor->buffer == NULL) {
        ggml_backend_view_init(tensor);
    }
}
```

**静态张量为什么跳过**：`is_allocated` 为 true（`data != NULL`），所以直接返回。

#### 4.6.2 第 632-674 行：原地复用的五个门槛

| 行 | 条件 | 含义 |
|---|---|---|
| 632 | `ggml_op_can_inplace(node->op)` | 算子必须支持原地（白名单见 §4.3） |
| 638 | `ggml_gallocr_is_own(galloc, parent)` | 父节点必须是 gallocr 分配的（不是静态/外部） |
| 643 | `!(parent->flags & OUTPUT)` | 输出不能被覆盖（`logits`、`loss`、`pred`、`ncorrect`） |
| 648 | `ggml_are_same_layout(node, parent)` | 形状 + 类型 + `nb` 全部相同 |
| 653 | `p_hn->n_children == 1 && p_hn->n_views == 0` | 父节点只有当前这一个消费者，且没有视图 |

**`ggml_are_same_layout`**（`src/ggml.c`）：

```cpp
bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) return false;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) return false;
        if (a->nb[i] != b->nb[i]) return false;
    }
    return true;
}
```

**注意 `nb` 也要相同**：转置视图的 `nb` 交换了，所以不能原地复用。

#### 4.6.3 第 653 行的 `n_children == 1`：为什么必须是 1

**如果父节点有 2 个消费者**：

```text
parent → node_A（原地复用）
parent → node_B（还要读 parent 的原始数据）
```

`node_A` 原地写会破坏 `parent`，`node_B` 读到的是被改过的数据——**错误**。

**如果父节点有视图**：

```text
parent → view（共享 parent 的 data）
parent → node_A（原地复用）
```

`node_A` 原地写会通过 `view` 被观察到，而 `view` 的语义是“父节点的快照”——**错误**。

**所以 `n_children == 1 && n_views == 0` 是充分必要条件**。

#### 4.6.4 第 666-670 行：复用时的三件事

```cpp
hn->buffer_id = p_hn->buffer_id;      // ① 继承 buffer
hn->addr      = p_hn->addr;           // ② 继承 offset
p_hn->allocated = false;              // ③ 标记父节点“不要释放”
ggml_gallocr_free_extra_space(galloc, node, parent);   // ④ 释放多余空间
```

**③ 为什么要把父节点标为 `allocated=false`**：因为 `free_node` 会检查这个标志：

```cpp
static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) return;
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    ...
    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}
```

如果父节点还是 `allocated=true`，当它的 `n_children` 归零时，`alloc_graph_impl` 的 784-786 行会调用 `free_node`，把这块内存还回去——但 `node` 正在用它，就会**双重释放**。

**④ `free_extra_space` 的作用**（`src/ggml-alloc.c:600-621`）：

```cpp
static void ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, struct ggml_tensor * node, struct ggml_tensor * parent) {
    size_t parent_size = ggml_backend_buft_get_alloc_size(buft, parent);
    size_t node_size   = ggml_backend_buft_get_alloc_size(buft, node);
    GGML_ASSERT(parent_size >= node_size);

    parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
    node_size   = aligned_offset(NULL, node_size,   p_alloc->alignment);

    if (parent_size > node_size) {
        struct buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}
```

**场景**：父节点是 `[500,500]`（1,000,064 B），子节点是 `[500,1]`（2,048 B）。复用父节点的地址后，子节点只用前 2,048 B，剩下的 998,016 B 被还给 free list。

**本例有没有触发**：没有。`node_4 = ADD(node_2, ...)` 的 `node_2` 和 `node_4` 都是 `[500,500]`，大小相同，`extra_size=0`。

#### 4.6.5 第 676-681 行：真正分配

```cpp
struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
size_t size = ggml_backend_buft_get_alloc_size(buft, node);
hn->buffer_id = buffer_id;
hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
```

**`size` 是未对齐的**（1,000,000），`dyn_tallocr_alloc` 内部会 `aligned_offset` 到 1,000,064。

---

### 4.7 源码逐行：`ggml_dyn_tallocr_alloc`（`src/ggml-alloc.c:202-310`）

这是 free list 分配器，负责“复用已释放的块”。

```cpp
202  static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
203      size = aligned_offset(NULL, size, alloc->alignment);      // ← 对齐
204
207      int best_fit_chunk = -1;
208      int best_fit_block = -1;
209      size_t max_avail = 0;
210
211      // find the best fitting free block besides the last block, within any chunk
212      for (int c = 0; c < alloc->n_chunks; ++c) {
213          struct tallocr_chunk * chunk = alloc->chunks[c];
214          size_t best_fit_size = SIZE_MAX;
215          for (int i = 0; i < chunk->n_free_blocks - 1; i++) {   // ← 注意 -1
216              struct free_block * block = &chunk->free_blocks[i];
217              max_avail = MAX(max_avail, block->size);
218              if (block->size >= size && block->size <= best_fit_size) {
219                  best_fit_chunk = c;
220                  best_fit_block = i;
221                  best_fit_size = block->size;
222              }
223          }
224      }
225
226      if (best_fit_block == -1) {
227          // no suitable block found, try the last block (this may grow a chunks size)
228          int64_t best_reuse = INT64_MIN;
229          for (int c = 0; c < alloc->n_chunks; ++c) {
230              struct tallocr_chunk * chunk = alloc->chunks[c];
231              if (chunk->n_free_blocks > 0) {
232                  struct free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
233                  max_avail = MAX(max_avail, block->size);
234                  int64_t reuse_factor = chunk->max_size - block->offset - size;
235                  // reuse_factor < 0 : amount of extra memory that needs to be allocated
236                  // reuse_factor = 0 : allocated free space exactly matches tensor size
237                  // reuse_factor > 0 : superfluous memory that will remain unused
238                  bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
239                  bool better_fit   = reuse_factor >= 0 && reuse_factor < best_reuse;
240                  if (block->size >= size && (better_reuse || better_fit)) {
241                      best_fit_chunk = c;
242                      best_fit_block = chunk->n_free_blocks - 1;
243                      best_reuse = reuse_factor;
244                  }
245              }
246          }
247      }
248
249      if (best_fit_block == -1) {
250          // none of the existing chunks have enough space left
251          best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
252          best_fit_block = 0;
253      }
...
261      struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
262      struct free_block    * block = &chunk->free_blocks[best_fit_block];
263      struct buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
264      block->offset += size;
265      block->size -= size;
266      if (block->size == 0) {
267          ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
268      }
269
270      return addr;
271  }
```

#### 4.7.1 三段式搜索

| 阶段 | 行 | 策略 | 什么时候命中 |
|---|---|---|---|
| ① best-fit | 212-224 | 在所有 chunk 的**非末尾**空闲块里找最小的够用块 | 有碎片可用 |
| ② 末尾块 | 226-247 | 用末尾块，允许扩容 | 没有合适的碎片，但末尾还有空间 |
| ③ 新 chunk | 249-253 | 创建新 chunk | 所有 chunk 都满了 |

**为什么 ① 排除末尾块（`n_free_blocks - 1`）**：末尾块是“可以增长”的块，如果 ① 用了它，会破坏“末尾块永远可增长”的假设。所以 ① 只在“固定大小的碎片”里找。

**② 的 `reuse_factor`**：

```text
reuse_factor = chunk->max_size - block->offset - size
```

- `< 0`：需要扩容 `-reuse_factor` 字节；
- `= 0`：正好填满；
- `> 0`：填完后还剩 `reuse_factor` 字节。

**238-239 的择优逻辑**：

```cpp
bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;   // 两个都需要扩容，选扩容少的
bool better_fit   = reuse_factor >= 0 && reuse_factor < best_reuse; // 两个都能装下，选剩余少的
```

#### 4.7.2 断点对照

| 日志 | 命中阶段 | 说明 |
|---|---|---|
| `ALLOC node_2 off=0 block_left=SIZE_MAX` | ③ 新 chunk | 第一次分配，chunk 刚创建，`max_size=SIZE_MAX` |
| `ALLOC node_5 off=1000064` | ② 末尾块 | 末尾块 offset=1,000,064，直接追加 |
| `ALLOC node_16 off=2020224` | ① best-fit | 复用了 `grad for loss` 刚释放的 128 B 块 |
| `ALLOC grad for node_5 off=1000064` | ① best-fit | 复用了 `node_5` 释放的 1,000,064 B 块 |
| `ALLOC node_24 off=1000064` | ① best-fit | 复用了 `grad for node_5` 释放的块（此时只剩 2,048 B 可用） |

**`block_left=SIZE_MAX` 的含义**：新 chunk 的初始 `free_blocks[0].size = MAX(min_size, max_chunk_size)`，`max_chunk_size = MIN(max_buffer_size, SIZE_MAX/2) = SIZE_MAX/2`（CUDA buft 的 `get_max_size` 返回 `SIZE_MAX`）。断点里打印的是 `SIZE_MAX`（因为 gdb 显示的是 `block->size`，而 `SIZE_MAX/2` 在加法后溢出回绕），实际是 `SIZE_MAX/2`。

---

## 5 训练中的 realloc 现象（重点）

### 5.1 现象

训练日志里每个 epoch 出现 **115 次**：

```text
ggml_gallocr_needs_realloc: graph has different number of nodes
ggml_gallocr_alloc_graph: cannot reallocate multi buffer graph automatically, call reserve
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n_impl: reallocating CUDA0 buffer from size 3.42 MiB to 3.42 MiB
```

30 epoch 累计 **3,450 次**。

### 5.2 统计

从训练日志里按行计数：

```text
"failed to allocate graph"                    3,450 次
"backend_ids_changed = 1"                     3,420 次
"backend_ids_changed = 0"                        30 次
"graph has different number of nodes"            30 次
"reallocating CUDA0 buffer"                       1 次
```

三个数字的关系是理解本节的关键：

```text
3,450 = 3,420（backend_ids_changed 触发）+ 30（needs_realloc 触发）
```

### 5.3 根因一：`backend_ids_changed`（3,420 次）

`ggml_backend_sched_alloc_splits` 的第一道判断：

```cpp
bool backend_ids_changed = false;
for (int i = 0; i < sched->graph.n_nodes; i++) {
    if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
        sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
        backend_ids_changed = true;
        break;
    }
}
if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
    // 强制 reserve
}
```

**注意 `||` 短路**：只要 `backend_ids_changed` 为 true，`ggml_gallocr_alloc_graph` 根本不会被调用，`needs_realloc` 的日志也不会打印。这解释了“3,420 次失败但只有 30 条 needs_realloc 日志”。

**到底是哪个索引不同？** 断点逐项比对：

```gdb
set $i = 0
while $i < sched->graph.n_nodes
if sched->node_backend_ids[$i] != sched->prev_node_backend_ids[$i]
printf "diff at idx=%d new=%d prev=%d
", $i, sched->node_backend_ids[$i], sched->prev_node_backend_ids[$i]
end
set $i = $i + 1
end
```

```text
BATCH#1 nodes=28 leafs=7  diff_kind=0 idx=-1                          ← 完全一致
BATCH#2 nodes=34 leafs=16 diff_kind=1 idx=0  new=2 prev=0             ← gb_opt
BATCH#3 nodes=28 leafs=7  diff_kind=1 idx=0  new=0 prev=2             ← gb_grad
BATCH#4 nodes=34 leafs=16 diff_kind=1 idx=0  new=2 prev=0             ← gb_opt
...
```

**差异在 `idx=0`，不是“数组长度不匹配”。**

- `gb_grad` 的 `nodes[0]` 是 `fc2.weight`，backend_id = **0**（CUDA0）。
- `gb_opt` 的 `graph_copy[0]` 是 `adamw_params (view)`（scheduler 插入的依赖节点），backend_id = **2**（CPU）。

回顾 02 §2.2 的节点表：`gb_opt` 的 `graph_copy` 前面多插了 2 个节点：

```text
graph_copy[0] = adamw_params (view)       backend = CPU   ← idx 0 差异的来源
graph_copy[1] = CUDA0#adamw_params#0      backend = CUDA0
graph_copy[2..] = 原 gb_opt 的 32 个节点
```

所以 `node_backend_ids[0]` 在 `0` 和 `2` 之间来回跳，且 `bufts[0]=CUDA0 != bufts[2]=CPU`，两个条件都满足 → `backend_ids_changed = true`。

**这是设计上的必然结果**：只要图里有跨后端输入，scheduler 就会在 `graph_copy` 前面插 CPU 节点，而 CPU 节点的 backend_id 与纯 GPU 图的首节点不同。**`backend_ids_changed` 的语义是“后端归属真的变了”，不是“图的大小变了”。**

**为什么 batch #1 是 `diff_kind=0`？** 因为 `prev_node_backend_ids` 是 `calloc` 出来的全 0，而 `gb_grad` 的 28 个节点也全是 0（CUDA0），所以完全一致。这时才会走到 `ggml_gallocr_alloc_graph`，命中 `needs_realloc`。

### 5.4 根因二：`needs_realloc`（30 次）

每个 epoch 第一次 `ggml_opt_alloc` 时，`backend_ids_changed` 恰好为 false（见上），于是走到 `ggml_gallocr_alloc_graph`，命中 `needs_realloc`：

```text
NEEDS_REALLOC galloc_nodes=0 graph_nodes=28 galloc_leafs=0 graph_leafs=7     ← epoch 1
NEEDS_REALLOC galloc_nodes=13 graph_nodes=28 galloc_leafs=2 graph_leafs=7    ← epoch 2+
ggml_gallocr_needs_realloc: graph has different number of nodes
```

`galloc_nodes=13` 是上一个 epoch 最后一张验证图 `gf` 的节点数。30 epoch × 1 次 = 30 条日志。

**注意这 30 次里，只有第 1 次真正重新 `cudaMalloc`**（`gb_grad` 的 3,588,224 B 是首次分配），第 2 次因为 `gb_opt` 需要 3,588,352 B 又分配了一次，之后都复用。

### 5.5 两张图的交替序列

```text
opt_period = 1000 / 500 = 2

物理 batch #1: ggml_opt_alloc(backward=true) → opt_i=0 → build_type=GRAD → gb_grad (28 节点)
物理 batch #2: ggml_opt_alloc(backward=true) → opt_i=1 → build_type=OPT  → gb_opt  (32 节点 + 2 input = 34)
物理 batch #3: gb_grad (28)
物理 batch #4: gb_opt  (34)
...
```

`ggml_gallocr` 只记住**上一次**的 `n_nodes`：

```cpp
static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
        return true;
    }
    if (galloc->n_leafs != graph->n_leafs) { ... return true; }
    for (int i = 0; i < graph->n_nodes; i++) { ... }
    return false;
}
```

所以 `28 → 34 → 28 → 34` 每次都触发 `needs_realloc`——**但只在 `backend_ids_changed` 为 false 时才会走到这里**。实际日志里 3,420 次是 `backend_ids_changed=1`（短路），只有 30 次是 `needs_realloc`（每个 epoch 第一次）。

### 5.6 为什么没有真的重新 cudaMalloc

`ggml_gallocr_reserve_n_impl` 只在**新的 chunk 大小超过已有 buffer 大小**时才重新分配：

```cpp
for (int i = 0; i < galloc->n_buffers; i++) {
    size_t new_size = 0;
    for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
        size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
        size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
        new_size += new_chunk_size;
        if (new_chunk_size > cur_chunk_size) realloc = true;
    }
    if (realloc) { ... ggml_vbuffer_free(...); ggml_vbuffer_alloc(...); }
}
```

`gb_grad` 需要 3,588,224 B，`gb_opt` 需要 3,588,352 B，后者大 128 B。第一次 reserve 时（`gb_grad`）分配 3,588,224；第二次（`gb_opt`）发现需要 3,588,352，于是：

```text
ggml_gallocr_reserve_n_impl: reallocating CUDA0 buffer from size 3.42 MiB to 3.42 MiB
```

**只发生 1 次**。之后两种图都落在 3,588,352 B 的 buffer 里，`new_size <= cur_size`，不再重新分配。

断点验证：

```text
CUDA_ALLOC_BUFFER size=3158272 (count=1)   ← 静态 buffer
CUDA_ALLOC_BUFFER size=4795520 (count=2)   ← 静态 buffer
RESERVE#1 nodes=28 leafs=7
CUDA_ALLOC_BUFFER size=3588224 (count=3)   ← gb_grad
RESERVE#2 nodes=34 leafs=16
CUDA_ALLOC_BUFFER size=3588352 (count=4)   ← gb_opt（+128 B）
RESERVE#3 nodes=28 leafs=7
RESERVE#4 nodes=34 leafs=16
RESERVE#5 nodes=28 leafs=7
RESERVE#6 nodes=34 leafs=16
TOTAL cuda_buffer_allocs=4 reserves=20 after 20 batches
```

### 5.7 代价：每 batch 一次 reserve（实测 42 µs）

即使不重新 `cudaMalloc`，`reserve_n_impl` 仍然要：

```text
① 重置所有 tallocr（n_chunks = 0）
② 重新遍历 graph，做一遍完整的生命周期分析（28/34 个节点）
③ 重新写 node_allocs[] / leaf_allocs[]
④ 检查 buffer 大小，不满足才 realloc
⑤ 调用 ggml_gallocr_alloc_graph，把 offset 写回 tensor->data
```

**真实开销必须用插桩测量，不能用 gdb 计时。**

早期用 gdb 断点测得的 `alloc=4.5 ms` 是**假象**：每个断点命中本身要几十到几百微秒（gdb 要停线程、解析符号、执行 `printf`），3,400 次断点累加后完全淹没了真实耗时。

改用**代码插桩**（在 `ggml_backend_sched_split_graph` / `alloc_splits` 首尾各打一次 `ggml_time_us()`，每 200 次打印一次累计值），连跑 3 次的结果：

```text
[PROF] n=3400 split_total= 57695 us alloc_total=144328 us alloc_avg=42 us alloc_max=430 us
[PROF] n=3400 split_total= 53463 us alloc_total=144303 us alloc_avg=42 us alloc_max=407 us
[PROF] n=3400 split_total= 52208 us alloc_total=142901 us alloc_avg=42 us alloc_max=425 us
```

换算成每 batch：

```text
split_graph   ≈ 15 µs
alloc_splits  ≈ 42 µs
合计          ≈ 57 µs / batch
```

3,400 个 batch × 57 µs ≈ **0.19 s**，占 6 s 总时间的 **3%**。**不是瓶颈。**

**为什么 `alloc_max=430 µs`？** 那是 epoch 边界第一次遇到 `gb_opt`（34 节点）时的 reserve，之后稳定在 42 µs。所以“每 batch 重新 reserve”确实发生，但单次成本只有几十微秒——因为：

- `ggml_gallocr_alloc_graph_impl` 是 O(V+E) 的线性遍历，34 个节点约 1 µs；
- `needs_realloc` 的逐节点检查是 O(V)，同样很快；
- 真正贵的 `cudaMalloc` 只发生 1 次（见 §5.4）。

### 5.8 真正的每 batch 成本构成

同一份插桩在 `ggml_opt_epoch` 的训练循环里测三段：

```cpp
t_b0; ggml_opt_alloc(opt_ctx, true);        // 选图 + split + alloc
t_b1; ggml_opt_dataset_get_batch(...);      // 100 次 H2D
t_b2; ggml_opt_eval(opt_ctx, result_train); // compute + sync + D2H
t_b3;
```

```text
[PROF2] n=3000 alloc_avg=158 h2d_avg=1141 comp_avg=496 alloc_max=716 us
[PROF2] n=3000 alloc_avg=159 h2d_avg=1114 comp_avg=493 alloc_max=683 us
[PROF2] n=3000 alloc_avg=159 h2d_avg=1138 comp_avg=489 alloc_max=2741 us
```

| 阶段 | 每 batch | 占比 |
|---|---|---|
| `ggml_opt_alloc`（含 split + alloc） | 158 µs | 9% |
| `dataset_get_batch`（100 次 H2D + sync） | **1,138 µs** | **64%** |
| `ggml_opt_eval`（kernel + D2H） | 490 µs | 27% |

合计 1,786 µs/batch × 3,420 batch ≈ **6.1 s**，与实测训练总时长 6 s 吻合。

**结论反转**：mnist 的真正瓶颈是**数据搬运**，不是显存规划。`ggml_opt_alloc` 的 158 µs 里，split 15 µs + alloc 42 µs，剩下约 100 µs 是 `ggml_opt_build` 里的图切换和 `ggml_graph_reset`。

这也解释了为什么源码注释强调 shard 大小：

```cpp
// The MNIST model is so small that the overhead from data shuffling is non-negligible, especially with CUDA.
// With a shard size of 10 this overhead is greatly reduced at the cost of less shuffling
```

`dataset_get_batch` 的 1,138 µs 里，100 次 `cudaMemcpyAsync + cudaStreamSynchronize` 占大头（微基准：31,360 B 带 sync 10.47 µs，400 B 带 sync 5.41 µs，加权 7.94 µs × 100 = 794 µs），剩余约 340 µs 是 **permutation 随机访问 190 MB 数据集导致的 cache miss**。

### 5.9 值不值得优化

**先给结论：对 mnist 不值得。**

实测 `alloc_splits` 每 batch 42 µs，3,400 个 batch 合计 144 ms，占 6 s 训练的 **2.4%**。而 `dataset_get_batch` 每 batch 1,138 µs，占 **64%**。

所以“每 batch 重新 reserve”在 mnist 上是一个**可观测但不重要**的现象。真正值得优化的是数据搬运（见 06 §5）。

**什么时候它才会变成瓶颈？**

```text
单 batch 计算时间 << alloc 时间   → alloc 占比高
```

- mnist：计算 490 µs，alloc 42 µs → 8.6%，可以忽略。
- 真实 LLM 的 decode（单 token，几十层）：计算可能 5~20 ms，alloc 如果按图的规模线性增长（几千节点），可能到 1~2 ms → 10% 左右，值得优化。
- 真实 LLM 的 prefill（长序列）：计算 100+ ms，alloc 完全被淹没。

**所以优化优先级是**：

```text
① 数据搬运（mnist 上占 64%）
② 冷启动（首次运行 22 s vs 稳态 6 s）
③ 显存规划（占 2.4%）
```

### 5.10 如果真的要做（三个方向）

**① 让 `gb_grad` 和 `gb_opt` 的节点数一致**

让 `gb_opt` 的 4 个 AdamW 节点在非 opt batch 也被包含，但通过 `flags` 跳过。这样两张图都是 34 节点，`needs_realloc` 直接返回 false，省掉整个 `reserve_n_impl`。

```cpp
// ggml_opt_alloc 里
graph = opt_ctx->gb_opt;                     // 总是用最大图
for (auto * n : adamw_nodes) {
    if (build_type == GRAD) n->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    else                    n->flags |=  GGML_TENSOR_FLAG_COMPUTE;
}
```

风险：`ggml_graph_reset` 和 gallocr 都读 flags，需要验证不会误释放参数。

**② 改造 gallocr，缓存多张图的规划结果**

```cpp
struct ggml_gallocr {
    // 当前：只有一份 node_allocs[]
    // 改进：按"图指纹"缓存
    struct {
        uint64_t fingerprint;
        struct node_alloc * node_allocs;
        struct leaf_alloc * leaf_allocs;
        int n_nodes, n_leafs;
    } cache[2];
};
```

指纹 = `hash(n_nodes, n_leafs, 每个节点的 op/ne/type/flags)`。训练只有 2 种图，2 个槽够用。

**注意**：不能用 `graph->uid` 做 key，因为 `split_graph` 每次都分配新 uid：

```cpp
graph->uid = ggml_graph_next_uid();
```

**③ 用 `ggml_backend_sched_reserve` 预留**

```cpp
ggml_backend_sched_reserve(sched, gb_opt);   // 按最大图预留
```

**这个方案单独用不能消除 realloc**：`needs_realloc` 仍然会因为 `n_nodes` 不同返回 true。它只能保证 buffer 不重新 `cudaMalloc`（而这一点当前实现已经做到了，因为 `gb_opt` 只比 `gb_grad` 大 128 B）。

**①②③ 的对比**：

| 方案 | 改动量 | 收益 | 风险 |
|---|---|---|---|
| ① 对齐节点数 | 改 `ggml-opt.cpp` | 省 42 µs/batch | 中（flags 交互） |
| ② 多图缓存 | 改 `ggml-alloc.c` | 省 42 µs/batch | 高（核心分配器） |
| ③ 显式 reserve | 改调用方 | 几乎为 0 | 低 |

**真正的教训**：不要看到 `needs_realloc` 的日志就假设它是瓶颈。**先用插桩量化，再决定是否优化**。本文最初用 gdb 断点计时得出“4.5 ms/batch、占总时间一半”的结论，后来用代码插桩推翻为“42 µs/batch、占 2.4%”——这是本次调试最重要的一次自我纠错。

---

## 6 推理路径的分配

### 6.1 断点

```text
EVAL n_splits=1 split0=[0,13) n_inputs=0 backend=CUDA0
  allocgraph[0]=fc2.weight op=0
  allocgraph[1]=fc1.weight op=0
  allocgraph[2]=node_2 op=29
  allocgraph[3]=fc1.bias op=0
  allocgraph[4]=node_4 op=2
  allocgraph[5]=node_5 op=91
  allocgraph[6]=node_6 op=29
  allocgraph[7]=fc2.bias op=0
  allocgraph[8]=logits op=2
  allocgraph[9]=loss_cross_entropy op=96
  allocgraph[10]=pred op=18
  allocgraph[11]=node_11 op=18
  allocgraph[12]=ncorrect op=19
```

### 6.2 推理的原地复用更激进

```text
[2] node_2   data=0x7fbdf1000000
[4] node_4   data=0x7fbdf1000000   ← ADD 原地
[5] node_5   data=0x7fbdf1000000   ← RELU 原地
[6] node_6   data=0x7fbdf10f4280
[8] logits   data=0x7fbdf10f4280   ← ADD 原地
```

**三个节点共用同一块 1 MB 显存**（`node_2 → node_4 → node_5`），因为推理没有反向，`node_2` 只有 `node_4` 一个消费者，`node_4` 只有 `node_5` 一个消费者。

对比训练：`node_4` 有 2 个消费者（`node_5` 前向 + `node_22` 反向），所以 `node_2 → node_4` 可以原地，但 `node_4 → node_5` 不行（`node_4` 还要留给反向用）。

### 6.3 推理的 compute buffer

```text
GALLOC bufs=3
  buft=CUDA0 vbuf_size=1020160 chunk0=1020160 usage=2
  buft=CUDA0 vbuf_size=1020160 chunk0=1020160 usage=2
  buft=CPU   vbuf_size=0
```

**1,020,160 B**，约为训练的 1/3.5：

```text
node_2/node_4/node_5 共用 1,000,064
node_6/logits 共用      20,096
                      = 1,020,160
```

`loss_cross_entropy`、`pred`、`node_11`、`ncorrect` 不在 compute buffer 里——它们在 `opt ctx_static`（推理时是 `mnist_model_init_from_file` 创建的 `ctx_static`），是静态分配。

---

## 7 本篇结论

1. **split 是“能力 + 优先级”决策**：已有 buffer → 权重所在后端 → GPU 扩展 → 能力兜底。本例全 CUDA，`n_splits=1`。
2. **`graph_copy` 是 scheduler 的私有图**：多 2 个节点用于跨后端输入的依赖和拷贝。
3. **gallocr 是单遍顺序分析**：遍历 `nodes[]`，分配输出、减少 src 计数、归零即释放。要求 `nodes[]` 是拓扑序。
4. **复用有两种**：原地复用（`op_can_inplace` + 唯一消费者）和 free list 复用（best-fit）。本例两者都有。
5. **`n_children` / `n_views` 是生命周期分析的全部状态**，不用引用计数，不用 GC。
6. **多图交替导致每 batch 重新 reserve**：3,450 次 reserve，但只有 1 次真正 `cudaMalloc`，每 batch 成本 42 µs，占总时间 2.4%——**不是瓶颈**。
7. **推理比训练复用更激进**：没有反向，`node_2 → node_4 → node_5` 三个节点共用同一块 1,000,064 B 的显存。
8. **真正的每 batch 成本**：数据搬运 1,138 µs（64%）> 计算 490 µs（27%）> 调度 158 µs（9%）。
9. **gdb 断点计时会严重失真**：断点命中本身的开销（几十到几百 µs）会淹没真实耗时，必须用代码插桩或微基准。
