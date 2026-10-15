```
# GGML 核心原理与推理流程简介
> 文档基于 llama.cpp / ggml 后端推理模型开发会话整理，聚焦 `ggml_context`、`ggml_cgraph`、张量内存分配、张量生命周期、后端Buffer分配等底层机制，面向开发人员。

## 一、核心概念总览
GGML 是轻量级机器学习张量库，是 llama.cpp 的底层基础。核心思想：**张量元数据（host CPU侧描述信息）和张量真实数据（设备内存/显存）分离管理**。
- `ggml_tensor`：张量描述结构体，记录形状、数据类型、源张量（view）等元信息；本身**不自带大数据缓冲区**。
- `ggml_context`：Host侧内存池，用来分配一批 `ggml_tensor`、`ggml_cgraph`、算子节点等元数据结构体。
- `ggml_backend`：后端抽象层（CPU/CUDA/Metal/Vulkan），负责设备内存分配、算子执行。
- `ggml_backend_buffer`：后端设备上一块连续内存（显存/设备内存），存放张量真实数值数据。
- `ggml_cgraph`：计算图，DAG有向无环图，保存算子节点、叶子张量，描述计算依赖关系。
- `ggml_gallocr`：Graph Allocator，**计算图专用内存分配器**，静态分析张量生命周期，实现中间张量内存重叠复用，降低推理显存占用。

> 两种完全独立内存：
> 1. Host内存：`ggml_context` 内存池，只存结构体、指针等元数据，占用很小；
> 2. 设备内存（显存）：`ggml_backend_buffer`，存放权重、激活张量真实数据，是内存占用大头。

## 二、两种 ggml_context：ctx（权重上下文） & ctx_cgraph（计算图上下文）
### 1. `ggml_context * ctx` —— 权重上下文
用途：存放模型权重张量元数据（W、bias等静态权重）。
**创建参数必须设置 `no_alloc = true`**
- `no_alloc=true`：代表当前 ctx 只分配张量结构体（元数据），**不分配张量的数据内存**；权重张量真实数据交给后端buffer统一分配。
- 内存池：只存放权重 `ggml_tensor` 结构体。
- 内存分配API：`ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);`
  - 自动遍历ctx内全部张量，跳过view视图张量；
  - 按后端对齐粒度，累加所有张量字节大小；
  - 在后端一次性分配一块连续设备buffer；
  - 自动为每个权重张量设置 `tensor->data`，指向buffer内对应偏移。
- 生命周期：模型加载时创建，全程复用，模型卸载时 `ggml_free(ctx)`。
> 权重张量属于叶子张量，生命周期贯穿整个推理会话，**不参与显存重叠复用**。

### 2. `ggml_context * ctx_cgraph` —— 计算图上下文
用途：专门用来构建推理计算图，存放**中间张量元数据、算子节点、ggml_cgraph对象**。
- `no_alloc=false`（默认），host内存池用来分配：`ggml_tensor`（中间激活张量）、`ggml_cgraph`结构体、节点数组。
- 注意：**ctx_cgraph host内存池只分配元数据，不分配中间张量的设备显存**。中间张量数据内存交给 `ggml_gallocr`。
- 生命周期：推理循环内可复用/重建；可单独释放，不会影响权重ctx。

> 关联关系：
> 算子构建时，可**跨context引用权重张量**：算子在ctx_cgraph创建，输入权重张量来自ctx。只是引用权重张量指针，**不会拷贝权重张量**。张量结构体归属各自context，但张量`data`指针可以指向同一块后端内存。

> ❌ 禁止在权重ctx里面构建计算图算子：会持续往权重ctx内存池堆积大量中间tensor元数据，无法单独回收。

## 三、完整推理时序（标准GGML后端推理流程）
```c
// 1. 创建权重ctx，no_alloc=true，创建权重张量元数据
ggml_init_params params_model = {.mem_size = ..., .no_alloc = true};
ggml_context * ctx = ggml_init(params_model);
ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, ...);
// 一次性分配所有权重张量的设备内存
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

// 2. 创建计算图上下文 ctx_cgraph，host内存池用于图元数据
ggml_init_params params_graph = {.mem_size = ..., .no_alloc = false};
ggml_context * ctx_cgraph = ggml_init(params_graph);

// 3. 在 ctx_cgraph 内创建计算图对象 gf
ggml_cgraph * gf = ggml_new_graph(ctx_cgraph); 
// 👉 分配：ggml_cgraph结构体、nodes/leafs指针数组（host侧，来自ctx_cgraph内存池，无显存分配）

// 4. 在ctx_cgraph构建算子，生成中间张量元数据
ggml_tensor * x = ggml_new_tensor_2d(ctx_cgraph, GGML_TYPE_F16, ...);
ggml_tensor * out = ggml_mul_mat(ctx_cgraph, w, x); // w来自权重ctx，只引用

// 5. 展开计算图，构建DAG依赖
ggml_build_forward_expand(gf, out);
// 👉 仅填充gf->nodes、gf->leafs指针，**不分配任何设备内存**

// 6. 创建图内存分配器，基于后端buffer类型
ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

// 7. 【核心】静态生命周期分析 + 分配中间张量设备显存
ggml_gallocr_alloc_graph(allocr, gf);
/*
内部执行逻辑：
1. 遍历整张DAG，统计每个张量引用次数use_counts；
2. 拓扑模拟执行，计算每个张量生命周期区间 [start_node, last_node]；
   start_node：张量在哪一个算子节点生成；
   last_node：张量最后一次被消费的节点；
3. 生命周期不重叠张量，可以复用同一块设备内存；
4. 计算峰值内存，后端分配连续设备buffer；
5. 为图内中间张量、输入叶子张量设置 tensor->data，指向buffer偏移；
⚠️ 权重张量跳过，不处理权重内存
*/

// 8. 后端执行整张计算图
ggml_backend_graph_compute(backend, gf);

// 9. 释放资源
ggml_gallocr_free(allocr); // 释放gallocr管理的设备buffer
ggml_free(ctx_cgraph);     // 释放计算图host元数据内存
ggml_free(ctx);            // 释放权重ctx元数据
ggml_backend_buffer_free(buffer); // 释放权重设备buffer
```

## 四、`ggml_gallocr_alloc_graph` 详解

函数原型：`bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);`

> 
> 作用：**分配计算图内所有中间张量、输入叶子张量的设备内存（显存）**。
> 不做：不分配张量元数据、不分配 ggml_cgraph 结构体，不处理权重张量。

核心特性：

1. **静态生命周期分析**：在图构建完成后一次性分析，属于编译期静态规划，不是运行时动态 GC；
2. **内存重叠复用**：张量生命周期区间不重叠，就可以复用同一块显存偏移，大幅降低推理峰值显存；
3. View 张量（切片、转置视图）：不单独分配内存，复用源张量 buffer 偏移；
4. 多次调用：图拓扑不变时，复用已有 buffer；图拓扑变化会重新分析、重新分配。

## 五、张量生命周期查看方法

> 
> 重要：`ggml_tensor` 结构体**没有内置字段存储 start_node /last_node**。生命周期信息仅存在`ggml_gallocr_alloc_graph`内部临时数组，函数执行完即销毁。

### 方法 1：ggml_graph_print (gf) （快速查看）

```
ggml_graph_print(gf);
```

打印所有算子节点、张量依赖关系。人工查看张量在哪一个节点产生，在哪几个节点被消费，手动判断最后使用节点。
缺点：不会直接输出 start_node /last_node。

### 方法 2：导出 DOT 图可视化

```
ggml_graph_dump_dot(gf, NULL, "graph.dot");
```

使用 graphviz 渲染：`dot -Tpng graph.dot -o graph.png`
可视化 DAG，直观看到张量依赖链路，判断张量被哪些后续算子使用。

### 方法 3：源码插桩（最精准）

修改 `ggml-alloc.c` 的 `ggml_gallocr_alloc_graph_impl`，在生命周期分析循环内增加打印：输出张量指针、shape、start_node、last_node。

### 方法 4：查看张量 data 偏移，反向判断复用关系

gallocr 分配完成后，遍历图节点张量，打印 `tensor->data` 在 gallocr buffer 内偏移：

- 两个张量`data`偏移完全相同：生命周期无重叠，显存复用；
- 偏移不同：生命周期重叠，必须占用独立内存区域。

### 方法 5：利用 cgraph->use_counts

`ggml_cgraph` 自带 `use_counts` 数组，保存每个张量**总引用次数**（静态统计值，不是运行时剩余引用计数）。

#### 三类张量生命周期总结

1. 权重张量（ctx 叶子张量）：生命周期贯穿全部推理节点，全程占用内存，**不参与内存复用**；
2. 中间激活张量（算子输出）：生命周期区间 `[start_node, last_node]`；last_node 执行完成后内存可复用；
3. View 视图张量：不分配独立内存，生命周期跟随源张量。

## 六、关键 API 对比汇总

表格

| API | 作用 | 分配内存位置 | 分配对象 | 是否内存复用 |
| --- | --- | --- | --- | --- |
| `ggml_backend_alloc_ctx_tensors(ctx, backend)` | 一次性分配 ctx 内张量设备内存 | 后端设备内存 | ctx 所有权重张量 | ❌ 静态一次性分配，不可复用 |
| `ggml_new_graph(ctx_cgraph)` | 创建计算图对象 gf | Host 内存池（ctx_cgraph） | ggml_cgraph 结构体、nodes/leafs 指针数组 | — host 元数据 |
| `ggml_gallocr_alloc_graph(allocr, gf)` | 生命周期分析，分配中间张量设备内存 | 后端设备内存 | 计算图中间张量、输入叶子张量 | ✅ 静态分析，支持内存重叠复用 |

## 七、常见踩坑清单

1. `ctx_cgraph` 的 `mem_size` 不足：`ggml_new_graph` / `ggml_new_tensor` 断言失败，属于 host 元数据内存不足，**不是显存不足**；
2. 不要在权重 ctx 中构建算子：会持续累积中间张量元数据，无法单独释放；
3. `ggml_gallocr_alloc_graph` 只规划内存、设置 data 指针，**不会拷贝张量数值**；数据拷贝需要手动调用`ggml_backend_tensor_set`；
4. 生命周期是**静态图分析结果**，图拓扑一旦变化，需要重新调用`ggml_gallocr_alloc_graph`重新分析和分配；
5. view 张量不会单独分配内存，只是引用源张量内存偏移。

## 八、参考资料

- HuggingFace Blog：Introduction to GGML [https://huggingface.co/blog/introduction-to-ggml](https://huggingface.co/blog/introduction-to-ggml)
- llama.cpp ggml 源码库：ggml.h、ggml-alloc.c、ggml-backend.c