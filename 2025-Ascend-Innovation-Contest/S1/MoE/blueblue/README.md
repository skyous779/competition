# MindNLP 模型优化详细说明 (DeepSeek & Qwen2-MoE)

本文档详细记录了针对 DeepSeek 和 Qwen2-MoE 模型的关键性能优化点，并附带了相应的核心代码实现。

## 1. DeepSeek 模型优化 (DeepseekMoE)

### 1.1 MoE 推理加速：Decode 阶段 (消除 Host-Device 同步)

优化痛点: 原有实现在循环中频繁调用 .item() 获取 scalar 值，导致每一次循环都会触发 Host 与 Device 之间的同步，严重拖慢小 Batch/单 Token 的推理速度。

改进方案: 一次性将所有 Expert 的索引和权重同步至 CPU (.tolist())，循环全程在 Host 端执行，无同步开销。

**源码实现** (`moe_infer_decode`):

**Python**

```
def moe_infer_decode(self, x, flat_expert_indices, flat_expert_weights):
    """
    优化后的 'decode' 模式推理 (seq_len=1)。
    优化点: 在循环外使用 .tolist() 替换循环内的 .item()，避免多次 Host-Device 同步。
    """
    expert_cache = ops.zeros_like(x)
  
    # 1. 一次性将 IDs 和 weights 同步到 CPU
    # flat_expert_indices 形状 (num_experts_per_tok,)
    # flat_expert_weights 形状 (num_experts_per_tok, 1)
    expert_ids_list = flat_expert_indices.tolist()
    weights_list = flat_expert_weights.view(-1).tolist() # .view(-1) 压缩形状

    # 2. 循环现在完全在 Host 上运行，没有同步
    for i in range(self.num_experts_per_tok):
        expert_id = expert_ids_list[i]
        weight = weights_list[i]
    
        expert = self.experts[expert_id]
        expert_out = expert(x)
        expert_cache += expert_out * weight
    return expert_cache
```

### 1.2 MoE 推理加速：Prefill 阶段 (循环控制上移)

优化痛点: 在处理长序列时，MoE 层的循环控制逻辑（计算每个 Expert 分配到的 token 数量）如果依赖 Tensor 操作，会导致图执行中断或多次同步。

改进方案: 将 bincount 和 cumsum 的结果转为 numpy/list，在 CPU 上计算切片索引 (start_idx, end_idx)，仅将必要的计算算子下发 Device。

**源码实现** (`moe_infer`):

**Python**

```
# 准备: 排序和计算每个expert的token数量
expert_cache = ops.zeros_like(x)
idxs = flat_expert_indices.argsort()
tokens_per_expert_cumsum = flat_expert_indices.bincount().cumsum(0)

# -------- 优化点: 将循环控制数据移至 CPU --------
# 一次性将累积和数组同步到CPU，避免在循环中逐个同步
tokens_per_expert_cumsum_cpu = tokens_per_expert_cumsum.asnumpy()
tokens_per_expert_list = tokens_per_expert_cumsum_cpu.tolist()

# 在CPU上预先计算好start_idx
start_indices_list = [0] * len(tokens_per_expert_list)
if len(tokens_per_expert_list) > 1:
    start_indices_list[1:] = tokens_per_expert_cumsum_cpu[:-1].tolist()

token_idxs = idxs // self.num_experts_per_tok

# 迭代: 在CPU上控制循环，在Device上执行计算
for i, end_idx in enumerate(tokens_per_expert_list):
    start_idx = start_indices_list[i]
    if start_idx == end_idx:
        continue
    # ... (后续 Gather-Compute-Scatter 逻辑)
```

## 2. Qwen2-MoE 模型优化

### 2.1 MoE 路由逻辑优化 (索引计算)

优化痛点: 原始实现可能使用了低效的循环或不兼容动态图的索引方式。

改进方案: 利用 mint.nonzero 获取稀疏索引，并优化索引加法逻辑。

**源码实现** (`Qwen2MoeSparseMoeBlock`):

**Python**

```
# 获取非零专家的索引 [row_idx, expert_idx, top_k_idx]
non_zero_df = mint.nonzero(expert_mask).asnumpy()
if non_zero_df.shape[0] > 0:
    expert_idx = non_zero_df[0][0]
    # ... (分组处理相同 expert 的 token) ...
    for i in range(1, non_zero_df.shape[0]):
        # 逻辑：将属于同一个 Expert 的 token 攒在一起处理
        if non_zero_df[i][0] == expert_idx:
            k += 1
        else:
            # 执行计算
            expert_layer = self.experts[expert_idx]
            # 使用 Tensor 索引
            idx = mindspore.Tensor(non_zero_df[j:j + k, 1], mindspore.int32)
            top_x = mindspore.Tensor(non_zero_df[j:j + k, 2], mindspore.int32)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        
            # 加权并累加回主干
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states.to(hidden_states.dtype))
        
            # 重置计数器
            expert_idx = non_zero_df[i][0]
            j = i
            k = 1
```

## 最终收益
| model_name | memory_reserved | memory_allocated | avg_prefill_latency | avg_decode_latency |
| :--- | :--- | :--- | :--- | :--- |
| Qwen1.5-MoE-A2.7B-Chat | 31.138512896 | 29.234176512 | 1.3811829090118408 | 0.1579372354156434 |
| deepseek-moe-16b-chat | 34.359738368 | 32.813018112 | 2.4653173287709556 | 0.1202096897995133 |


## 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 100 |
| Prefill时延得分 | 142.3168    |
| Decode时延得分 | 427.4573     |
| **总分** | **223.258** |