<!-- zhiye:{"syncMode":"full","title":"第 18 题：Group Query Attention（GQA）与 Multi-Query Attention（MQA）","tags":["AI","面试","导入资料","基础整合"],"visibility":"public","folderPath":["外部资料","Agent 面试 300 题","01-LLM基础"],"pinned":false,"source":{"url":"https://github.com/RiddleGo/agent-interview-300","branch":"main","path":"01-LLM基础/18-GQA-MQA.md","sha":"f9874ea22ba7aaaa0253502dc25c513cd563e77f","commit":"e29d19298e86f4a779064883e23b1a72bf6687fb"},"referenceOnly":false} -->
# 第 18 题：Group Query Attention（GQA）与 Multi-Query Attention（MQA）

## 一、MHA、GQA、MQA 的差别

令 Query 头数为 h_q，KV 头数为 h_kv，每头维度 d_h。

- MHA：h_kv=h_q，每个 Q 头有对应的 K/V 头。
- GQA：1<h_kv<h_q，若干 Q 头共享一个 KV 头，标准均匀分组要求 h_q 可被 h_kv 整除。
- MQA：h_kv=1，所有 Q 头共享一组 K/V。

共享 K/V 不等于共享 Q，也不意味着同组 Q 头得到相同注意力权重：各自的 Q 不同，仍分别计算分数和 softmax。

## 二、形状与显存手算

投影矩阵 W_Q 的形状为 d×(h_q d_h)，W_K/W_V 为 d×(h_kv d_h)。每层 KV cache 粗略形状为 [2,batch,h_kv,seq_len,d_h]，不同实现的维度次序可以不同。

容量约为 2×层数×总缓存 token 数×h_kv×d_h×每元素字节数。

固定 h_q=32，其余条件相同，GQA h_kv=8 的 KV 容量是 MHA 的 1/4，MQA 为 1/32。采用 32 层、d_h=128、2 字节元素，8192 token：MHA 约 4 GiB，8-KV-head GQA 约 1 GiB，MQA 约 128 MiB；不计权重和其他开销。

## 三、收益与边界

减少 KV 投影及缓存容量，也有机会降低 decode 的 KV 读取压力。但 Q 头仍然存在，注意力分数与加权聚合不会自动按 h_kv/h_q 等比例下降，端到端加速取决于 kernel 是否有效复用共享 KV、上下文长度、batch 及瓶颈。

GQA/MQA 对 KV 表示施加共享约束，可能带来质量与效率权衡；不能仅凭结构断言某个实际模型“表达最弱”或质量必然更差，需要比较训练方式和评测结果。现有 MHA 权重不能随意平均后直接宣称无损转换，转换与适配需专门方法。

## 四、面试答法

MHA 每个 Q 头配一组 KV；GQA 让一组 Q 头共享 KV；MQA 是只有一组 KV 的极端情况。它们主要减少 KV cache 和相关访存，在质量与推理效率之间取舍。显存按 KV 头数算，不能误用 Query 头数。

参考：[GQA](https://arxiv.org/abs/2305.13245)。

## 整理来源

结合 DeepInfer 项目精读中的通用基础内容整理（2026-11-02）。教学数值与普通 Transformer 推导不代表 DeepInfer 实测；项目源码、MTPVarlen、调度和部署边界仍见原项目笔记。

---

*上一篇：[第 17 题 - 模型量化](./17-模型量化.md)*  
*下一篇：[第 19 题 - RoPE](./19-RoPE.md)*
