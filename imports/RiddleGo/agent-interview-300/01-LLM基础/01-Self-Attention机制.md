<!-- zhiye:{"syncMode":"full","title":"第 1 题：Transformer 中的 Self-Attention 机制与计算复杂度","tags":["AI","面试","导入资料","基础整合"],"visibility":"public","folderPath":["外部资料","Agent 面试 300 题","01-LLM基础"],"pinned":false,"source":{"url":"https://github.com/RiddleGo/agent-interview-300","branch":"main","path":"01-LLM基础/01-Self-Attention机制.md","sha":"670e78fd819a14ec753fbaafa10e3f01af0f7926","commit":"e29d19298e86f4a779064883e23b1a72bf6687fb"},"referenceOnly":false} -->
# 第 1 题：Transformer 中的 Self-Attention 机制与计算复杂度

## 一、Self-Attention 的定义与计算

输入 X∈R^(n×d)，每个头计算 Q=XW_Q、K=XW_K、V=XW_V，再计算：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

Q 用来查询，K 用来匹配，V 提供被加权聚合的信息。可见范围由 mask M 决定：双向注意力可以读取整段有效输入，因果注意力只能读取当前位置及过去。

缩放的直觉：若 Q/K 各分量独立、零均值且单位方差，点积方差约为 d_k，除以 √d_k 后稳定到常数量级，缓解 softmax 饱和。这是初始化附近的解释，不是训练后各分量始终独立的保证。

**无位置编码、无固定顺序掩码的自注意力是置换等变，不是输出完全不变。**若输入变成 PX，输出随之变成 P·Attention(X)；分数矩阵则变成 P(QKᵀ)Pᵀ。因此它本身不能表达 token 的绝对排列。固定的 causal mask 会引入顺序结构，不能直接套用上述无掩码结论。

## 二、多头怎么拆：每个头看完整输入

设 h 个头，d_k=d_v=d/h。每个 W_Qi 的形状都是 d×d_k，每个头都对完整 d 维输入投影，不是先把原始输入切成 h 份。

采用行向量约定 XW 时：

$$
W_Q=[W_{Q1}\;W_{Q2}\;\cdots\;W_{Qh}]\in\mathbb R^{d\times d}
$$

这里是**沿列横向拼接**。一次大投影 XW_Q 等价于各头独立投影，再把输出 reshape 成 (batch, heads, tokens, head_dim)。某些框架按 (out_features,in_features) 存储权重，所以其存储表示是沿行堆叠；不要混淆数学约定和存储布局。

## 三、多头怎么合：concat 后做输出投影

$$
Y=[H_1\;\cdots\;H_h]W_O=\sum_i H_iW_{O,i}
$$

W_O∈R^(h·d_v×d)，W_O,i 是对应的行块。它在注意力子层内混合各头输出并对齐残差维度。去掉 W_O 会改变该子层的结构，但不能说“各头永远无法交流”，后续 FFN 等层仍可以混合通道。

头编号没有固定语义。要交换两个头且保持函数不变，必须同时交换它们完整的 Q、K、V 投影，并交换 W_O 对应行块；只交换 V 和 W_O 通常不够，因为两头的注意力分布可能不同。

## 四、手算：两个头产生不同注意力分布

取 d=4、h=2，X 的三行为 [1,0,1,0]、[0,1,0,1]、[1,1,0,0]，教学上令 Q/K/V 投影均为单位矩阵。为观察 x1 对后面位置的注意力，本例采用**无 causal mask 的双向注意力**；实际 Decoder 的 x1 不能看未来。

对 x1：

- head1 的原始点积为 [1,0,1]，除以 √2 后 softmax≈[0.4011,0.1978,0.4011]，输出≈[0.8022,0.5989]。
- head2 的原始点积为 [1,0,0]，除以 √2 后 softmax≈[0.5035,0.2483,0.2483]，输出≈[0.5035,0.2483]。
- concat 得到约 [0.8022,0.5989,0.5035,0.2483]，再乘 W_O。若输出第一维取两个头第一维之和，就约为 1.3057。

这是特殊单位投影下的教学例子；一般训练后的每个头都通过稠密投影读取完整输入。原始分数为 0 不等于注意力权重为 0，真正屏蔽要使用 mask。

## 五、为什么用多头？成本是否增加？

多头提供多张独立的 softmax 分布，可以同时按不同关系聚合信息；分工由训练学习，不是架构规定某个头必然负责语法。它提供有用的归纳偏置，但不能无条件声称“所有任务上严格强于同宽单头”，也不能靠简单的秩相加就证明其全面优势。

在固定总宽度 d、h·d_k=d 的标准 MHA 中，忽略 bias：QKV 共 3d² 参数，W_O 为 d²，总共 4d²，与 h 无关。QKV+O 的投影 FLOPs 是 **8nd²**（一次乘加按 2 FLOPs），不是 12nd²。多头注意力两个乘法合计约 4n²d，主要乘法量也不随 h 线性增长；但 softmax、内存布局、矩阵尺寸和 kernel 效率会变化，所以不能说增加头数完全免费。

## 六、复杂度：分清 prefill、decode 和实现方式

标准 MHA 的完整前向约为 O(nd²+n²d)，不能不比较 n 和 d 就把平方项宣布为所有场景的主导项。单头完整矩形注意力的 QKᵀ 与 AV 合计约 4n²d_k；因果实现跳过上三角可以减少常数，量级不变。

朴素实现物化 h 张 n×n 分数矩阵，中间空间为 O(hn²)；固定头数时常简写为 O(n²)。FlashAttention 分块融合、在线计算 softmax，避免物化完整矩阵，优化 IO 和中间显存，但精确稠密注意力的算术量级仍为平方级。

普通 decode 的一个新 Q 面对 S 个历史及当前 K/V，单头注意力约 4Sd_k，即 O(Sd_k)。KV cache 避免重算旧位置，不让历史读取消失。输入长 L、再增量推进 T 步，累计注意力约 O((TL+T²)d)，另加每步投影和 FFN 成本。

分块 prefill：q 个新位置读取 c 个已有位置，矩形配对数 q(c+q)；因果有效配对为 qc+q(q+1)/2。因此相同 q、不同 c 的块，成本可以明显不同。q=1024 时，c=0 与 c=32768 的矩形配对数相差 33 倍；这不是端到端时间必然相差 33 倍。

## 七、面试答法与自测

Self-Attention 用 Q/K 匹配、softmax 归一化后加权 V；多头把投影后的通道分组，各自学习一张注意力分布，再 concat 并用 W_O 混合。固定总宽度时多头不增加主要投影参数量。完整前向需分别计算线性投影 O(nd²) 和注意力 O(n²d)，decode 用 KV cache 后仍需线性读取历史。

自测：d=512、h=8，则 d_k=64，QKV+O 参数量为 4×512²；交换头要交换 Q/K/V 及输出投影行块；0 分不会被 softmax 自动屏蔽。

参考：[Transformer](https://arxiv.org/abs/1706.03762)、[FlashAttention](https://arxiv.org/abs/2205.14135)。

## 整理来源

结合 DeepInfer 项目精读中的通用基础内容整理（2026-11-02）。教学数值与普通 Transformer 推导不代表 DeepInfer 实测；项目源码、MTPVarlen、调度和部署边界仍见原项目笔记。

---

*下一篇：[第 2 题 - 位置编码：绝对 vs 相对](./02-位置编码.md)*
