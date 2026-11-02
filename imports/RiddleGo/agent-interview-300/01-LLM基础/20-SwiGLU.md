<!-- zhiye:{"syncMode":"full","title":"第 20 题：SwiGLU 激活函数相比 ReLU / GELU 的优势","tags":["AI","面试","导入资料","基础整合"],"visibility":"public","folderPath":["外部资料","Agent 面试 300 题","01-LLM基础"],"pinned":false,"source":{"url":"https://github.com/RiddleGo/agent-interview-300","branch":"main","path":"01-LLM基础/20-SwiGLU.md","sha":"cbe8d6ae4c39c927381a94c844d59d2d85ac71da","commit":"e29d19298e86f4a779064883e23b1a72bf6687fb"},"referenceOnly":false} -->
# 第 20 题：SwiGLU 激活函数相比 ReLU / GELU 的优势

## 一、先理解 FFN：逐位置的非线性变换

普通稠密 Transformer 的 FFN 对各位置使用同一套参数，但逐位置计算，不在该子层直接混合不同 token。输入本身已经包含前面注意力聚合的上下文，因而“逐位置计算”不等于“输出与上下文无关”。

统一采用行向量写法，忽略 bias：

$$
\operatorname{FFN}(x)=\sigma(xW_1)W_2
$$

W1 为 d×f，W2 为 f×d，常见教学配置 f=4d。残差与归一化在 block 中配合使用，Pre-Norm/Post-Norm 的具体位置不同。注意力负责跨位置聚合，FFN 负责对每个位置表示进行非线性加工。

## 二、ReLU、GELU 与 SwiGLU

ReLU(x)=max(0,x)，正半轴导数为 1，负半轴为 0，不能说“所有非零点梯度都是 1”。GELU 的定义是 xΦ(x)，工程中可以采用近似式。

Swish_β(x)=x·sigmoid(βx)，常用 β=1 时即 SiLU。SwiGLU FFN 为：

$$
\operatorname{FFN}_{\mathrm{SwiGLU}}(x)=\left[\operatorname{SiLU}(xW_g)\odot(xW_u)\right]W_d
$$

Wg/Wu 为 d×f，Wd 为 f×d。两路投影，一路经过 SiLU，逐元素相乘后映射回 d。它不是单纯把 ReLU 替换成一个标量函数，而是更换了 FFN 的门控结构。

SiLU 门控值不限制在 [0,1]，因此“开关”只是类比。门控可提供更灵活的特征组合，相关实验在特定训练设置下改善质量，不保证所有任务、所有配置严格优于 ReLU/GELU。

## 三、为什么中间维常从 4d 改成约 8d/3？

忽略 bias 和逐元素算子，一次乘加计 2 FLOPs，M 个位置：

- 普通 FFN：2df 参数，4Mdf FLOPs。f=4d 时为 8d² 参数、16Md² FLOPs。
- SwiGLU：3df 参数，6Mdf FLOPs。
- 对齐参数量，令 3df=8d²，得到 f=8d/3；实际通常按硬件友好的倍数取整，也可能采用其他倍率。

标准 MHA 的 QKV+O 为 4d² 参数、**8Md² FLOPs**，不是 12Md²。与 f=4d 的普通 FFN 相加，线性层参数为 12d²，FFN 占 8/12=2/3，线性投影 FLOPs 也占 16/(16+8)=2/3。

**这个 2/3 只适用于上述结构和统计范围。**加入 QKᵀ/AV 的序列交互、embedding、不同 FFN 宽度、GQA 或 MoE 后，占比会变，不能说 FFN 在所有模型、所有长度下恒占总计算的 2/3。长序列的注意力交互可能很昂贵。

## 四、FFN 是否就是知识存储器？

研究发现某些 FFN 激活和权重可以用模式检测、key-value 记忆等视角解释，也有模型编辑方法利用 FFN 修改事实关联。但这不是“所有知识只在 FFN”的定理，更不能用参数占比直接证明知识存储位置。知识和行为也依赖注意力、embedding 及跨层交互。

## 五、从门控 FFN 到 MoE

MoE 用路由器选择一部分专家 FFN，每个 token 激活 top-k 等形式的子集，扩大总参数容量而不每次运行全部专家。k 不是固定为 2，不同架构还可能有共享专家。

单 token 成本由激活专家数、专家宽度、共享专家和路由/通信开销决定；“总参数更多”不等于“单 token 计算必然不变”。容量约束和负载均衡还可能引入 token 间的调度耦合。

FFN 也不总是计算受限：小 batch decode 中权重访问可能主导；较大的 prefill/batch 更容易提高矩阵复用与算术强度，要根据实际形状和硬件分析。

## 六、面试答法与自测

FFN 是逐位置共享参数的非线性变换。SwiGLU 用两路投影，一路 SiLU 门控后与另一条相乘，再投影回原维度；相比普通两矩阵 FFN 多一个矩阵，因此常缩小中间维来对齐预算。优势是门控结构在实验中表现良好，不能当成理论上的无条件优势。

自测：普通 f=4d 的 FFN 参数为 8d²，SwiGLU 令 f≈8d/3 可近似对齐；QKV+O 为 8Md² FLOPs；2/3 指特定结构的线性层占比，不包括所有注意力计算。

参考：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)、[FFN Key-Value Memories](https://arxiv.org/abs/2012.14913)、[Switch Transformers](https://arxiv.org/abs/2101.03961)。

## 整理来源

结合 DeepInfer 项目精读中的通用基础内容整理（2026-11-02）。教学数值与普通 Transformer 推导不代表 DeepInfer 实测；项目源码、MTPVarlen、调度和部署边界仍见原项目笔记。

---

*上一篇：[第 19 题 - RoPE](./19-RoPE.md)*  
*下一篇：进入 [Agent 核心概念](../02-Agent核心概念)（第 21 题）*
