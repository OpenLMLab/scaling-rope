### Scaling Laws of RoPE-based Extrapolation

论文：https://arxiv.org/abs/2310.05209  
&emsp; Xiaoran Liu, Hang Yan, Shuo Zhang, Chenxin An, Xipeng Qiu, Dahua Lin  
&emsp; School of Computer Science, Fudan University  
&emsp; Shanghai AI Lab  
知乎：https://zhuanlan.zhihu.com/p/660073229

**代码、权重、评测 coming soon**

## 外推效果

### 4K长度续训

放大base，即使在原始长度（4K）文本上续训，也能显著的改进模型的外推效果，展现出以下的特点：

1. 模型的外推效果很好，甚至可以直接外推超过续训长度；这个效果和Code LLaMA使用16K续训但取得了100K长度的外推是相一致的。
2. 模型外推存在一个明显的上界，在这个范围内以内，模型语言建模的困惑度和准确率基本保持在一个稳定的范围内。但是一旦超过一个界限以后，模型的外推表现会严重的退化，表现在困惑度和准确率出现急剧的上升。
3. 模型的外推效果随base的变化稳步提升，并且随base变化比较均匀，随着base增大，模型能够稳定地外推到更长的词窗。
4. 相较于dynamic NTK，base放大并续训的方法，在超过外推上界后的崩坏趋势是远远超过dyanmic NTK的退化速度的；因此，对于放大base并续训，超过外推上界后的效果总是会落后于dynamic NTK的。但是在外推上界之内，该方案的效果是远远好于dynamic NTK的。

<p align="center">
    <img src="https://pic4.zhimg.com/v2-632e630ef8b0572f36a4c1d0f80c093f_r.jpg" width="600"/>
<p>

缩小base，并在原始训练长度（4K）文本上续训，仍然取得了显著的外推效果提升，展现出以下的特点：

1. 模型可以外推，但超过训练长度后，模型的困惑度仍然会上升，但是上升的会比较平坦；并且base越小上升越慢，曲线越平缓。
2. 模型外推并不存在一个明显的上界，模型语言建模的困惑度和准确率始终随上下文长度增加稳步退化。
3. 模型的外推效果随base的变化并不是一个均匀的过程，base在2608至652区间中，提升加速显著，并且在base取500时最终超过dynamic NTK方案。
4. 相较于dynamic NTK，虽然对于10000至652的绝大多数base，其效果都无法超过dynamic NTK，但是当base取到足够小之后，其外推曲线会足够平缓以至于在50K乃至更长长度一致优于dynamic NTK。

<p align="center">
    <img src="https://pic2.zhimg.com/v2-acb4d5b7a983a6bb86be8a1fec79a435_r.jpg" width="600"/>
<p>

### 16K长度续训

放大或缩小base，并在16K长文本上续训，外推效果显著提升，相较于原始长度续训，展现出以下的特点：
1. base越小外推效果越好，但base越大外推效果先变差后变好，base=10000不再是外推最差的base。
2. 无论base=500，还是base=1000000，都可以胜任100K长度的外推。

<p align="center">
    <img src="https://pic3.zhimg.com/v2-456112f9c3a1e964c2d1f33a63c7888e_r.jpg" width="600"/>
<p>

缩小base（base=500）和 放大base（base=1000000），在更长序列（1M）上外推效果的测试效果：

| | 128K | 256K | 512K | 1M |
| --- |:---:|:---:|:---:|:---:|
| base=500 | 9.15 | 12.41 | 22.78 | 51.28 |
| base=500 log-scaled | 9.13 | 10.01 | 12.07 | 19.07 |
| base=1000000 | 7.07 | 76.82 | 1520.41 | 8349.9 |

## 原理解释

### RoPE外推的临界维度

**引理 1. (临界维度的定义)** 对于基于RoPE的大语言模型（RoPE-based LLMs），假设其预训练文本长度为 $T_\text{train}$，自注意力头维度数量为$d$，即 $\bm{q}_t,\bm{k}_s\in\mathbb{R}^d$。那么存在这样一个维度， $d_\text{extra}$ ：前$d_\text{extra}$个维度 感知了对应维度上全周期的位置编码，后 $d-d_\text{extra}$ 个维度 只感知了对应维度上一个周期内的部分编码，如下式所示。  
$$\begin{aligned}
T_n=\frac{2\pi}{\theta_n}=2\pi\cdot{10000}^{\frac{2n}{d}}\leq T_\text{train}\text{,} &\text{\quad for\ }n=0,\cdots,d_\text{extra}/2-1\text{,} \\
T_n=\frac{2\pi}{\theta_n}=2\pi\cdot{10000}^{\frac{2n}{d}}>T_\text{train}\text{,} &\text{\quad for\ }n=d_\text{extra}/2,\cdots,d/2-1\text{.} 
\end{aligned}\tag{12}$$  
因此，对于基于RoPE的大语言模型，我们将 $d_\text{extra}$，即 $\bm{q}_t,\bm{k}_s$ 中感知了全周期位置编码的维度的数量，称作 **RoPE外推的临界维度**（**critical dimension for RoPE-based extrapolation**），计算方式如下式所示。  
$$d_\text{extra}=2\left\lceil{\dfrac{d}{2}}\log_{10000}{\dfrac{T_\text{train}}{2\pi}}\right\rceil
\text{.}\tag{13}$$

对于LLaMA2，根据其训练长度 $T_\text{train}=4096$ ，注意力头维度 $d=128$ ，可以得到 $d_\text{extra}=92$ ，即**LLaMA2中前92维度都是感知了完整的位置信息**，在外推时是比较可靠的，**后36维度 由于没有感知完整的位置信息 是外推问题的根源**。

<p align="center">
    <img src="https://pic3.zhimg.com/v2-f96a63e1a1df8b7cc50ce4a065cbdf4e_r.jpg" width="600"/>
<p>

### RoPE外推的缩放法则

**定理 3. (扩展的RoPE外推的缩放法则)** 对于基于RoPE的大语言模型（RoPE-based LLMs），假设其预训练文本长度 $T_\text{train}$，对应临界维度 $d_\text{extra}$，如果在微调阶段将base调整为$\beta>1$，并且使用更长长度长度 $T_\text{tune}\geq T_\text{train}$ 的文本续训，那么模型的外推能力不降；当且仅当 $\beta=10000$ 且 $T_\text{tune}=T_\text{train}$ 时，外推效果不变。此外，存在一个 **临界base**  $\beta_0$ ，根据 续训文本长度 $T_\text{tune}$ 和 预训练文本长度 $T_\text{train}$ 决定：  
$$\beta_0={10000}^{\log_{\frac{T_\text{train}}{2\pi}}{\frac{T_\text{tune}}{2\pi}}}\text{.}\tag{16a}$$  
如果 $\beta>\beta_0$，外推上界根据 base取值 $\beta$ 和 临界维度 $d_\text{extra}$ 决定:  
$$T_\text{extra}=2\pi\cdot\beta^{d_\text{extra}\cdot\frac{1}{d}}= 2\pi\cdot\beta^{\left\lceil{\frac{d}{2}}\log_{10000}{\frac{T_\text{train}}{2\pi}}\right\rceil\cdot{\frac{2}{d}}}\text{.}\tag{16b}$$  
如果 $\beta\leq\beta_0$，外推上界就是续训长度 $T_\text{tune}$，但是 临界维度会更新如下：  
$$d'_\text{extra}=2\left\lceil{\frac{d}{2}}\log_{\beta}{\frac{T_\text{tune}}{2\pi}}\right\rceil\geq2\left\lceil{\frac{d}{2}}\log_{10000}{\frac{T_\text{train}}{2\pi}}\right\rceil=d_\text{extra}\text{.}\tag{16c}$$  
虽然如此，如果 $\beta$ 足够小，模型还是可以外推超过 $T_\text{tune}$；特别地，如果 $\beta$ 小于如下的 $\beta_1,\beta_2,\beta_3$，外推效果会得到显著提升。  
$$\beta_1 = \frac{2 T_\text{tune}}{\pi}\text{, \quad}\beta_2 = \frac{T_\text{tune}}{\pi}\text{, \quad}\beta_3 = \frac{T_\text{tune}}{2\pi}\text{.}\tag{16d}$$  

将不同base取值下续训LLaMA2实际支持的最大上下文长度，对比理论外推上界，两者呈现惊人的重合。

<p align="center">
    <img src="https://pic2.zhimg.com/v2-3edf0f17bbc67a35878b4eee67aebde9_r.jpg" width="600"/>
<p>

<!-- ## 代码结构

## 任务评测 -->
