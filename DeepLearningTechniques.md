# Deep Learning Techniques: Complete Documentation

## Table of Contents
1. Fundamentals of Neural Networks
2. Convolutional Neural Networks (CNN)
3. Recurrent Neural Networks (RNN)
4. Long Short-Term Memory (LSTM)
5. Gated Recurrent Units (GRU)
6. Attention Mechanisms
7. Transformer Architecture
8. Advanced Transformer Variants
9. Modern Architectures (2023-2025)
10. Comparative Analysis

---

## 1. Fundamentals of Neural Networks

### 1.1 Basic Neural Network Architecture

A neural network consists of interconnected layers of neurons that transform input data through learned weights and biases.

**Forward Propagation:**

For a single neuron:
$$z = \sum_{i=1}^{n} w_i x_i + b$$

$$a = \sigma(z)$$

Where:
- $w_i$ = weights
- $x_i$ = inputs
- $b$ = bias
- $\sigma$ = activation function
- $a$ = output activation

**Common Activation Functions:**

1. **ReLU (Rectified Linear Unit):**
   $$\text{ReLU}(x) = \max(0, x)$$

2. **Sigmoid:**
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

3. **Tanh:**
   $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

4. **Softmax (for multi-class classification):**
   $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

**Backpropagation:**

Loss function (Mean Squared Error):
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Gradient descent update:
$$w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}$$

Where $\eta$ is the learning rate.

**Advantages:**
- Universal function approximators
- Can learn complex patterns
- Flexible architecture

**Disadvantages:**
- Prone to overfitting
- Requires large amounts of data
- Computationally expensive

---

## 2. Convolutional Neural Networks (CNN)

### 2.1 Architecture Overview

CNNs are specialized for processing grid-like data (images, videos, time series).

**Key Components:**

1. **Convolutional Layer**
2. **Pooling Layer**
3. **Fully Connected Layer**

### 2.2 Convolutional Layer

**Operation:**

For input $X$ and kernel $K$:
$$(X * K)(i,j) = \sum_{m}\sum_{n} X(i+m, j+n) \cdot K(m,n)$$

With multiple filters:
$$Y^k = \sigma(X * W^k + b^k)$$

Where:
- $Y^k$ = output feature map for filter $k$
- $W^k$ = weight kernel for filter $k$
- $b^k$ = bias for filter $k$

**Output size calculation:**
$$O = \frac{I - K + 2P}{S} + 1$$

Where:
- $O$ = output dimension
- $I$ = input dimension
- $K$ = kernel size
- $P$ = padding
- $S$ = stride

### 2.3 Pooling Layer

**Max Pooling:**
$$y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}$$

**Average Pooling:**
$$y_{i,j} = \frac{1}{|R_{i,j}|}\sum_{(m,n) \in R_{i,j}} x_{m,n}$$

### 2.4 Popular CNN Architectures

**LeNet-5 (1998):**
- First successful CNN
- Used for digit recognition

**AlexNet (2012):**
- 8 layers (5 conv + 3 FC)
- ReLU activation
- Dropout regularization

**VGGNet (2014):**
- Deep architecture (16-19 layers)
- Small 3×3 filters throughout
- Simple and uniform architecture

**ResNet (2015):**

Introduces skip connections:
$$H(x) = F(x) + x$$

Where $F(x)$ is the residual mapping.

**Residual Block:**
$$y = \sigma(F(x, \{W_i\}) + x)$$

This solves the vanishing gradient problem in very deep networks.

**Inception/GoogLeNet (2014):**

Parallel convolutions of different sizes:
- 1×1, 3×3, 5×5 convolutions
- Max pooling
- Concatenation of outputs

**EfficientNet (2019):**

Compound scaling:
$$\text{depth: } d = \alpha^\phi$$
$$\text{width: } w = \beta^\phi$$
$$\text{resolution: } r = \gamma^\phi$$

Subject to: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ and $\alpha \geq 1, \beta \geq 1, \gamma \geq 1$

### 2.5 Modern CNN Variants

**MobileNet:**
- Depthwise separable convolutions
- Reduces parameters significantly

**DenseNet:**
$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

Each layer receives all previous feature maps as input.

**Advantages of CNNs:**
- Parameter sharing (weight sharing in kernels)
- Translation invariance
- Spatial hierarchy learning
- Fewer parameters than fully connected networks
- Excellent for image-related tasks

**Disadvantages:**
- Not naturally suited for sequential data
- Fixed input size (without modifications)
- Poor at capturing long-range dependencies
- Requires large labeled datasets

**Usage:**
- Image classification
- Object detection
- Semantic segmentation
- Video analysis
- Medical image analysis

---

## 3. Recurrent Neural Networks (RNN)

### 3.1 Basic RNN Architecture

RNNs process sequential data by maintaining a hidden state.

**Forward Pass:**
$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$y_t = W_{hy}h_t + b_y$$

Where:
- $h_t$ = hidden state at time $t$
- $x_t$ = input at time $t$
- $y_t$ = output at time $t$
- $W_{hh}$, $W_{xh}$, $W_{hy}$ = weight matrices
- $b_h$, $b_y$ = bias vectors

### 3.2 Backpropagation Through Time (BPTT)

Loss across time:
$$L = \sum_{t=1}^{T} L_t$$

Gradient computation:
$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T}\frac{\partial L_t}{\partial W}$$

**Vanishing Gradient Problem:**

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t}\frac{\partial h_i}{\partial h_{i-1}}$$

If $|\frac{\partial h_i}{\partial h_{i-1}}| < 1$, gradients vanish exponentially.

### 3.3 Bidirectional RNN

$$\overrightarrow{h_t} = f(W_{\overrightarrow{h}}\overrightarrow{h_{t-1}} + W_x x_t + b_{\overrightarrow{h}})$$
$$\overleftarrow{h_t} = f(W_{\overleftarrow{h}}\overleftarrow{h_{t+1}} + W_x x_t + b_{\overleftarrow{h}})$$
$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

**Advantages:**
- Can process sequential data
- Maintains temporal information
- Variable-length input handling

**Disadvantages:**
- Vanishing/exploding gradient problem
- Difficulty learning long-term dependencies
- Sequential processing (slow training)
- Limited parallelization

**Usage:**
- Time series prediction
- Natural language processing
- Speech recognition
- Music generation

---

## 4. Long Short-Term Memory (LSTM)

### 4.1 LSTM Architecture

LSTM solves the vanishing gradient problem through gating mechanisms.

**LSTM Cell Equations:**

**Forget Gate:**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate:**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell State:**
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State:**
$$h_t = o_t \odot \tanh(C_t)$$

Where:
- $\sigma$ = sigmoid function
- $\odot$ = element-wise multiplication
- $f_t$ = forget gate activation
- $i_t$ = input gate activation
- $o_t$ = output gate activation
- $C_t$ = cell state
- $h_t$ = hidden state

### 4.2 Gradient Flow

The cell state $C_t$ allows gradients to flow unchanged:
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

This prevents vanishing gradients for long sequences.

### 4.3 LSTM Variants

**Peephole LSTM:**

Gates can look at the cell state:
$$f_t = \sigma(W_f \cdot [C_{t-1}, h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [C_{t-1}, h_{t-1}, x_t] + b_i)$$
$$o_t = \sigma(W_o \cdot [C_t, h_{t-1}, x_t] + b_o)$$

**Advantages:**
- Learns long-term dependencies
- Solves vanishing gradient problem
- Selective memory through gates
- Stable gradient flow

**Disadvantages:**
- More complex than standard RNN (4× parameters)
- Computationally expensive
- Sequential processing (no parallelization)
- Can still struggle with very long sequences

**Usage:**
- Machine translation
- Speech recognition
- Text generation
- Video captioning
- Sentiment analysis

---

## 5. Gated Recurrent Units (GRU)

### 5.1 GRU Architecture

GRU is a simpler alternative to LSTM with fewer gates.

**GRU Equations:**

**Reset Gate:**
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**Update Gate:**
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**Candidate Hidden State:**
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

**Hidden State:**
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 5.2 GRU vs LSTM

**Parameter Count:**
- LSTM: 4 gates (forget, input, output, cell)
- GRU: 2 gates (reset, update)

**Advantages:**
- Simpler than LSTM (fewer parameters)
- Faster training and inference
- Less prone to overfitting
- Comparable performance to LSTM

**Disadvantages:**
- Less expressive than LSTM in some tasks
- Still sequential processing
- Not as widely adopted

**Usage:**
- Similar to LSTM applications
- Preferred when computational resources are limited
- Small to medium-sized datasets

---

## 6. Attention Mechanisms

### 6.1 Basic Attention

Attention allows models to focus on relevant parts of the input.

**Attention Score:**
$$e_{ij} = a(s_{i-1}, h_j)$$

Where:
- $s_{i-1}$ = decoder state
- $h_j$ = encoder hidden state
- $a$ = alignment model (typically a feedforward network)

**Attention Weights (using softmax):**
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T}\exp(e_{ik})}$$

**Context Vector:**
$$c_i = \sum_{j=1}^{T}\alpha_{ij}h_j$$

### 6.2 Types of Attention

**1. Additive (Bahdanau) Attention:**
$$e_{ij} = v^T \tanh(W_1 s_{i-1} + W_2 h_j)$$

**2. Multiplicative (Luong) Attention:**

**Dot Product:**
$$e_{ij} = s_{i-1}^T h_j$$

**General:**
$$e_{ij} = s_{i-1}^T W h_j$$

**Concatenation:**
$$e_{ij} = W[s_{i-1}; h_j]$$

### 6.3 Self-Attention

Each element attends to all elements in the same sequence.

**Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = queries matrix
- $K$ = keys matrix
- $V$ = values matrix
- $d_k$ = dimension of keys

**Why scaling by $\sqrt{d_k}$?**

For large $d_k$, dot products grow large in magnitude, pushing softmax into regions with extremely small gradients.

### 6.4 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Benefits:**
- Allows model to attend to different representation subspaces
- Increases model capacity

**Advantages of Attention:**
- Handles variable-length sequences
- Focuses on relevant information
- Interpretable (can visualize attention weights)
- Solves information bottleneck in seq2seq

**Disadvantages:**
- Quadratic complexity with sequence length
- Requires more computation than RNN
- Can overfit on small datasets

**Usage:**
- Machine translation
- Text summarization
- Question answering
- Image captioning

---

## 7. Transformer Architecture

### 7.1 Overview

Transformers (2017, "Attention Is All You Need") revolutionized deep learning by relying entirely on attention mechanisms.

**Architecture Components:**
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Networks
3. Positional Encoding
4. Layer Normalization
5. Residual Connections

### 7.2 Positional Encoding

Since Transformers have no recurrence, position information must be injected:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = position in sequence
- $i$ = dimension
- $d_{model}$ = model dimension

### 7.3 Encoder Layer

**Single Encoder Layer:**

1. **Multi-Head Self-Attention:**
$$\text{MHA}(X) = \text{MultiHead}(X, X, X)$$

2. **Add & Norm:**
$$X' = \text{LayerNorm}(X + \text{MHA}(X))$$

3. **Position-wise Feed-Forward:**
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

4. **Add & Norm:**
$$X'' = \text{LayerNorm}(X' + \text{FFN}(X'))$$

### 7.4 Decoder Layer

**Single Decoder Layer:**

1. **Masked Multi-Head Self-Attention** (prevents attending to future tokens)
2. **Add & Norm**
3. **Multi-Head Cross-Attention** (attends to encoder output)
4. **Add & Norm**
5. **Position-wise Feed-Forward**
6. **Add & Norm**

**Masking in Self-Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$$

Where $M$ is a mask matrix with $-\infty$ for future positions.

### 7.5 Complete Transformer

**Encoder:**
$$\text{Encoder}(X) = \text{EncoderLayer}_N(...\text{EncoderLayer}_1(X + PE))$$

**Decoder:**
$$\text{Decoder}(Y, \text{Enc}) = \text{DecoderLayer}_N(...\text{DecoderLayer}_1(Y + PE, \text{Enc}))$$

**Output:**
$$P(y) = \text{softmax}(W_o \cdot \text{Decoder}(Y, \text{Enc}) + b_o)$$

### 7.6 Computational Complexity

**Self-Attention:** $O(n^2 \cdot d)$
- $n$ = sequence length
- $d$ = dimension

**RNN:** $O(n \cdot d^2)$

For short sequences with large $d$, RNN is more efficient. For long sequences, Transformer parallelization compensates.

**Advantages:**
- Highly parallelizable (no sequential dependency)
- Captures long-range dependencies effectively
- State-of-the-art performance on many tasks
- Scalable architecture

**Disadvantages:**
- Quadratic complexity with sequence length
- Requires large amounts of data
- Memory intensive
- No inherent notion of word order (needs positional encoding)

**Usage:**
- Natural language processing (BERT, GPT)
- Machine translation
- Text generation
- Code generation
- Protein folding (AlphaFold)

---

## 8. Advanced Transformer Variants

### 8.1 BERT (Bidirectional Encoder Representations from Transformers)

**Architecture:** Encoder-only Transformer

**Pre-training Objectives:**

1. **Masked Language Modeling (MLM):**
   - Randomly mask 15% of tokens
   - Predict masked tokens

$$L_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i | \text{context})$$

2. **Next Sentence Prediction (NSP):**
   - Predict if sentence B follows sentence A

**Input Representation:**
$$\text{Input} = \text{Token Embeddings} + \text{Segment Embeddings} + \text{Position Embeddings}$$

**Usage:**
- Text classification
- Named entity recognition
- Question answering
- Sentiment analysis

### 8.2 GPT (Generative Pre-trained Transformer)

**Architecture:** Decoder-only Transformer

**Training Objective (Causal Language Modeling):**
$$L = -\sum_{i=1}^{n} \log P(x_i | x_1, ..., x_{i-1})$$

**Auto-regressive Generation:**
$$P(x_{1:n}) = \prod_{i=1}^{n} P(x_i | x_{1:i-1})$$

**GPT Evolution:**
- **GPT-1:** 117M parameters
- **GPT-2:** 1.5B parameters
- **GPT-3:** 175B parameters
- **GPT-4:** Estimated 1.76T parameters (mixture of experts)

**Usage:**
- Text generation
- Few-shot learning
- Translation
- Summarization
- Code generation

### 8.3 T5 (Text-to-Text Transfer Transformer)

**Architecture:** Encoder-Decoder Transformer

**Unified Framework:** All tasks as text-to-text

Examples:
- Translation: "translate English to German: That is good."
- Classification: "sentiment: This movie is terrible."

**Training:**
$$L = -\sum_{i=1}^{n} \log P(y_i | x, y_{<i})$$

### 8.4 Vision Transformer (ViT)

Applies Transformers directly to images.

**Image Patching:**
1. Split image into patches: $x \in \mathbb{R}^{H \times W \times C}$ → patches $\in \mathbb{R}^{N \times (P^2 \cdot C)}$
2. Linear projection: $z_0 = [x_{class}; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_{pos}$

Where:
- $N = \frac{HW}{P^2}$ = number of patches
- $P$ = patch size
- $E$ = embedding matrix

**Position Embeddings:** Learnable 1D position embeddings

**Classification:**
$$y = \text{MLP}(z_L^0)$$

Where $z_L^0$ is the class token output from the final layer.

### 8.5 Efficient Transformer Variants

**Linformer:**
- Reduces attention complexity from $O(n^2)$ to $O(n)$
- Projects keys and values to lower dimension

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q(EK)^T}{\sqrt{d_k}}\right)(FV)$$

Where $E, F \in \mathbb{R}^{k \times n}$ with $k \ll n$

**Reformer:**
- Locality-Sensitive Hashing (LSH) attention
- Reduces complexity to $O(n \log n)$

**Performer:**
- Uses random feature maps to approximate attention
- Linear complexity $O(n)$

**Longformer:**
- Combines local windowed attention with global attention
- $O(n)$ complexity

**Attention patterns:**
- **Sliding window:** Local context
- **Global attention:** For specific tokens
- **Dilated sliding window:** Increased receptive field

### 8.6 Sparse Transformers

**Fixed Patterns:**
$$\text{Attention}(Q, K, V)_{ij} = \begin{cases} 
\text{standard} & \text{if } (i,j) \in S \\
0 & \text{otherwise}
\end{cases}$$

Where $S$ defines sparsity pattern (e.g., local, strided, block-sparse).

---

## 9. Modern Architectures (2023-2025)

### 9.1 Mamba (State Space Models)

**State Space Formulation:**
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

**Discretization:**
$$h_k = \bar{A}h_{k-1} + \bar{B}x_k$$
$$y_k = Ch_k$$

Where:
$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

**Selective State Space:**
- Parameters $B$, $C$, $\Delta$ become input-dependent
- Linear complexity $O(n)$ with sequence length

**Advantages:**
- Linear scaling with sequence length
- Efficient training and inference
- Competitive with Transformers on long sequences

**Disadvantages:**
- More complex implementation
- Less mature ecosystem
- Not yet proven across all domains

### 9.2 RetNet (Retentive Networks)

**Retention Mechanism:**
$$\text{Retention}(Q, K, V) = QK^T \odot M \cdot V$$

Where $M$ is a causal mask with decay.

**Three Computation Paradigms:**
1. **Parallel:** Training mode (similar to Transformers)
2. **Recurrent:** Inference mode (like RNN)
3. **Chunkwise:** Efficient long sequence processing

**Recurrent Formulation:**
$$S_n = \gamma S_{n-1} + k_n^T v_n$$
$$o_n = q_n S_n$$

**Advantages:**
- Training parallelism like Transformers
- Inference efficiency like RNNs
- Linear complexity during inference

### 9.3 Mixture of Experts (MoE)

**Gating Network:**
$$G(x) = \text{softmax}(\text{TopK}(x \cdot W_g))$$

**MoE Layer:**
$$y = \sum_{i=1}^{n} G(x)_i E_i(x)$$

Where $E_i$ are expert networks.

**Load Balancing Loss:**
$$L_{aux} = \alpha \cdot \text{CV}(\text{Expert Load})$$

**Sparse MoE (GPT-4, Mixtral):**
- Only activate top-k experts
- Reduces computation while maintaining capacity

**Switch Transformer:**
- Uses top-1 routing (single expert per token)
- Simplified load balancing

### 9.4 Flash Attention

**Standard Attention Memory Bottleneck:**
- Materialized $QK^T$ matrix: $O(n^2)$ memory

**Flash Attention Algorithm:**
1. Tiles computation into blocks
2. Computes attention incrementally
3. Never materializes full attention matrix

**IO Complexity:**
- Standard: $O(n^2 d)$
- Flash Attention: $O(n^2 d / M)$ where $M$ is SRAM size

**Flash Attention 2:**
- Further optimizations
- 2-3× speedup over original

**Advantages:**
- Exact attention (not approximate)
- Reduced memory usage
- Faster training

### 9.5 Rotary Position Embeddings (RoPE)

Used in models like LLaMA, PaLM.

**Rotation Matrix:**
$$f_{q,k}(x_m, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} W_q x_m^{(1)} \\ W_q x_m^{(2)} \end{pmatrix}$$

**Properties:**
- Encodes absolute position
- Incorporates relative position in attention
- Better extrapolation to longer sequences

### 9.6 Multi-Query and Grouped-Query Attention

**Multi-Head Attention (MHA):**
- Separate K, V for each head: $h \times d_k$ memory

**Multi-Query Attention (MQA):**
- Shared K, V across heads: $d_k$ memory
- Faster inference, slight quality loss

**Grouped-Query Attention (GQA):**
- Middle ground: groups of heads share K, V
- $g \times d_k$ memory where $1 < g < h$

$$\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_{1,1}, ..., \text{head}_{g,k})W^O$$

### 9.7 Constitutional AI and RLHF

**Reinforcement Learning from Human Feedback:**

**Reward Model:**
$$r_\theta(x, y) = \text{Reward Score}$$

Trained on human preferences: $y_w \succ y_l$ (win vs loss)

**Preference Loss:**
$$L = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

**PPO Objective:**
$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**Constitutional AI:**
- Self-critique and revision
- Principle-based training

### 9.8 Diffusion Transformers (DiT)

Combines Diffusion Models with Transformers for image generation.

**Forward Diffusion:**
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**Reverse Process (Denoising):**
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Transformer Backbone:**
- Processes image patches
- Conditions on timestep $t$ and optional class/text

### 9.9 Perceiver and Perceiver IO

**Cross-Attention to Latent Array:**
$$z = \text{CrossAttend}(L, X)$$

Where:
- $L$ = learned latent array (fixed size)
- $X$ = input (any modality, variable size)

**Complexity:**
- Standard Transformer: $O(n^2)$
- Perceiver: $O(nm)$ where $m \ll n$

**Perceiver IO:**
- Decoder cross-attends from latent to output queries
- Handles arbitrary output structures

---

## 10. Comparative Analysis

### 10.1 Complexity Comparison

| Architecture | Time Complexity | Space Complexity | Parallelization |
|-------------|-----------------|------------------|-----------------|
| RNN | $O(n)$ | $O(1)$ | Sequential |
| LSTM/GRU | $O(n)$ | $O(1)$ | Sequential |
| CNN | $O(n \cdot k)$ | $O(k)$ | Parallel |
| Transformer | $O(n^2 \cdot d)$ | $O(n^2)$ | Parallel |
| Linformer | $O(n \cdot k)$ | $O(n \cdot k)$ | Parallel |
| Reformer | $O(n \log n)$ | $O(n \log n)$ | Parallel |
| Mamba/SSM | $O(n)$ | $O(1)$ | Recurrent/Parallel |

### 10.2 Task Suitability

**Image Tasks:**
- **Best:** CNN, ViT, ConvNeXt
- **Emerging:** Diffusion Transformers

**Sequential Tasks (NLP):**
- **Best:** Transformers (BERT, GPT, T5)
- **Long Sequences:** Longformer, BigBird, Mamba
- **Efficient:** RetNet, RWKV

**Time Series:**
- **Traditional:** LSTM, GRU
- **Modern:** Temporal Fusion Transformer, Informer
- **Efficient:** N-BEATS, NLinear

**Multi-Modal:**
- **Best:** CLIP, Flamingo, GPT-4V
- **Unified:** Perceiver IO, Unified-IO

### 10.3 Training Considerations

**Data Requirements:**
- **Low:** Transfer learning with pre-trained models, fine-tuning
- **Medium:** CNNs, LSTMs with proper regularization
- **High:** Transformers from scratch, ViT, large language models
- **Very High:** Foundation models (GPT-4, PaLM, Claude)

**Computational Resources:**

**Training FLOPs Estimation:**

For Transformers:
$$\text{FLOPs} \approx 6PD + 12LD^2$$

Where:
- $P$ = number of parameters
- $D$ = model dimension
- $L$ = sequence length

**Inference Cost:**

| Model Type | Parameters | Inference Time (relative) |
|-----------|-----------|---------------------------|
| LSTM | 1× | 1× |
| Transformer (base) | 3× | 0.5× (parallel) |
| MoE Transformer | 10× (sparse) | 1× (active params) |
| Mamba | 1× | 0.3× (linear) |

### 10.4 Architecture Selection Guide

**Decision Tree:**

```
Is your data sequential?
├─ No (Images/Static) → CNN, ViT
└─ Yes
   ├─ Short sequences (< 512 tokens)?
   │  └─ LSTM/GRU or Transformer
   └─ Long sequences (> 512 tokens)?
      ├─ Need bidirectional context?
      │  ├─ Yes → Longformer, BigBird
      │  └─ No → GPT-style decoder, Mamba
      └─ Extremely long (> 100K tokens)?
         └─ Mamba, RWKV, Ring Attention
```

---

## 11. Advanced Training Techniques

### 11.1 Optimization Algorithms

**Adam (Adaptive Moment Estimation):**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

Bias correction:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

Update:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$

Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**AdamW (Adam with Weight Decay):**

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_{t-1}\right)$$

**Lion Optimizer (2023):**

$$c_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$m_t = \beta_2 m_{t-1} + (1-\beta_2)g_t$$
$$\theta_t = \theta_{t-1} - \eta \cdot \text{sign}(c_t)$$

Benefits:
- More memory efficient
- Better generalization in some cases
- Larger learning rates possible

**AdaFactor:**

Reduces memory by not storing second moments explicitly:
$$v_t = \text{row-wise and column-wise factorization}$$

Useful for very large models.

### 11.2 Learning Rate Schedules

**Warmup with Cosine Decay:**

$$\eta_t = \begin{cases}
\frac{t}{T_{warmup}}\eta_{max} & t \leq T_{warmup} \\
\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{max} - T_{warmup}}\pi\right)\right) & t > T_{warmup}
\end{cases}$$

**Linear Warmup with Linear Decay:**

$$\eta_t = \eta_{max} \cdot \min\left(\frac{t}{T_{warmup}}, 1 - \frac{t - T_{warmup}}{T_{max} - T_{warmup}}\right)$$

**OneCycle Policy:**

1. Linear warmup: $\eta_{min} \to \eta_{max}$
2. Cosine annealing: $\eta_{max} \to \eta_{min}$

### 11.3 Regularization Techniques

**Dropout:**

$$y = \frac{1}{1-p} \cdot \text{mask}(x)$$

Where mask randomly zeros elements with probability $p$.

**DropConnect:**

Drops connections instead of activations:
$$y = (W \odot M)x$$

**Layer Dropout (Stochastic Depth):**

Randomly drops entire layers during training:
$$H_l = \begin{cases}
F_l(H_{l-1}) + H_{l-1} & \text{with prob } 1-p_l \\
H_{l-1} & \text{with prob } p_l
\end{cases}$$

Often used in ResNets and Transformers.

**Label Smoothing:**

Instead of hard labels:
$$y_i = \begin{cases}
1 & \text{if } i = \text{true class} \\
0 & \text{otherwise}
\end{cases}$$

Use soft labels:
$$y_i = \begin{cases}
1 - \epsilon & \text{if } i = \text{true class} \\
\frac{\epsilon}{K-1} & \text{otherwise}
\end{cases}$$

Where $K$ is number of classes, typically $\epsilon = 0.1$.

**Mixup:**

$$\tilde{x} = \lambda x_i + (1-\lambda)x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda)y_j$$

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$, typically $\alpha = 0.2$.

**CutMix (for images):**

Combines regions from two images:
$$\tilde{x} = M \odot x_i + (1-M) \odot x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda)y_j$$

Where $M$ is a binary mask, $\lambda = \frac{\text{area of cut region}}{\text{total area}}$.

### 11.4 Normalization Techniques

**Batch Normalization:**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma\hat{x}_i + \beta$$

Where $\mu_B$, $\sigma_B^2$ are batch statistics.

**Issues:** Dependent on batch size, problematic for small batches or sequences.

**Layer Normalization:**

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

Where statistics computed across features (not batch).

**Preferred for Transformers and NLP.**

**RMS Normalization:**

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma$$

Where:
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}$$

**Benefits:**
- Simpler than LayerNorm
- No mean centering
- Slightly faster

**Group Normalization:**

Divides channels into groups and normalizes within groups:
$$\hat{x}_i = \frac{x_i - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}}$$

**Use case:** Computer vision with small batch sizes.

### 11.5 Gradient Clipping and Scaling

**Gradient Clipping by Norm:**

$$g = \begin{cases}
g & \text{if } ||g|| \leq \theta \\
\frac{\theta \cdot g}{||g||} & \text{if } ||g|| > \theta
\end{cases}$$

**Gradient Clipping by Value:**

$$g_i = \max(\min(g_i, \theta), -\theta)$$

**Gradient Accumulation:**

For effective large batch training:
$$g_{acc} = \frac{1}{N}\sum_{i=1}^{N}g_i$$

Update weights after $N$ mini-batches.

**Mixed Precision Training:**

Uses FP16 for speed, FP32 for stability:

1. Forward pass in FP16
2. Loss scaling: $L_{scaled} = S \cdot L$
3. Backward pass in FP16
4. Unscale gradients: $g = \frac{g_{FP16}}{S}$
5. Update weights in FP32

**BFloat16:**

Alternative to FP16 with same exponent range as FP32:
- Better stability
- No loss scaling needed
- Slightly less precise mantissa

---

## 12. Emerging Architectures and Concepts

### 12.1 Hyena Hierarchy

**Hyena Operator:**

Replaces attention with long convolutions:
$$y = (x * h) \odot (x * h')$$

Where $*$ denotes convolution, $\odot$ element-wise multiplication.

**Advantages:**
- Sub-quadratic complexity
- Better scaling to very long sequences
- Faster inference

**Performance:**
- Competitive with Transformers on language modeling
- Superior on long-range tasks

### 12.2 RWKV (Receptance Weighted Key Value)

**Linear Attention Formulation:**

$$\text{RWKV}(t) = \frac{\sum_{\tau=1}^{t-1} e^{w(t-\tau)} \cdot k_\tau \otimes v_\tau}{\sum_{\tau=1}^{t-1} e^{w(t-\tau)} \cdot k_\tau}$$

Where:
- $w$ = position-dependent weight decay
- $\otimes$ = outer product

**Time-mixing and Channel-mixing:**

$$\text{out}_t = \sigma(r_t) \odot \text{wkv}_t$$

**Benefits:**
- Linear complexity
- RNN-like inference
- Transformer-like parallelizable training

### 12.3 Mega (Moving Average Equipped Gated Attention)

Combines exponential moving average with gated attention:

**EMA Component:**
$$h_t = \alpha_t \odot h_{t-1} + (1-\alpha_t) \odot x_t$$

**Gated Attention:**
$$\text{out}_t = \text{EMA}(x)_t \odot \text{Attention}(x)_t$$

**Advantages:**
- Linear time complexity
- Strong inductive bias for sequences
- Efficient on long contexts

### 12.4 Monarch Mixer

**Block-diagonal structured matrices:**

$$M = P^T L_1 P L_2$$

Where:
- $P$ = permutation matrix
- $L_1, L_2$ = block-diagonal matrices

**Complexity:**
- Standard: $O(n^2)$
- Monarch: $O(n \log n)$

**Applications:**
- Efficient mixing in MLP-Mixer style architectures
- Hardware-efficient implementations

### 12.5 Differential Transformer

**Key Innovation:**

Cancels noise in attention:
$$\text{DiffAttn} = \text{softmax}(Q_1K_1^T) - \lambda \cdot \text{softmax}(Q_2K_2^T)$$

Where heads are split into pairs.

**Benefits:**
- Better signal-to-noise ratio
- Focuses on relevant information
- Improved context utilization

### 12.6 Context-Aware Transformers

**Memorizing Transformers:**

Adds external memory:
$$\text{Attention}(Q, [K; K_{mem}], [V; V_{mem}])$$

**Retrieval-Augmented Generation:**

$$P(y|x) = P_{LM}(y | [x; \text{retrieve}(x)])$$

Where retrieve() fetches relevant context from external knowledge base.

**k-NN-LM:**

Interpolates with nearest neighbors:
$$P(y|x) = \lambda P_{LM}(y|x) + (1-\lambda) P_{kNN}(y|x)$$

### 12.7 Multi-Modal Architectures

**CLIP (Contrastive Language-Image Pre-training):**

**Contrastive Loss:**
$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)} + \log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ji}/\tau)}\right]$$

Where $s_{ij} = \text{sim}(I_i, T_j)$ is cosine similarity.

**Flamingo:**

Interleaves vision and language:
$$y = \text{LM}([x_{text}; \text{Perceiver}(x_{image})])$$

Uses gated cross-attention between modalities.

**Kosmos:**

Unified model treating images as foreign language:
$$P(y|x_1, ..., x_n)$$

Where $x_i$ can be text or image tokens.

**GPT-4V and Gemini:**

Native multi-modal transformers:
- Process images directly as tokens
- Unified architecture for all modalities

---

## 13. Specialized Applications

### 13.1 Graph Neural Networks (GNN)

**Message Passing:**

$$h_v^{(k+1)} = \text{UPDATE}^{(k)}\left(h_v^{(k)}, \text{AGGREGATE}^{(k)}(\{h_u^{(k)} : u \in N(v)\})\right)$$

**Graph Convolutional Network (GCN):**

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

Where:
- $\tilde{A} = A + I$ (adjacency + self-loops)
- $\tilde{D}$ = degree matrix
- $H^{(l)}$ = node features at layer $l$

**Graph Attention Networks (GAT):**

$$h_i' = \sigma\left(\sum_{j \in N(i)}\alpha_{ij}W h_j\right)$$

**Attention coefficients:**
$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i || Wh_j]))}{\sum_{k \in N(i)}\exp(\text{LeakyReLU}(a^T[Wh_i || Wh_k]))}$$

**Graph Transformer:**

Applies self-attention with graph structure:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + A_{mask}\right)V$$

**Applications:**
- Social network analysis
- Molecular property prediction
- Recommendation systems
- Knowledge graphs

### 13.2 Neural Architecture Search (NAS)

**Search Space:**

Define possible operations and connections.

**Search Strategy:**

1. **Reinforcement Learning:**
   - Controller RNN generates architectures
   - Reward based on validation accuracy

2. **Evolution:**
   - Mutate and crossover architectures
   - Select best performers

3. **Differentiable NAS (DARTS):**

$$\alpha_{(i,j)} = \text{softmax}_o\left(\frac{\alpha_{(i,j)}^o}{\tau}\right)$$

Mixed operation:
$$\bar{o}^{(i,j)}(x) = \sum_{o \in O}\frac{\exp(\alpha_{(i,j)}^o)}{\sum_{o' \in O}\exp(\alpha_{(i,j)}^{o'})}o(x)$$

**Performance Estimation:**

- Weight sharing
- Early stopping
- Learning curve extrapolation

**Notable Results:**
- EfficientNet (compound scaling)
- AmoebaNet
- NASNet

### 13.3 Continual Learning

**Catastrophic Forgetting Problem:**

Model forgets previous tasks when learning new ones.

**Elastic Weight Consolidation (EWC):**

$$L(\theta) = L_B(\theta) + \sum_i\frac{\lambda}{2}F_i(\theta_i - \theta_{A,i}^*)^2$$

Where:
- $L_B$ = loss on task B
- $F_i$ = Fisher information (importance of parameter)
- $\theta_{A,i}^*$ = optimal parameters for task A

**Progressive Neural Networks:**

Freeze previous task columns, add new columns for new tasks.

**Memory Replay:**

Store examples from previous tasks and replay during new task training.

### 13.4 Few-Shot Learning

**Prototypical Networks:**

$$d(\mathbf{x}, c_k) = ||\mathbf{f}_\theta(\mathbf{x}) - \mathbf{c}_k||^2$$

Where $\mathbf{c}_k = \frac{1}{|S_k|}\sum_{(x_i, y_i) \in S_k}\mathbf{f}_\theta(x_i)$ is the prototype.

**MAML (Model-Agnostic Meta-Learning):**

**Inner loop (task-specific):**
$$\theta_i' = \theta - \alpha\nabla_\theta L_{T_i}(f_\theta)$$

**Outer loop (meta-update):**
$$\theta = \theta - \beta\nabla_\theta\sum_{T_i \sim p(T)}L_{T_i}(f_{\theta_i'})$$

**Matching Networks:**

$$P(y|\hat{x}, S) = \sum_{(x_i, y_i) \in S}a(\hat{x}, x_i)y_i$$

Where $a(\hat{x}, x_i)$ is attention mechanism.

**In-Context Learning (LLMs):**

GPT-3 style few-shot:
$$P(y|x) = P_{LM}(y | [\text{examples}; x])$$

No gradient updates needed.

---

## 14. Hardware Considerations

### 14.1 GPU Optimization

**Tensor Cores:**

Specialized hardware for matrix multiplication:
- Mixed precision (FP16 compute, FP32 accumulate)
- Higher throughput for specific sizes

**Optimal Batch Sizes:**

$$\text{Throughput} \propto \frac{\text{Batch Size}}{\text{Time per Batch}}$$

Trade-off: Larger batches → better GPU utilization but may hurt generalization.

**Kernel Fusion:**

Combine operations to reduce memory transfers:
```
Instead of: LayerNorm → Dropout → Residual (3 kernels)
Use: FusedLayerNormDropoutResidual (1 kernel)
```

### 14.2 Distributed Training

**Data Parallelism:**

Each device processes different data:
$$\theta_{new} = \theta_{old} - \eta \cdot \frac{1}{N}\sum_{i=1}^{N}\nabla_i$$

**Model Parallelism:**

Split model across devices:
- **Pipeline parallelism:** Different layers on different devices
- **Tensor parallelism:** Split tensors within layers

**Pipeline Efficiency:**

$$\text{Efficiency} = \frac{T_{ideal}}{T_{actual}} = \frac{m \cdot t}{m \cdot t + (p-1) \cdot t}$$

Where:
- $m$ = number of micro-batches
- $p$ = number of pipeline stages
- $t$ = time per micro-batch

**ZeRO (Zero Redundancy Optimizer):**

**Stage 1:** Partition optimizer states
**Stage 2:** + Partition gradients
**Stage 3:** + Partition parameters

Memory reduction factor: $N_d$ (number of devices)

**Ring All-Reduce:**

Bandwidth optimal collective communication:
$$\text{Time} = 2(N_d - 1) \cdot \frac{M}{N_d \cdot B}$$

Where:
- $N_d$ = number of devices
- $M$ = message size
- $B$ = bandwidth

### 14.3 Quantization

**Post-Training Quantization:**

$$\hat{w} = \text{round}\left(\frac{w}{s}\right) \cdot s$$

Where $s = \frac{\max(|w|)}{2^{b-1}-1}$ is scaling factor.

**Quantization-Aware Training:**

Simulate quantization during training:
$$\hat{w} = \text{quantize}(w) + (w - \text{quantize}(w)).\text{detach}()$$

Straight-through estimator for gradients.

**INT8 Quantization:**

8-bit integers instead of FP32:
- 4× memory reduction
- 2-4× speedup
- Minimal accuracy loss with QAT

**INT4 and Lower:**

Requires careful calibration:
- GPTQ: Layer-wise quantization
- AWQ: Activation-aware weight quantization
- SmoothQuant: Migrates difficulty from activations to weights

### 14.4 Pruning

**Magnitude Pruning:**

Remove weights with smallest absolute values:
$$\text{mask}_i = \begin{cases}
1 & |w_i| > \text{threshold} \\
0 & \text{otherwise}
\end{cases}$$

**Structured Pruning:**

Remove entire channels, filters, or attention heads:
- Maintains regular computation patterns
- Better hardware utilization

**Lottery Ticket Hypothesis:**

Dense networks contain sparse subnetworks that can train from scratch to similar accuracy.

**Iterative Magnitude Pruning:**

1. Train network
2. Prune $p$% of weights
3. Reset remaining weights to initialization
4. Repeat

---

## 15. Loss Functions and Training Objectives

### 15.1 Classification Losses

**Cross-Entropy Loss:**

$$L = -\sum_{i=1}^{C}y_i\log(\hat{y}_i)$$

**Binary Cross-Entropy:**

$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Focal Loss (for imbalanced data):**

$$L = -\alpha_t(1-p_t)^\gamma\log(p_t)$$

Where:
- $p_t$ = model probability for true class
- $\gamma$ = focusing parameter (typically 2)
- $\alpha_t$ = weighting factor

Focuses on hard examples.

### 15.2 Regression Losses

**Mean Squared Error (MSE):**

$$L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

**Mean Absolute Error (MAE):**

$$L = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$

**Huber Loss (robust to outliers):**

$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

### 15.3 Sequence Generation Losses

**Teacher Forcing:**

$$L = -\sum_{t=1}^{T}\log P(y_t^* | y_{<t}^*, x)$$

Where $y^*$ are ground truth tokens.

**Scheduled Sampling:**

Mix teacher forcing with model predictions:
$$y_{t-1} = \begin{cases}
y_{t-1}^* & \text{with prob } \epsilon_t \\
\hat{y}_{t-1} & \text{with prob } 1-\epsilon_t
\end{cases}$$

**BLEU Score (evaluation):**

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N}w_n\log p_n\right)$$

Where:
- $p_n$ = n-gram precision
- $BP$ = brevity penalty
- Typically $N=4$

### 15.4 Contrastive Losses

**InfoNCE Loss:**

$$L = -\log\frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(z_i, z_j)/\tau)}$$

Where $z_i^+$ is positive pair, others are negatives.

**Triplet Loss:**

$$L = \max(0, ||a - p||^2 - ||a - n||^2 + \alpha)$$

Where:
- $a$ = anchor
- $p$ = positive
- $n$ = negative
- $\alpha$ = margin

---

## 16. Evaluation Metrics

### 16.1 Classification Metrics

**Accuracy:**
$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision:**
$$\text{Prec} = \frac{TP}{TP + FP}$$

**Recall:**
$$\text{Rec} = \frac{TP}{TP + FN}$$

**F1 Score:**
$$F_1 = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$$

**Macro F1:**
$$F_1^{macro} = \frac{1}{C}\sum_{i=1}^{C}F_1^i$$

**Matthews Correlation Coefficient:**
$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 16.2 Language Model Metrics

**Perplexity:**
$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i|w_{<i})\right)$$

Lower is better.

**ROUGE (for summarization):**

ROUGE-N (n-gram overlap):
$$\text{ROUGE-N} = \frac{\sum_{S \in \text{ref}}\sum_{\text{gram}_n \in S}\text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{ref}}\sum_{\text{gram}_n \in S}\text{Count}(\text{gram}_n)}$$

**BERTScore:**

Uses contextual embeddings:
$$R_{\text{BERT}} = \frac{1}{|x|}\sum_{x_i \in x}\max_{y_j \in y}\text{sim}(x_i, y_j)$$

---

## 17. Best Practices and Tips

### 17.1 Initialization

**Xavier/Glorot Initialization:**
$$W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

**He Initialization (for ReLU):**
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

### 17.2 Debugging Neural Networks

**Checklist:**

1. **Overfit single batch:** Model should achieve near-zero loss
2. **Check loss curves:** Training and validation
3. **Gradient norms:** Monitor for vanishing/exploding
4. **Activation distributions:** Check for dead neurons
5. **Weight updates:** Verify weights are changing

**Common Issues:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss not decreasing | Bad learning rate | Reduce LR, check gradients |
| Loss exploding | Gradient explosion | Gradient clipping, lower LR |
| Train good, val bad | Overfitting | Regularization, more data |
| Both losses plateau | Underfitting | Larger model, train longer |

### 17.3 Hyperparameter Tuning

**Learning Rate:** Most important
- Start: $10^{-3}$ to $10^{-4}$
- Use learning rate finder
- Reduce on plateau

**Batch Size:**
- Larger: More stable, faster per epoch
- Smaller: Better generalization, more updates
- Sweet spot: 32-512 for most tasks

**Weight Decay:**
- Typical: $10^{-5}$ to $10^{-2}$
- Prevents overfitting

**Dropout Rate:**
- Typical: 0.1-0.5
- Higher for smaller datasets

---

## 18. Summary and Future Directions

### 18.1 Architecture Evolution Timeline

```
1980s: Backpropagation, MLPs
1990s: CNNs (LeNet), RNNs
2000s: Deep networks, LSTM
2012: AlexNet (Deep Learning revolution)
2014: GAN, VGG, Inception
2015: ResNet, Attention
2017: Transformer
2018: BERT, GPT
2020: GPT-3, ViT
2022: ChatGPT, Stable Diffusion
2023: GPT-4, LLaMA, Mamba
2024-2025: Efficient architectures, Multi-modal models
```

### 18.2 Current Research Directions

1. **Efficiency:**
   - Sparse models
   - Quantization
   - Distillation
   - Linear-complexity architectures

2. **Multi-Modal:**
   - Unified architectures
   - Cross-modal reasoning
   - Embodied AI

3. **Long Context:**
   - Infinite context windows
   - Efficient attention mechanisms
   - Memory augmentation

4. **Interpretability:**
   - Mechanistic interpretability
   - Attribution methods
   - Causal understanding

5. **Alignment:**
   - RLHF improvements
   - Constitutional AI
   - Value learning

### 18.3 Practical Recommendations

**For Beginners:**
- Start with pre-trained models
- Focus on fine-tuning
- Use standard architectures (ResNet, BERT)

**For Practitioners:**
- Experiment with recent architectures
- Profile your models
- Consider deployment constraints

**For Researchers:**
- Explore hybrid approaches
- Focus on efficiency and scalability
- Benchmark rigorously

---

This documentation covers the major deep learning architectures from foundational to cutting-edge. Each technique has its place, and the best choice depends on your specific task, data, and computational constraints.
