# 🔬 Technical Specification: Model Mode M7 (HOI-BiLSTM + Attention)

This document provides the formal mathematical and layer-by-layer architectural specification of the **HOI-BiLSTM Network with Temporal Attention (`Model Mode M7`)** that was trained on your local machine.

---

## 🏗️ Architectural Topology

```
                  ┌────────────────────────────────────────────────────────┐
                  │ Input: Sampled Video Sequence (T=16 frames, 82 classes)│
                  └──────────────────────────┬─────────────────────────────┘
                                             │
      ┌──────────────────────┬───────────────┴──────────────┬──────────────────────┐
      ▼                      ▼                              ▼                      ▼
1. Visual Stream      2. Posture Stream              3. Object Stream       4. HOI Stream
  ResNet-50              Keypoint R-CNN                 Faster R-CNN           Geometric
   (2048-d)                  (85-d)                         (7-d)                (16-d)
      │                      │                              │                      │
      ▼                      ▼                              ▼                      ▼
 [Linear 256]           [Linear 128]                   [Linear 64]            [Linear 64]
 [LayerNorm]            [LayerNorm]                    [LayerNorm]            [LayerNorm]
 [GELU + Drop 0.3]      [GELU + Drop 0.3]              [GELU + Drop 0.3]      [GELU + Drop 0.3]
      │                      │                              │                      │
      └──────────────────────┼──────────────────────────────┼──────────────────────┘
                             ▼
              [Concatenation Layer]  (256 + 128 + 64 + 64 = 512 dimensions)
                             │
                             ▼
                 [Multimodal Fusion Dense Layer]
                     Linear(512, 512) + LayerNorm + GELU + Dropout(0.3)
                             │
                             ▼
                [2-Layer Bidirectional LSTM]
                    Input: 512 | Hidden: 256 per direction
                    Output: 512-d spatiotemporal representation per time step
                             │
                             ▼
               [Temporal Attention Pooling Head]
                    e_t = w^T * tanh(W_att * h_t + b_att)
                    alpha_t = softmax(e_t)
                    c = sum(alpha_t * h_t)  (512-d context vector)
                             │
                             ▼
                   [Classification Head]
                    Linear(512, 256) + LayerNorm + GELU + Dropout(0.3)
                    Linear(256, 82) -> 82 Class Logits
```

---

## 📐 Mathematical Formulation

### 1. Branch Feature Projections
For each time step $t \in \{1, \dots, 16\}$:
\begin{align}
\mathbf{z}_t^{vis} &= \text{Dropout}(\text{GELU}(\text{LN}(\mathbf{W}_{v} \mathbf{x}_t^{vis} + \mathbf{b}_v))) \in \mathbb{R}^{256} \\
\mathbf{z}_t^{pos} &= \text{Dropout}(\text{GELU}(\text{LN}(\mathbf{W}_{p} \mathbf{x}_t^{pos} + \mathbf{b}_p))) \in \mathbb{R}^{128} \\
\mathbf{z}_t^{obj} &= \text{Dropout}(\text{GELU}(\text{LN}(\mathbf{W}_{o} \mathbf{x}_t^{obj} + \mathbf{b}_o))) \in \mathbb{R}^{64} \\
\mathbf{z}_t^{hoi} &= \text{Dropout}(\text{GELU}(\text{LN}(\mathbf{W}_{h} \mathbf{x}_t^{hoi} + \mathbf{b}_h))) \in \mathbb{R}^{64}
\end{align}

### 2. Multimodal Fusion
The projected branch representations are concatenated into a unified multimodal vector $\mathbf{u}_t \in \mathbb{R}^{512}$ and passed through a shared fusion layer:
\begin{equation}
\mathbf{f}_t = \text{Dropout}(\text{GELU}(\text{LN}(\mathbf{W}_f [\mathbf{z}_t^{vis}; \mathbf{z}_t^{pos}; \mathbf{z}_t^{obj}; \mathbf{z}_t^{hoi}] + \mathbf{b}_f))) \in \mathbb{R}^{512}
\end{equation}

### 3. Bidirectional Recurrent Modeling (BiLSTM)
A 2-layer Bidirectional LSTM models temporal forward and backward kinematic evolution:
\begin{align}
\overrightarrow{\mathbf{h}}_t &= \text{LSTM}_{fwd}(\mathbf{f}_t, \overrightarrow{\mathbf{h}}_{t-1}) \\
\overleftarrow{\mathbf{h}}_t &= \text{LSTM}_{bwd}(\mathbf{f}_t, \overleftarrow{\mathbf{h}}_{t+1}) \\
\mathbf{h}_t &= [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{512}
\end{align}

### 4. Temporal Attention Pooling
Rather than taking solely the final hidden state $\mathbf{h}_{16}$, a temporal attention mechanism computes dynamic relevance weights $\alpha_t$ across all 16 frames:
\begin{align}
e_t &= \mathbf{v}_{att}^T \tanh(\mathbf{W}_{att} \mathbf{h}_t + \mathbf{b}_{att}) \\
\alpha_t &= \frac{\exp(e_t)}{\sum_{k=1}^{16} \exp(e_k)} \\
\mathbf{c} &= \sum_{t=1}^{16} \alpha_t \mathbf{h}_t \in \mathbb{R}^{512}
\end{align}

### 5. Final Classification
The temporal context vector $\mathbf{c}$ is mapped to the final 82-class probability distribution:
\begin{equation}
\hat{\mathbf{y}} = \mathbf{W}_{out} \left( \text{Dropout}(\text{GELU}(\text{LN}(\mathbf{W}_{cls} \mathbf{c} + \mathbf{b}_{cls}))) \right) + \mathbf{b}_{out} \in \mathbb{R}^{82}
\end{equation}
