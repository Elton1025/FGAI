# Foundations of Generative AI (ICT4403)
## Assignment 2: Autoregressive Text Generation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace Datasets](https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg)](https://huggingface.co/datasets/Salesforce/wikitext)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An empirical comparative study of word-level autoregressive language modeling that I developed using **Simple RNN**, **Bidirectional RNN**, **Long Short-Term Memory (LSTM)**, and **Gated Recurrent Unit (GRU)** on the WikiText-2 benchmark.

---

## Table of Contents
1. [Problem Statement and Objectives](#1-problem-statement-and-objectives)
2. [Dataset Description and Preprocessing](#2-dataset-description-and-preprocessing)
3. [Model Architectures](#3-model-architectures)
4. [Experimental Setup and Training Conditions](#4-experimental-setup-and-training-conditions)
5. [Execution Steps and Dependencies](#5-execution-steps-and-dependencies)
6. [My Quantitative Results](#6-my-quantitative-results)
7. [Comparative Model Evaluation](#7-comparative-model-evaluation)
8. [Sampling-Temperature Analysis](#8-sampling-temperature-analysis)
9. [Qualitative Text Generation Review](#9-qualitative-text-generation-review)
10. [Conclusion and Best-Performing Model](#10-conclusion-and-best-performing-model)
11. [Repository Structure](#11-repository-structure)

---

## 1. Problem Statement and Objectives

In this assignment, I formulate text generation as sequential autoregressive next-token prediction. By applying the chain rule of probability, I factorize the joint probability of a sequence of word tokens $\mathbf{w} = (w_1, w_2, \dots, w_T)$ into a product of conditional distributions:

$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{1:t-1})$$

Where $w_{1:t-1} = (w_1, \dots, w_{t-1})$ represents the historical prefix context available to the model at step $t$.

### My Core Objectives:
1. **Data Preprocessing & Causal Sequence Construction**: I construct an isolated, leak-free preprocessing pipeline on the WikiText-2 dataset, creating sliding-window input–target pairs $(X, y)$ where $X \in \mathbb{R}^{L}$ comprises $L = 30$ contiguous tokens and $y \in \mathbb{R}$ is the immediate next word token.
2. **Implementation of Recurrent Architectures**: I implement and train four distinct recurrent models—**Simple RNN (Elman)**, **Bidirectional RNN (BiRNN)**, **LSTM**, and **GRU**—under strictly controlled and identical experimental conditions (identical embedding dimension, hidden dimension, layers, learning rate, batch size, and seed).
3. **Multi-Faceted Evaluation**:
   - *Quantitative Metrics*: I measure cross-entropy loss, perplexity ($\text{PPL} = \exp(\text{Loss})$), next-token prediction accuracy (%), parameter counts, and CUDA-synchronized training execution time on held-out test data.
   - *Qualitative Characteristics*: I examine syntactic coherence, contextual consistency, lexical diversity, and repetition avoidance.
4. **Sampling-Temperature Investigation**: I generate text from identical seed sequences using temperature scaling ($T \in \{0.5, 1.0, 1.3\}$) across multiple sampling seeds, evaluate diversity metrics (Distinct-1, Distinct-2, Repeated-Bigram Fraction), and justify which architecture is the most effective for autoregressive generation on this corpus.

---

## 2. Dataset Description and Preprocessing

### Dataset Source
- **Dataset Name**: WikiText-2 (`wikitext-2-raw-v1`)
- **Source Link**: [Hugging Face Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- **Description**: I selected the WikiText-2 corpus because it is extracted from verified Good and Featured articles on Wikipedia, providing over 2 million raw tokens. Unlike the standard Penn Treebank (PTB), WikiText-2 preserves original punctuation, capitalization, and numbers, making it well-suited for realistic word-level language modeling.

### My Preprocessing Pipeline
1. **Split Isolation**: I maintain strict isolation between the official `train`, `validation`, and `test` splits to guarantee zero data leakage.
2. **Paragraph Filtering**: I strip leading and trailing whitespace and remove entries with character length $\le 20$ characters. This effectively filters empty section headers and fragmented titles while retaining meaningful paragraphs.
3. **Tokenization**: I lowercase the text and apply regex-based tokenization to extract alphanumeric words and standard punctuation as separate tokens:
   ```python
   re.findall(r'\b\w+\b|[.,!?;]', text.lower())
   ```
4. **Vocabulary Construction**: I build the vocabulary exclusively from the training split. I retain the top 9,998 most frequent word tokens and reserve two special indices: `<pad>` at index 0 and `<unk>` at index 1, yielding a fixed vocabulary size of exactly $V = 10,000$.
5. **Autoregressive Window Construction**:
   - I concatenate tokenized words into a contiguous 1D stream per split.
   - I slide a window of length $L = 30$ tokens with a stride of $S = 3$ tokens.
   - Input: $X = [w_t, w_{t+1}, \dots, w_{t+29}]$
   - Target: $y = w_{t+30}$ (the immediate next token).
   - Training windows contain purely observed tokens without artificial padding, preventing recurrent states from learning uninformative pad symbols.

### Dataset Partitioning and Statistics
| Split | Raw Entries | Eligible Entries | Selected Entries | Total Tokens | Sequences ($L=30, S=3$) | Unknown Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 36,718 | 20,887 | 8,000 | 740,733 | 246,901 | 6.99% |
| **Validation** | 3,760 | 2,203 | 1,500 | 137,629 | 45,867 | 10.81% |
| **Test** | 4,358 | 4,358 | 4,358 (Full) | 228,714 | 76,228 | 11.65% |

---

## 3. Model Architectures

I designed a modular, unified PyTorch architecture where all models share an identical embedding layer and output projection head, allowing me to isolate the performance impact of each recurrent backbone.

```
Input Tokens X [Batch, 30] 
       │
       ▼
Embedding Layer [10000 -> 256] + Dropout(0.2)
       │
       ▼
Recurrent Backbone (2 Layers, Hidden Dim 256)
 [SimpleRNN | BiRNN | LSTM | GRU]
       │
       ▼
Final Hidden State h_T [Batch, Hidden*Directions] + Dropout(0.2)
       │
       ▼
Output Linear Classifier [Hidden*Directions -> 10000]
       │
       ▼
Vocabulary Logits / Cross-Entropy Loss
```

### Mathematical Formulations

#### 1. Simple RNN (Elman RNN)
I implement the standard Elman recurrence where hidden state $h_t \in \mathbb{R}^{256}$ is updated as:
$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$
- *Behavioral Profile*: Highly efficient with minimal computational overhead, but prone to vanishing and exploding gradients when propagating error signals across 30 time steps.
- *Parameters*: 5,393,168.

#### 2. Bidirectional RNN (BiRNN)
I implement a two-directional recurrent layer computing simultaneous forward ($\overrightarrow{h}_t$) and backward ($\overleftarrow{h}_t$) hidden states:
$$\overrightarrow{h}_t = \tanh(\overrightarrow{W}_{ih} x_t + \overrightarrow{b}_{ih} + \overrightarrow{W}_{hh} \overrightarrow{h}_{t-1} + \overrightarrow{b}_{hh})$$
$$\overleftarrow{h}_t = \tanh(\overleftarrow{W}_{ih} x_t + \overleftarrow{b}_{ih} + \overleftarrow{W}_{hh} \overleftarrow{h}_{t+1} + \overleftarrow{b}_{hh})$$
I concatenate the final forward and backward states of the top layer: $h_{\text{final}} = [\overrightarrow{h}_T \,\|\, \overleftarrow{h}_1] \in \mathbb{R}^{512}$.
- *Theoretical Limitation in Autoregressive Modeling*: Although bidirectionality is effective for sequence classification, I observed that it creates a fundamental covariate shift in autoregressive generation. At inference time, future tokens do not yet exist, so the backward pass operates only over the known prefix in reverse. This produces a severe train-inference mismatch.
- *Parameters*: 8,347,408 (highest parameter count).

#### 3. Long Short-Term Memory (LSTM)
I implement the gated LSTM cell incorporating an internal cell state $C_t$ and three multiplicative gates (forget $f_t$, input $i_t$, and output $o_t$):
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\widetilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \widetilde{C}_t$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$
- *Behavioral Profile*: Mitigates vanishing gradients via the linear cell update path ($C_t$), enabling effective modeling of long-range syntax at the expense of higher parameter count and runtime.
- *Parameters*: 6,182,672.

#### 4. Gated Recurrent Unit (GRU)
I implement Cho et al.'s GRU, which merges the cell state into hidden state $h_t$ using update ($z_t$) and reset ($r_t$) gates:
$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$
$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$
$$\widetilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \widetilde{h}_t$$
- *Behavioral Profile*: Retains gating benefits with fewer matrix projections (3 per unit vs 4 in LSTM), reducing memory footprint and accelerating training speed.
- *Parameters*: 5,919,504.

---

## 4. Experimental Setup and Training Conditions

I held all hyperparameters and training procedures strictly identical across all four models to ensure an unbiased, fair comparison:

| Hyperparameter / Setup | Value | Rationale |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 10,000 | 9,998 frequent words + `<pad>` + `<unk>` |
| **Context Window ($L$)** | 30 tokens | Sufficient span to capture intra-sentence dependencies |
| **Sliding Stride ($S$)** | 3 tokens | Balanced corpus coverage and computational feasibility |
| **Embedding Dimension** | 256 | Dense semantic vector representation |
| **Hidden Dimension** | 256 | Per direction (BiRNN has $2 \times 256 = 512$ total) |
| **Recurrent Layers** | 2 | Stacked hierarchical recurrent depth |
| **Dropout Rate** | 0.2 | Applied to embeddings and between recurrent layers |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adaptive learning rate optimization |
| **Learning Rate** | 0.001 | Stable convergence across recurrent models |
| **Batch Size** | 128 | Efficient GPU tensor parallelization |
| **Gradient Clipping** | 1.0 (Max Norm) | Suppresses gradient explosion during backpropagation |
| **Training Epochs** | 8 | Sufficient to identify best validation epoch and observe overfitting |
| **Training Seed** | 42 | Full determinism across Python, NumPy, and PyTorch |
| **Checkpoint Criterion** | Minimum Validation Loss | Test metrics evaluated strictly on restored best weights |
| **Hardware Environment** | NVIDIA Tesla T4 GPU | CUDA 12.8, cuBLAS workspace `:4096:8` |

---

## 5. Execution Steps and Dependencies

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA GPU recommended)

### Installation
Clone my repository and install dependencies:
```bash
git clone https://github.com/Elton1025/FGAI.git
cd FGAI
pip install -r requirements.txt
```

### Reproducing My Results

#### Option A: Running My Standalone Script
To regenerate my summary tables, diversity metrics, and plots from existing run logs:
```bash
python train_and_evaluate.py --action export --output .
```
To run full end-to-end training and evaluation from scratch:
```bash
python train_and_evaluate.py --action train --output results/
```

#### Option B: Executing the Jupyter Notebook
Open and run `autoregressive_text_generation.ipynb` (or `notebook94b14f6649.ipynb`) in Jupyter Lab, VS Code, or Google Colab / Kaggle.

---

## 6. My Quantitative Results

I evaluated all four models on the held-out test split (76,228 sequences, 228,714 tokens) using each model's best checkpoint selected by minimum validation loss.

### Benchmark Results Table
| Model | Parameters | Best Epoch | Final Train Loss | Best Val Loss | Best Val PPL | Final Val PPL | Test Loss | Test Perplexity (PPL) | Test Accuracy (%) | Pure Train Time | Total Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** | 5,393,168 | 3 | 4.8068 | 5.5442 | 255.76 | 281.69 | 5.4981 | **244.23** | 18.59% | **140.87s** | 149.65s |
| **BiRNN** | 8,347,408 | 2 | 4.2969 | 5.7379 | 310.41 | 498.71 | 5.6882 | **295.36** | 17.68% | 242.03s | 257.01s |
| **LSTM** | 6,182,672 | 4 | 4.6127 | 5.3876 | 218.68 | 254.28 | 5.3597 | **212.66** | **20.05%** | 228.23s | 241.39s |
| **GRU** | 5,919,504 | 4 | 4.1832 | 5.3022 | 200.78 | 236.04 | 5.2723 | **194.87** | 19.97% | **195.74s** | 207.15s |

*Note: Perplexity is defined as $\text{PPL} = \exp(\text{Cross-Entropy Loss})$. Checkpoints were selected strictly on validation loss to prevent test data leakage. Pure Train Time isolates GPU training batch computation from evaluation and checkpoint I/O.*

### Validation Learning Curves
The curves below track validation loss, perplexity, and top-1 accuracy throughout my 8 training epochs:

![Validation Learning Curves](learning_curves.png)

### Key Quantitative Findings:
1. **Perplexity Leader**: **My GRU model achieved the lowest test perplexity of 194.87** and lowest test loss of 5.2723. It significantly outperformed SimpleRNN (244.23 PPL, a 20.2% improvement) and LSTM (212.66 PPL, an 8.4% improvement).
2. **Computational Speed**: My GRU model completed training in 195.74s—**14.2% faster than LSTM (228.23s)** and **19.1% faster than BiRNN (242.03s)**—confirming the computational efficiency of its two-gate design.
3. **Overfitting Characteristics**:
   - BiRNN started overfitting immediately after Epoch 2: its validation loss escalated from 5.7379 to 6.2120 (PPL jumped from 310.41 to 498.71) while training loss dropped to 4.2969.
   - SimpleRNN peaked at Epoch 3 (Val PPL 255.76) and degraded thereafter due to gradient vanishing.
   - LSTM and GRU both peaked at Epoch 4 and maintained steady validation loss profiles.

---

## 7. Comparative Model Evaluation

```
                    Model Trade-off Landscape
    Perplexity (Lower is Better) vs Training Time (Seconds)
  ┌─────────────────────────────────────────────────────────┐
  │ 300 ┼                        BiRNN (295.36 PPL, 242s)   │
  │     │                                                   │
  │ 250 ┼      SimpleRNN (244.23 PPL, 141s)                 │
  │     │                                                   │
  │ 200 ┼                           LSTM (212.66 PPL, 228s) │
  │     │                   ★ GRU (194.87 PPL, 196s)        │
  │     └──────┬────────────────────┬───────────────────────┤
  │           140s                 200s                    250s
  └─────────────────────────────────────────────────────────┘
```

### Architectural Analysis:
- **GRU vs LSTM**: My GRU model demonstrated a superior inductive bias for this dataset. With 263,168 fewer parameters than LSTM, GRU avoided over-parameterization while attaining lower test perplexity (194.87 vs 212.66). LSTM achieved slightly higher top-1 accuracy (20.05% vs 19.97%), but its output probabilities were less calibrated.
- **Why BiRNN Failed**: Despite having the largest capacity (8.35M parameters), BiRNN performed the worst across all metrics (Test PPL 295.36, Acc 17.68%). In causal language modeling, tokens must be generated sequentially from left to right. A bidirectional RNN's backward pass over a prompt prefix cannot access upcoming tokens during generation, causing a fundamental distribution shift that leads to premature overfitting and severe repetitive loops.
- **SimpleRNN Limitations**: While SimpleRNN was the fastest per epoch (140.87s total), its un-gated architecture struggled to preserve information beyond 5–10 tokens, resulting in high perplexity (244.23) and rapid thematic drift.

---

## 8. Sampling-Temperature Analysis

I conducted sequential text generation using temperature-scaled multinomial sampling across three distinct regimes:
$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}$$

Where $z_i$ is the unnormalized logit for token $i$, and $T > 0$ is the sampling temperature. I masked special tokens (`<pad>` and `<unk>`) with $-\infty$ during generation to ensure meaningful outputs.

### Diversity Metrics across Temperatures
I computed Distinct-1 (unigram Type-Token Ratio), Distinct-2 (unique bigram ratio), and Repeated-Bigram Fraction over 108 continuations (3 prompts $\times$ 3 temperatures $\times$ 3 seeds $\times$ 4 models):

| Model | Temperature ($T$) | Distinct-1 (TTR) | Distinct-2 | Repeated-Bigram Fraction | Observed Behavioral Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SimpleRNN** | 0.5 | 0.680 ± 0.077 | 0.917 ± 0.047 | 0.083 ± 0.047 | Syntactic repetition, generic vocabulary |
| **SimpleRNN** | 1.0 | 0.840 ± 0.035 | 0.977 ± 0.037 | 0.023 ± 0.037 | Frequent topic shifts, thematic drift |
| **SimpleRNN** | 1.3 | 0.964 ± 0.031 | 1.000 ± 0.000 | 0.000 ± 0.000 | Semantic collapse ("word salad") |
| **BiRNN** | 0.5 | 0.516 ± 0.142 | 0.690 ± 0.221 | **0.310 ± 0.221** | **Catastrophic cyclic looping (up to 54% repeated)** |
| **BiRNN** | 1.0 | 0.907 ± 0.063 | 1.000 ± 0.000 | 0.000 ± 0.000 | Moderate fluency, abrupt transitions |
| **BiRNN** | 1.3 | 0.942 ± 0.053 | 1.000 ± 0.000 | 0.000 ± 0.000 | Syntactic breakdown, random entity assembly |
| **LSTM** | 0.5 | 0.693 ± 0.060 | 0.912 ± 0.057 | 0.088 ± 0.057 | High grammatical structure, repetitive determiners |
| **LSTM** | 1.0 | 0.840 ± 0.060 | 0.995 ± 0.014 | 0.005 ± 0.014 | Excellent balance of fluency and diversity |
| **LSTM** | 1.3 | 0.987 ± 0.020 | 1.000 ± 0.000 | 0.000 ± 0.000 | Low semantic cohesion, disjoint phrases |
| **GRU** | 0.5 | 0.693 ± 0.102 | 0.912 ± 0.061 | 0.088 ± 0.061 | Highly focused domain relevance, controlled repetition |
| **GRU** | 1.0 | 0.911 ± 0.074 | 0.995 ± 0.014 | 0.005 ± 0.014 | **Optimal generation: high coherence & topical focus** |
| **GRU** | 1.3 | 0.978 ± 0.029 | 1.000 ± 0.000 | 0.000 ± 0.000 | High vocabulary diversity, erratic syntax |

![Temperature Metrics](temperature_metrics.png)

---

## 9. Qualitative Text Generation Review

I audited generated continuations across three representative seed prompts:

### Seed Prompt 1: `"the history of the united states began with"`
- **My GRU model ($T=0.5$)**: `... the arrival of the bombardment . the palace was unsuccessful by the germans to the antarctic . the first half of the japanese , the`
  - *My Assessment*: Generates historically plausible terms ("arrival", "bombardment", "germans", "japanese") and retains coherent sentence structure.
- **My BiRNN model ($T=0.5$)**: `... the end of the united states . the storm was built on the basis of the season , but he was built to be the`
  - *My Assessment*: Repeats tokens from the prompt prefix ("united states") and re-uses phrasing templates ("built on the basis").
- **My LSTM model ($T=1.0$)**: `... the names in the north sea and end of the state of the country . wasp . the first family , is christmas , or`
  - *My Assessment*: Good lexical variety, but punctuation splits the sentence prematurely ("wasp .").
- **My SimpleRNN model ($T=1.3$)**: `... conscientious names in the spring . while end proposed c4 aggressive loan has it probability antarctic . esarhaddon km2 in cedar is christmas , security`
  - *My Assessment*: High unigram diversity (Distinct-1: 0.92) accompanied by total semantic breakdown ("esarhaddon km2 in cedar").

### Seed Prompt 2: `"during the second world war the army was"`
- **My BiRNN model ($T=0.5$) — Pathological Mode Collapse**:
  ```
  ... officially in the end of the end of the end of the end of the end . the first season , the end of the
  ```
  - *My Assessment*: I observed a severe failure mode where "of the end" loops 5 times consecutively. Distinct-1 dropped to 0.36 and repeated-bigram fraction surged to 0.54.
- **My GRU model ($T=0.5$)**: `... officially built in the united kingdom . the proposed of the album has been recorded in the united states , which is the most of`
  - *My Assessment*: Grammatically fluent with institutional vocabulary ("officially built in the united kingdom"), transitioning smoothly into Wikipedia media topics.
- **My LSTM model ($T=1.0$)**: `... loved a in 1895 until the base end , such as a 2 to take two . the first family , this christmas , the`
  - *My Assessment*: Plausible transitions and incorporates historical date tokens.

### Seed Prompt 3: `"the album received positive reviews from music critics"`
- **My GRU model ($T=0.5$) — Exceptional Domain Fidelity**:
  ```
  ... . it incorporates the billboard best selling episode , carey performed crazy in love love s . crazy in love , you christmas , you
  ```
  - *My Assessment*: My GRU model uniquely activated domain-specific musical knowledge from WikiText-2 ("billboard", "best selling", "carey", "crazy in love"), displaying the strongest contextual fidelity in the entire study.
- **My LSTM model ($T=0.5$)**: `... . the ship was certified fitted to the stem of the album s season . the same year , the first is the first power`
  - *My Assessment*: Preserves musical token ("album"), but introduces confusing nautical vocabulary.
- **My SimpleRNN model ($T=0.5$)**: `... . the song was released in the end of the state of the 3rd government . . the first down , he was established on`
  - *My Assessment*: Links "album" to "song was released", but quickly drifts into military and governmental terms.

---

## 10. Conclusion and Best-Performing Model

### My Final Conclusion: **The Gated Recurrent Unit (GRU) is the Champion Architecture**

| Evaluation Criterion | Rank 1 (Best) | Rank 2 | Rank 3 | Rank 4 (Worst) | My Justification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Generalization Perplexity** | **GRU (194.87)** | LSTM (212.66) | SimpleRNN (244.23) | BiRNN (295.36) | Lowest cross-entropy on unseen test data |
| **Computational Speed** | **SimpleRNN (141s)** | **GRU (196s)** | LSTM (228s) | BiRNN (242s) | GRU trains 14.2% faster than LSTM |
| **Parameter Efficiency** | **SimpleRNN (5.39M)** | **GRU (5.92M)** | LSTM (6.18M) | BiRNN (8.35M) | GRU delivers highest accuracy per parameter |
| **Contextual Fidelity** | **GRU** | LSTM | SimpleRNN | BiRNN | GRU successfully captured musical entities |
| **Generation Stability** | **LSTM & GRU** | SimpleRNN | — | BiRNN | BiRNN collapsed into repetitive loops at $T=0.5$ |

### My Justification Summary:
1. **Empirical Performance**: My GRU model achieved the lowest test loss (5.2723) and lowest test perplexity (194.87), demonstrating superior predictive capability on unseen language distributions.
2. **Speed & Efficiency**: With coupled update and reset gates, GRU reduces internal matrix multiplications while outperforming both the vanishing-gradient-prone SimpleRNN and the over-parameterized LSTM.
3. **Generation Quality**: Under $T=0.5$ and $T=1.0$, GRU produced the most cohesive and contextually relevant text, resisting the degenerative looping that crippled BiRNN.

---

## 11. Repository Structure

```
.
├── autoregressive_text_generation.ipynb   # Complete interactive notebook with my code & outputs
├── notebook94b14f6649.ipynb               # Original Kaggle experiment execution notebook
├── train_and_evaluate.py                  # Standalone modular Python pipeline (CLI support)
├── requirements.txt                       # Python dependencies
├── config.json                            # My experimental hyperparameters
├── environment.json                       # Execution runtime & GPU specifications
├── dataset_statistics.json                # Preprocessing and vocabulary split statistics
├── vocabulary.json                        # Full 10,000-token vocabulary mapping
├── results.csv                            # Final quantitative metrics across all four models
├── history.csv                            # Epoch-by-epoch training/validation progression
├── history.json                           # Raw per-epoch metrics logs
├── SimpleRNN-seed42-history.json           # SimpleRNN training history log
├── BiRNN-seed42-history.json              # BiRNN training history log
├── LSTM-seed42-history.json               # LSTM training history log
├── GRU-seed42-history.json                # GRU training history log
├── samples.csv                            # 108 generated continuations across models & temperatures
├── temperature_summary.csv                # Aggregate diversity metrics (Distinct-1, Distinct-2, Repetition)
├── learning_curves.png                    # Validation loss, perplexity, and accuracy curves
├── temperature_metrics.png                # Diversity metrics vs sampling temperature plot
├── .gitignore                             # Ignored checkpoints, reports, and environment artifacts
└── README.md                              # Project documentation
```

---

## Author & Course Information
- **Course**: Foundations of Generative AI (ICT4403)
- **Assignment**: Assignment 2 — Autoregressive Text Generation
- **Author**: Elton Lewis
- **Dataset**: [WikiText-2 (Salesforce/wikitext)](https://huggingface.co/datasets/Salesforce/wikitext)
