# 📄 Research Paper & Thesis Writing Roadmap

> **Working Directory**: `D:\AKHI THESIS UP\Paper writing`  
> **Master File**: [`main.tex`](main.tex)  
> **Bibliography**: [`references.bib`](references.bib)  
> **Target Format**: IEEE Conference / Journal Double-Column (`IEEEtran`) or Master's Thesis

---

## 📑 Manuscript Structure & Section Manifest

| Section File | Title | Key Contents & Academic Focus | Status |
| :--- | :--- | :--- | :---: |
| [`main.tex`](main.tex) | **Master Document** | Document class, packages, author blocks, abstract, and section inputs. | Ready |
| [`sections/01_introduction.tex`](sections/01_introduction.tex) | **I. Introduction** | Sports action challenges, 2D CNN-LSTM vs 3D CNN vs Transformer evolution, data leakage problem, core contributions. | Ready |
| [`sections/02_related_work.tex`](sections/02_related_work.tex) | **II. Related Work** | Classical/2D methods, 3D ConvNets (I3D, SlowFast, X3D), Video Vision Transformers (ViViT, Swin), and Self-Supervised VideoMAE. | Ready |
| [`sections/03_methodology.tex`](sections/03_methodology.tex) | **III. Methodology** | Spatiotemporal tubelet tokenization ($16\times16\times2$), ViT-B encoder, attention heads, classification head, layer-wise decay, fp16 AMP + gradient checkpointing. | Ready |
| [`sections/04_experimental_setup.tex`](sections/04_experimental_setup.tex) | **IV. Experimental Setup** | 82-class dataset taxonomy, Zero-Leakage Group-Wise Splitting formulation, Kaggle 2x T4 DDP & Colab Drive sync, hyperparameters. | Ready |
| [`sections/05_results_and_discussion.tex`](sections/05_results_and_discussion.tex) | **V. Results & Discussion** | Main benchmark table vs. 4 baselines (Frozen Head, X3D-S, SlowFast-R50, Video Swin-T), CNN vs ViT analysis, confusion matrix breakdown. | Ready |
| [`sections/06_ablation_studies.tex`](sections/06_ablation_studies.tex) | **VI. Ablation Studies** | Temporal frames (8 vs 16), spatial resolution (160 vs 224), fine-tuning depth (frozen vs top-3 vs full), data scaling curve (10% to 100%). | Ready |
| [`sections/07_conclusion.tex`](sections/07_conclusion.tex) | **VII. Conclusion** | Summary of findings, computational takeaways, future multimodal pose extensions. | Ready |
| [`references.bib`](references.bib) | **Bibliography** | Complete BibTeX entries for VideoMAE, Video Swin, X3D, SlowFast, Kinetics, UCF101, and foundation vision papers. | Ready |

---

## 🛠️ How to Compile the Paper

### Method 1: On Overleaf (Recommended)
1. Zip the entire `Paper writing` folder.
2. Go to [Overleaf.com](https://www.overleaf.com/) $\to$ **New Project** $\to$ **Upload Project**.
3. Select the zip file. Overleaf will automatically detect `main.tex` and compile it with 1 click.

### Method 2: Locally via Command Line (TeX Live / MiKTeX)
```bash
cd "D:\AKHI THESIS UP\Paper writing"
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

---

## 📊 Figure & Table Checklist for the Paper

- [x] **Table I**: Main benchmark comparison across all 5 models (Top-1, Top-5, F1, Params, Latency).
- [x] **Table II**: Ablation table for frame count (8 vs 16) and resolution (160 vs 224).
- [x] **Table III**: Ablation table for adaptation depth (Linear probe vs Top-3 vs Full fine-tuning).
- [ ] **Figure 1**: End-to-end architecture diagram of VideoMAE for sports (Tubelet $\to$ ViT $\to$ Head).
- [ ] **Figure 2**: 82-Class Confusion Matrix heatmap.
- [ ] **Figure 3**: Training and validation loss/accuracy convergence curves.
- [ ] **Figure 4**: Spatiotemporal attention rollout visualization on key sports actions.
