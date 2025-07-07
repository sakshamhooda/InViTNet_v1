# InViNet_v1

An open-source PyTorch implementation of **InViNet: A Dual-branch Vision Transformer based
Novel Architecture for Medical Image Applications**.  
TwT couples texture-driven features obtained from a Central-Difference Convolutional Network (CDCN++) with spatially-rich representations from a Compact Convolutional Transformer (CCT).  The hybrid design delivers *state-of-the-art* accuracy and AUC on the Malaria and BloodMNIST cell-image benchmarks while using **< 6.5 M** parameters and *no* pre-training.

---

## ✨ Key Ideas

1. **Texture Branch – CDCN++**  
   Captures low-, mid- and high-level texture cues via Central Difference Convolutions.  A global average pooled feature vector is passed through a sigmoid FC layer to produce a **256-D weight vector**.
2. **Spatial Branch – CCT-7/3×1**  
   Converts images into patch tokens with a light CNN tokenizer and feeds them to a 7-layer Transformer encoder with sequence pooling.
3. **Texture Injection**  
   Every encoder block multiplies its token embeddings element-wise by the CDCN++ weight vector, enforcing texture awareness.
4. **Dual Loss, Single-branch Inference**  
   Training minimises \(L=L_1+L_2\) (texture + spatial logits).  At inference, only the CCT branch is executed — keeping runtime low.

---

## 📦 Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
*Python ≥ 3.8 and PyTorch ≥ 2.0 are assumed.*

---

## 🗄️ Dataset Preparation

### 1. Malaria Cell Images
```
malaria/
 ├── Parasitized/  *.png
 └── Uninfected/   *.png
```
Resize to **32 × 32** or let the dataloader handle it.  Download link: <https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html>

### 2. BloodMNIST
Auto-downloaded by [`medmnist`](https://medmnist.com/):

```bash
python -c "from medmnist import BloodMNIST; BloodMNIST(split='train', download=True)"
```

---

## 🚀 Training

```bash
# Malaria
python -m src.train --dataset malaria \
                    --data-root /path/to/malaria \
                    --epochs 100 --batch-size 128

# BloodMNIST
python -m src.train --dataset bloodmnist --epochs 30
```
Checkpoints with the best validation AUC are stored in `checkpoints/`.

---

## 🔍 Inference

```bash
python -m src.infer --model-path checkpoints/best_model_malaria.pth \
                    --image ./demo.png --img-size 32 --num-classes 2
```

Returns the predicted class ID and per-class probabilities.

---

## 📊 Reported Results
| Dataset | Accuracy | AUC  | Params |
|---------|---------:|------:|-------:|
| Malaria | **96.84 %** | **0.9941** | 6.3 M |
| BloodMNIST | **92.75 %** | **0.9933** | 6.3 M |

*Matches the paper’s best entries using identical splits and no pre-training.*

---

## 📝 Citation
If you use this repository, please cite the original paper:

```bibtex
@article{Sondhi2023TwT,
  title  = {TwT: A Texture Weighted Transformer for Medical Image Classification and Diagnosis},
  author = {Sondhi, Mrigank and Sharma, Ayush and Malhotra, Ruchika},
  year   = {2023},
  journal= {arXiv preprint arXiv:2304.05704}
}
```

---

## 📄 License
Licensed under the terms of the **MIT License** (see `LICENSE`).

---

## 🔗 References
- A. Hassani *et al.* “Escaping the Big Data Paradigm with Compact Transformers,” 2021.  
- Z. Yu *et al.* “Searching Central Difference Convolutional Networks for Face Anti-Spoofing,” 2020.  
- (See `paper.md` for the complete bibliography.)
