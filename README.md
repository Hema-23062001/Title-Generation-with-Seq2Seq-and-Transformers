# 📘 Title Generation with Seq2Seq Models

This repository contains the implementation and analysis of **title generation for Wikipedia articles** using both **RNN-based Seq2Seq architectures** and **state-of-the-art Transformer models**.  

The project is divided into three parts:  
- **Part A (Preprocessing.ipynb):** Dataset preparation and preprocessing  
- **Part B (RNN_Models.ipynb):** RNN Seq2Seq models with various enhancements  
- **Part C (Transformer_models.ipynb):** Transformer-based models (T5 fine-tuning and Flan-T5 zero-shot prompting)  

---

##  Repository Structure

```
├── Preprocessing.ipynb   # Preprocessing pipeline (cleaning, stopword removal, stemming, lemmatization, stats)
├── RNN_Models.ipynb   # Seq2Seq RNN models (Basic, GloVe, Hierarchical Encoder, Double Decoder, Beam Search)
├── Transformer_models.ipynb   # Transformer models (T5 fine-tuning, Flan-T5 zero-shot with prompt engineering)
├── REPORT.pdf # Detailed project report with results and analysis
└── README.md     # Project documentation
```

---

## Dataset
- Source: Wikipedia articles  
- **Training set:** 13,379 articles  
- **Validation set:** 500 articles  
- **Test set:** 100 articles  

### Preprocessing Steps
1. Train-validation split  
2. Punctuation & non-ASCII removal  
3. Stopword removal  
4. Stemming  
5. Lemmatization  

Stopword removal gave the biggest impact, reducing text length by **40–50%**.

---

## Implemented Models

### 🔹 RNN Seq2Seq (Part B)
- Basic Encoder–Decoder (GRU-based)  
- GloVe Embedding integration (300-d pre-trained vectors)  
- Hierarchical Encoder (word-level + sentence-level GRUs)  
- Double Decoder (stacked GRUs for deeper decoding)  
- Beam Search Decoding  

### 🔹 Transformer Models (Part C)
- **T5 Fine-tuning** (`google-t5/t5-small`)  
- **Flan-T5 Zero-shot prompting** (base and large models) with **prompt variations**  

---

##  Results

| Model Variant | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------------|---------|---------|---------|
| Basic RNN (Greedy) | 0.3591 | 0.0870 | 0.3591 |
| T5 Fine-tuned | 0.7822 | 0.6007 | 0.7822 |
| Flan-T5 Large (Prompt 1, Beam) | **0.8255** | **0.6456** | **0.8240** |

**Key Insights:**
- Simple **RNN Seq2Seq** surprisingly outperformed enhanced RNN variants.  
- **Transformers massively outperformed RNNs**, even in zero-shot settings.  
- **Flan-T5 Large + Prompt Engineering** beat the fine-tuned T5 model.  
- **Prompt choice & model size** are critical for Transformer performance.  

---

##  Getting Started

### 1. Clone this repo
```bash
git clone https://github.com/yourusername/Title-Generation-with-Seq2Seq-and-Transformers.git
cd Title-Generation-with-Seq2Seq-and-Transformers
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Main dependencies:**
- `torch`
- `transformers`
- `nltk`
- `pandas`
- `numpy`
- `matplotlib`

### 3. Run Notebooks
- **Preprocessing:** `Preprocessing.ipynb`  
- **Seq2Seq Models:** `RNN_Models.ipynb`  
- **Transformer Models:** `Transformer_models.ipynb`  

---

##  Future Work
- Test larger models (Flan-T5-XL, Flan-T5-XXL)  
- More sophisticated **prompt engineering**  
- Hybrid fine-tuning + instruction tuning approaches  
- Human evaluation beyond ROUGE metrics  
- Explore **regularization** for RNNs  
- Build domain-specific embeddings trained on Wikipedia articles  

---

## Author
**Hema R**  
Department of Computer Science & Engineering  
Indian Institute of Technology, Kharagpur  
