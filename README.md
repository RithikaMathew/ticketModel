# REAL WORLD DATASET: Domain-Adapted Multi-Label Classification for Power Distribution Event Tickets

## Overview
Ticket is a research-grade pipeline for classifying power distribution event tickets using domain-adapted DistilBERT and advanced multi-label techniques. The project includes:
- Domain adaptation of DistilBERT using power engineering textbooks (PDFs)
- Custom multi-label classifier with class imbalance mitigation
- K-Fold cross-validation, threshold optimization, and ensemble baselines
- Comprehensive evaluation metrics and visualizations
- Production-ready model export and inference

## Features
- **Domain Adaptation**: Fine-tunes DistilBERT on extracted text from power engineering PDFs
- **Multi-Label Classification**: Assigns multiple categories to event tickets (damaged pole, flooding, vegetation, other)
- **Class Imbalance Handling**: Weighted BCE loss, per-class threshold optimization
- **Evaluation**: Exact match accuracy, F1-micro/macro, Jaccard, AUPRC, MCC, LRAP, coverage error
- **Baselines**: ML Models
- **Visualizations**: Confusion matrices, error analysis, co-occurrence heatmaps

## Directory Structure
```
├── distilbert-domain-adapted-final/   # Domain-adapted DistilBERT encoder from domain_adaptation.ipynb
├── results/                          # Evaluation metrics, thresholds, baselines
├── saved_model_crosscheck/           # Final multi-label DistilBERT classifier and tokenizer using distilbert-domain-adapted-final
├── visualizations/                   # Plots and figures
├── utility_pdfs/                     # Source PDFs for domain adaptation
├── DistilbertTrain.ipynb             # Main training notebook
├── domain_adaptation.ipynb           # Domain adaptation notebook
├── RESEARCH_PAPER_METHODOLOGY.md     # Full methodology and results
├── README.md                         # Project overview and instructions
```

## Getting Started
1. **Clone the repository:**
   ```
   git clone https://github.com/RithikaMathew/ticketModel.git
   cd ticketModel
   ```
2. **Set up environment:**
   - Use Conda or Python venv (see below)
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
3. **Run notebooks:**
   - `domain_adaptation.ipynb`: Extract and clean PDF text, fine-tune DistilBERT encoder
   - `DistilbertTrain.ipynb`: Train multi-label classifier, evaluate, visualize

## Environment Setup
- **Conda:**
  ```
  conda create -n ticket python=3.12
  conda activate ticket
  pip install -r requirements.txt
  ```
- **Virtualenv:**
  ```
  python -m venv ticket
  .\ticket\Scripts\activate
  pip install -r requirements.txt
  ```

## Model Export & Inference
- Final models are saved in `saved_model_crosscheck/` and `distilbert-domain-adapted-final/`
- Use HuggingFace Transformers to load and run inference

## Citation

Rithika Mathew, Siyuan Du, Mahdi Zarif, Bruce Stephen, James VanZwieten, and Yufei Tang.  
**Improving Power Utility Ticket Processing via Domain-Adaptive Transfer Learning and Large Language Models.**  
*IEEE Transactions on Power Delivery*, 2025 (Under review).
