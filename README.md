# Ticket Classification System

A machine learning system for automatically classifying utility service tickets using DistilBERT. This system can predict ticket categories like vegetation issues, damaged poles, flooding, and other utility-related problems from ticket comments.

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Git (optional, for cloning)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/RithikaMathew/ticketFPL.git
   cd ticketFPL-1
   ```

2. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **You're ready to go!** The system comes with a pre-trained model in the `saved_model_crosscheck/` directory.

## 📊 Usage Options

### Option 1: Python Script (Recommended for Production)
```bash
python3 ticket.py
```
- **Input**: Expects `Milton.xlsx` in the project directory
- **Output**: `Milton_preprocessed_with_predictions.xlsx`

### Option 2: Jupyter Notebook (Recommended for Training/Development)
```bash
jupyter notebook ticket.ipynb
```
- Use this for model training, experimentation, and detailed analysis
- Contains both training and prediction workflows

## 📁 Input File Requirements

### Excel File Format
Your input Excel file must contain these columns:

| Column Name | Description | Required |
|-------------|-------------|----------|
| `Dispatch Center Comments` | Comments from dispatch center | Yes |
| `Service Center Comments` | Comments from service center | Yes |
| `CI` | Customer Impact (number of customers affected) | Yes |
| `County Name` or `Franchise Name` | Location information | Yes |

### Sample Input Data Format
```
Ticket Number | Dispatch Center Comments | Service Center Comments | CI
TKT001       | Tree down on power line  | Crew dispatched         | 150
TKT002       | Flooding near substation | Water pumps needed      | 500
TKT003       | Damaged utility pole     | Pole replacement req    | 75
```

### Supported File Formats
- `.xlsx` (Excel format) - Primary supported format
- `.xls` (Legacy Excel) - Also supported

## 📤 Output Files

### 1. Preprocessed Data (`Milton_preprocessed.xlsx`)
Contains the cleaned and processed input data with:
- `merged_comments`: Combined dispatch and service center comments
- `clean_comments`: Preprocessed text for ML model
- `unknown_words`: Detected abbreviations and unknown terms

### 2. Predictions (`Milton_preprocessed_with_predictions.xlsx`)
Final output with ML predictions:
- All original columns
- `predicted_crosscheck`: Predicted categories (e.g., "vegetation", "damaged pole", "flooding")
- Optional: `prediction_confidences`: Confidence scores for each prediction

### 3. Visualization Files (if using notebook)
- `multilabel_comprehensive_evaluation.png`: Model performance metrics
- Various analysis charts showing category distributions and performance

## 🏷️ Prediction Categories

The model can predict these ticket categories:

| Category | Description | Example Keywords |
|----------|-------------|------------------|
| **Vegetation** | Tree-related issues | tree, limb, vegetation, pruning |
| **Damaged Pole** | Utility pole problems | pole damaged, pole down, pole hit |
| **Flooding** | Water-related issues | flood, water, storm surge |
| **Other** | General utility issues | equipment failure, maintenance |

### Multi-label Predictions
- Tickets can have multiple categories (e.g., "vegetation, damaged pole")
- Categories are comma-separated in the output

## 🔧 Advanced Configuration

### Custom Input Files
To use a different input file, modify the file path in `ticket.py`:
```python
input_file = Path("your_file_name.xlsx")  # Change this line
```

### Model Retraining
If you want to train a new model:

1. **Prepare training data** with a `crosscheck` column containing correct labels
2. **Use the Jupyter notebook** `ticket.ipynb`
3. **Run the training cells** to create a new model
4. **The new model** will be saved to `saved_model_crosscheck/`

### Training Data Format
For training, your Excel file needs an additional column:
- `crosscheck`: Ground truth labels (e.g., "vegetation", "damaged pole", "flooding")

## 🛠️ Troubleshooting

### Common Issues

**Error: "No module named 'pandas'"**
```bash
pip3 install -r requirements.txt
```

**Error: "File 'Milton.xlsx' not found"**
- Make sure your Excel file is in the project directory
- Check the filename matches exactly (case-sensitive)
- Verify the file is not corrupted

**Error: "Column 'Dispatch Center Comments' not found"**
- Check your Excel file has the required column names
- Column names are case-sensitive and must match exactly

**Memory Issues with Large Files**
- The system can handle 35,000+ rows efficiently
- For very large files (100k+ rows), consider processing in batches

### Performance Tips

**For Large Datasets:**
- Run on a machine with at least 8GB RAM
- GPU acceleration is supported but not required
- Processing time: ~1 second per 100 rows

**Model Accuracy:**
- The system achieves ~85-90% accuracy on utility tickets
- Performance varies based on comment quality and completeness

## 📈 System Architecture

```
Input Excel File
      ↓
Text Preprocessing (Clean comments, merge fields)
      ↓
DistilBERT Model (Pre-trained transformer)
      ↓
Multi-label Classification (Predict categories)
      ↓
Output Excel File (With predictions)
```

## 🤝 Support

### Getting Help
1. **Check this README** for common solutions
2. **Review error messages** carefully - they usually indicate the specific issue
3. **Verify input file format** matches requirements

### Sample Files
- `Milton.xlsx`: Example input file format
- `saved_model_crosscheck/`: Pre-trained model files
- `requirements.txt`: Required Python packages

## 📝 File Structure

```
ticketFPL-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── ticket.py                         # Main Python script
├── ticket.ipynb                      # Jupyter notebook (training + prediction)
├── Milton.xlsx                       # Sample input file
├── saved_model_crosscheck/           # Pre-trained model directory
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── categories.pkl
│   ├── mlb.pkl
│   └── thresholds.pkl
└── output files (generated after running)
    ├── Milton_preprocessed.xlsx
    └── Milton_preprocessed_with_predictions.xlsx
```

## 🔬 Technical Details

### Model Information
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Multi-label text classification
- **Training**: 5-fold cross-validation with early stopping
- **Optimization**: Per-class threshold optimization for imbalanced data

### Text Preprocessing
- Lowercasing and punctuation removal
- Utility-specific abbreviation expansion (e.g., "tx" → "transformer")
- Unknown word detection using NLTK + custom utility vocabulary

### Performance Metrics
- **F1-Score (Micro)**: ~0.87
- **Exact Match Accuracy**: ~0.83
- **Hamming Loss**: <0.15

## 📄 License

This project is part of utility infrastructure analysis. Please ensure appropriate data privacy and security measures when processing real utility data.

---

**Need more help?** Check the Jupyter notebook (`ticket.ipynb`) for detailed examples and step-by-step explanations of the entire workflow.