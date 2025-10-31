"""# 1. Test Model on New Data (Excel)

### A. Pre-process datset= clean and merge comments
"""

import pandas as pd
import re
import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import nltk
from nltk.corpus import words # Keep NLTK words
from nltk.stem import WordNetLemmatizer # Keep NLTK Lemmatizer
# import enchant # Remove dependency on enchant

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading NLTK word corpus...")
    nltk.download('words')

try:
    nltk.data.find('corpora/wordnet') # Need wordnet for lemmatizer
except LookupError:
    print("Downloading NLTK WordNet corpus...")
    nltk.download('wordnet')


# Initialize WordNet lemmatizer for handling inflected forms
lemmatizer = WordNetLemmatizer()

# Create English words set from NLTK
english_words_nltk = set(words.words())

# Custom utility-specific terms
utility_terms = {
    'tx', 'pm', 'sec', 'primary', 'neutral', 'fiberglass', 'spans',
    'pole', 'poles', 'class', 'ft', 'feet', 'foot', 'kv', 'voltage', 'amp', 'amps',
    'wire', 'wires', 'line', 'lines', 'transformer', 'circuit', 'breaker',
    'utility', 'electric', 'electrical', 'power', 'outage', 'restoration',
    'crew', 'crews', 'tech', 'technician', 'engineer', 'supervisor',
    'residential', 'commercial', 'industrial', 'customer', 'customers',
    'repair', 'replace', 'install', 'maintenance', 'inspection',
    'underground', 'overhead', 'distribution', 'transmission',
    'substation', 'feeder', 'lateral', 'tap', 'splice', 'connection',
    'prott', 'tt', 'veg', 'tkt', 'fpl', 'sarasota', 'dr', 'kwh', 'cust', 'rd', 'nw', 'tkt', 'mtr', 'ok', 'cust', 'jon', 'vreeland' # Add more common abbreviations
}

# Combine NLTK words and utility terms for the known words set
known_words = english_words_nltk.union(utility_terms)

# ====================
# Step 1: Load dataset and initial analysis
# ====================

input_file = Path("Milton.xlsx")  # original/uncleaned test file
df = pd.read_excel(input_file)

print(f"Original dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for duplicates - show examples but don't remove
duplicates = df[df.duplicated(keep=False)]
if len(duplicates) > 0:
    print(f"\nFound {len(duplicates)} duplicate rows (showing first 3):")
    print(duplicates.head(3))
    print("Note: Keeping duplicates as requested")
else:
    print(f"No duplicate rows found in original data")

# ====================
# Step 2: Merge comments and remove empty rows
# ====================
df["merged_comments"] = df[["Dispatch Center Comments", "Service Center Comments"]].fillna("").agg(" ".join, axis=1)

# Remove rows where merged_comments is empty
print(f"\nShape before removing empty comments: {df.shape}")
df = df[df["merged_comments"].str.strip() != ""].reset_index(drop=True)
print(f"Shape after removing empty comments: {df.shape}")

# Remove the original comment columns
df = df.drop(columns=["Dispatch Center Comments", "Service Center Comments"])

print(f"Processing {len(df)} rows with valid comments...")

# ====================
# Step 3: Preprocessing and Unknown Words Detection (Revised)
# ====================
def preprocess_vectorized(texts):
    """Vectorized preprocessing using pandas string methods"""
    texts = texts.astype(str).str.lower()
    texts = texts.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()
    return texts

def find_unknown_words_revised(text):
    """Find words that are not in known_words set, including lemmatization check"""
    # Clean text and split into words
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text).lower())
    words_in_text = set(clean_text.split())

    unknown = []
    for word in words_in_text:
        if (len(word) > 1 and
            not word.isdigit() and
            not re.match(r'^\d+[a-z]*$', word)):  # Exclude numbers and patterns like "35ft"

            # Check if word is in known_words set
            if word not in known_words:
                # Try lemmatizing to catch inflected forms (e.g., "emailed" -> "email")
                lemma_v = lemmatizer.lemmatize(word, pos='v')  # Try verb form
                lemma_n = lemmatizer.lemmatize(word, pos='n')  # Try noun form
                lemma_a = lemmatizer.lemmatize(word, pos='a')  # Try adjective form
                lemma_r = lemmatizer.lemmatize(word, pos='r')  # Try adverb form

                if (lemma_v not in known_words and
                    lemma_n not in known_words and
                    lemma_a not in known_words and
                    lemma_r not in known_words and
                    word.isalpha()): # Only include purely alphabetic unknown words
                    unknown.append(word)

    return sorted(list(set(unknown)))


df["clean_comments"] = preprocess_vectorized(df["merged_comments"])

# Add unknown words column using the revised function
print("Detecting unknown words/abbreviations using revised method...")
df["unknown_words"] = df["merged_comments"].apply(
    lambda x: ", ".join(find_unknown_words_revised(x)) if find_unknown_words_revised(x) else ""
)

print(f"Processing {len(df)} rows with valid comments...")


# ====================
# Step 4: Save results with Excel formatting
# ====================

print(f"Final dataset columns: {list(df.columns)}")
print(f"Final dataset shape: {df.shape}")

# Dynamically name output file
output_file = input_file.stem + "_preprocessed.xlsx"
df.to_excel(output_file, index=False)

print(f"\n✅ Enhanced classification complete. File saved as {output_file}")
print(f"📊 Processed {len(df)} rows total")

"""### B. Use saved model to classify"""

# New code cell for applying model to milton.xlsx
import pandas as pd
import os
import pickle
from pathlib import Path
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

class MultiLabelDistilBERTClassifier:
    """Multi-label classifier using DistilBERT for crosscheck categories"""

    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.mlb = None
        self.categories = None
        self.optimal_thresholds = None

    def advanced_preprocessing(self, text):
        """Enhanced preprocessing for utility text"""
        if pd.isna(text) or text == '':
            return ''

        text = str(text).lower()
        text = re.sub(r'[^\w\s\-\.]', ' ', text)

        utility_replacements = {
            r'\bxfmr\b': 'transformer',
            r'\btx\b(?!\s*\d)': 'transformer',
            r'\boh\b': 'overhead',
            r'\bug\b': 'underground',
            r'\bpri\b': 'primary',
            r'\bsec\b': 'secondary',
            r'\bkv\b': 'kilovolt',
            r'\bpole\b': 'utility pole',
            r'\btree\b': 'vegetation',
            r'\blimb\b': 'vegetation',
            r'\bflood\b': 'flooding water',
            r'\bwater\b': 'flooding',
        }

        for pattern, replacement in utility_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        text = ' '.join(text.split())
        return text

    def predict(self, text, use_optimal_thresholds=True):
        """Make predictions on some text"""
        if self.model is None:
            return "Model not trained yet"

        cleaned_text = self.advanced_preprocessing(text)

        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        with torch.no_grad():
            inputs = self.tokenizer(
                cleaned_text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        # Apply per-class thresholds
        predicted_labels = []
        thresholds_used = {}

        for i, cat in enumerate(self.categories):
            if use_optimal_thresholds and getattr(self, 'optimal_thresholds', None):
                threshold = self.optimal_thresholds.get(cat, 0.5)
            else:
                threshold = 0.5

            thresholds_used[cat] = threshold
            if probs[i] > threshold:
                predicted_labels.append(cat)

        # If no predictions, pick the highest probability or 'other' if sufficiently likely
        if not predicted_labels:
            if 'other' in self.categories and probs[self.categories.index('other')] > 0.1:
                predicted_labels = ['other']
            else:
                predicted_labels = [self.categories[int(np.argmax(probs))]]

                # Enforce "other" exclusivity: drop if combined with other labels
        if 'other' in predicted_labels and len(predicted_labels) > 1:
            predicted_labels = [lbl for lbl in predicted_labels if lbl != 'other']
            if not predicted_labels:  # Rare: if only 'other' was left after drop, fallback to max non-other
                non_other_probs = {cat: probs[i] for i, cat in enumerate(self.categories) if cat != 'other'}
                if non_other_probs:
                    predicted_labels = [max(non_other_probs, key=non_other_probs.get)]
                else:
                    predicted_labels = ['other']  # Truly no others

        result = {
            'predicted_categories': predicted_labels,
            'confidences': {self.categories[i]: float(probs[i]) for i in range(len(probs))},
            'thresholds_used': thresholds_used
        }

        return result

    def load_model(self, save_dir="./saved_model_crosscheck"):
        """Load pre-trained model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(save_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(save_dir)

        with open(os.path.join(save_dir, 'mlb.pkl'), 'rb') as f:
            self.mlb = pickle.load(f)
        with open(os.path.join(save_dir, 'categories.pkl'), 'rb') as f:
            self.categories = pickle.load(f)

        threshold_path = os.path.join(save_dir, 'thresholds.pkl')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'rb') as f:
                self.optimal_thresholds = pickle.load(f)

        print(f"Model loaded from {save_dir}")

# Ensure classifier is available (use in-memory if present, otherwise load saved model)
try:
    clf_ok = isinstance(classifier, MultiLabelDistilBERTClassifier) and classifier.model is not None
except NameError:
    clf_ok = False

if not clf_ok:
    print('Classifier not found in memory — attempting to load from ./saved_model_crosscheck')
    classifier = MultiLabelDistilBERTClassifier()
    if os.path.exists('./saved_model_crosscheck'):
        classifier.load_model('./saved_model_crosscheck')
    else:
        raise FileNotFoundError("Saved model directory './saved_model_crosscheck' not found. Train and save model first.")

# Locate test file
candidate = Path(output_file)

print(f"📄 You are using this test file: {candidate}")

if candidate.exists():
    data_path = candidate
    print(f"✅ Found file: {data_path}")
else:
    data_path = None
    print("❌ File not found.")

#checks
if data_path is None:
    raise FileNotFoundError("Could not find test file (tried classified_enhanced.xlsx, milton.xlsx, milton.xls)")

print(f'Loading data from: {data_path}')
df_milton = pd.read_excel(data_path)

if 'clean_comments' not in df_milton.columns:
    raise KeyError("Column 'clean_comments' not found in milton file")

print(f'Processing {len(df_milton)} rows...')

# Preprocess text using classifier's preprocessing
df_milton['processed_text'] = df_milton['clean_comments'].apply(classifier.advanced_preprocessing)

# Run predictions row-by-row and store results
preds = []
confidences_list = []

for idx, txt in enumerate(df_milton['processed_text']):
    if (idx + 1) % 5000 == 0:
        print(f'  Processed {idx + 1}/{len(df_milton)} rows...')

    res = classifier.predict(txt, use_optimal_thresholds=True)

    # Store just the category names (comma-separated)
    preds.append(', '.join(res['predicted_categories']))

    # Store confidences as a clean dictionary (optional - for your reference)
    confidences_list.append(res['confidences'])

# Add predictions to dataframe
df_milton['predicted_crosscheck'] = preds

# Optional: Add confidence scores as separate column if you want them for analysis
# Uncomment the line below if you want confidence scores in a separate column
# df_milton['prediction_confidences'] = confidences_list

# Save results to new Excel file
out_file = data_path.with_name(data_path.stem + '_with_predictions.xlsx')
df_milton.to_excel(out_file, index=False)
print(f'\n✓ Predictions saved to: {out_file}')
print(f'\nSample predictions:')
print(df_milton[['clean_comments', 'predicted_crosscheck']].head(10))
