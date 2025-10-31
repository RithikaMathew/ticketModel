## Ticket Classification — Testing Guide

This README is focused on running the provided, pre-trained ticket classification pipeline to generate predictions from an Excel file. It does NOT include training instructions.

## Quick Start (for non-ML users)

1. Open a terminal and navigate to the project folder:

```bash
cd /Users/xrconnectivity/Desktop/ticketFPL-1
```

2. Install Python dependencies (if not already installed):

```bash
pip3 install -r requirements.txt
```

3. Run the prediction script (uses only the saved model):

```bash
python3 ticket.py
```

Note:
- The script loads the trained model from `./saved_model_crosscheck` — no training required and no model downloads during prediction.

## Required input file

- Default filename: `Milton.xlsx` (place it in the project root) — you may use any Excel file, see "Advanced Configuration" below.
- The script expects these columns (case-sensitive) for core operation:
      - `Dispatch Center Comments` (required)
      - `Service Center Comments` (required)

- Optional columns used only for visualizations (when running `graphs.py`):
      - `CI` (customer impact)
      - `County Name` or `Franchise Name`

Supported formats: `.xlsx` (primary), `.xls` (legacy Excel)

## Advanced Configuration — Custom input files

If you want to use a different input filename, update the `input_file` in `ticket.py` or create a small wrapper. Example edit in `ticket.py`:

```python
from pathlib import Path
input_file = Path("your_input_file.xlsx")
```

I can add a wrapper script like `run_predictions.py --input your_input.xlsx` if you'd prefer not to edit the file.

## What the script produces

- `<inputname>_preprocessed.xlsx` — preprocessed data with added columns:
      - `merged_comments` — merged comment text
      - `clean_comments` — cleaned and normalized text ready for the model
      - `unknown_words` — comma-separated tokens the script considered unknown or domain-specific

- `<inputname>_with_predictions.xlsx` — final output with predictions:
      - All original columns
      - `predicted_crosscheck` — comma-separated predicted categories (e.g., "vegetation, damaged pole")

Files are written to the current working directory; the output filenames are derived from your input filename (stem + suffix).

## Visualizations

- Visualizations are produced by `graphs.py`. If you want charts, run `graphs.py` after generating the predictions file. `graphs.py` uses the optional `CI` and location columns when available.

## Troubleshooting (common issues)

- "No module named 'pandas'" — install dependencies:

```bash
pip3 install -r requirements.txt
```

- "File not found" — ensure your input file is in the project root and spelled exactly.

- "Column 'Dispatch Center Comments' not found" — verify the Excel file has the required column names.

- For very large files, processing is slower; using a machine with a GPU and more RAM will speed up prediction.

## Quick checks (after running)

1. Confirm the `<inputname>_with_predictions.xlsx` file exists in the project root.
2. Open the file and verify `predicted_crosscheck` values for a few rows.

## File layout (relevant files)

```
ticketFPL-1/
├── requirements.txt
├── ticket.py                     # Run this to preprocess + predict using saved model
├── graphs.py                     # Generate visualizations from predictions
├── saved_model_crosscheck/       # Pre-trained model (used by ticket.py)
└── (outputs generated after running)
```

## 📄 License

This project is part of utility infrastructure analysis. Please ensure appropriate data privacy and security measures when processing real utility data.

