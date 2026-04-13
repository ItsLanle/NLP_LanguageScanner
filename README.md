# Inappropriate Comments Scanner

## Overview
This project uses Natural Language Processing (NLP) to detect harmful language in text, including:
- Bullying
- Sarcasm
- Implicit bias
- General harmful content

## Features
- Context-aware classification using BERT
- Multi-class labeling
- Interactive Streamlit interface

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model (optional)
python train.py

### 3. Run app
streamlit run app.py

## Goal
To create a smarter moderation tool that understands context rather than relying on simple keyword filtering.
