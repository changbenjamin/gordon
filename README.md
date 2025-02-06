# GordonRamsay Hypothesis Evaluation Model

## Overview

This repository contains code for a machine learning project aimed at evaluating the "goodness" of scientific hypotheses. The project utilizes two-stage neural networks to predict the quality of scientific hypotheses based on paper abstracts from bioRxiv and medRxiv.

## Data

The project utilizes data from:

*   **bioRxiv and medRxiv:** Paper titles and abstracts.
*   **PubMed Relative Citation Ratio (RCR):** Used as proxy labels for hypothesis quality.

The data processing pipeline includes:

*   **PubMed Querying:**  Fetching relevant metadata.
*   **Abstract Processing:** Cleaning and preparing abstract text for embedding generation.
*   **Hypothesis Distillation (using Llama):**  Extracting the core hypothesis as a question and summarizing the background from abstracts.
*   **Embedding Generation:** Creating text embeddings using both BioBERT and Llama models.

## Usage

### Generating Embeddings:

*   **BioBERT Embeddings:** Run `biobert_embeddings.py` to generate BioBERT embeddings for your dataset.
    ```bash
    python biobert_embeddings.py
    ```
    Output embeddings will be saved in pickle files (e.g., `biobert_embeddings_hypothesis.pkl`, `biobert_embeddings_background.pkl`).

*   **Llama Embeddings:** Run `llama_embeddings.py` to generate Llama embeddings.
    ```bash
    python llama_embeddings.py
    ```
    Output embeddings will be saved in pickle files (e.g., `llama_embeddings_hypothesis.pkl`, `llama_embeddings_background.pkl`).

### Training Models:

*   ** Run `gordon_training.py` to train the two-stage scoring model, including hyperparameter tuning using Optuna.
    ```bash
    python gordon_training.py
    ```
    Trained model weights for Stage 1 and Stage 2 will be saved as `.pth` files (e.g., `gordonramsay_stage1_BioBERT.pth`, `gordonramsay_stage2_BioBERT.pth`).

### Using the RCR Prediction Tool:

*   **Run `gordon.py`:** This script loads the trained two-stage scoring model (Llama or BioBERT, depending on configuration) and prompts you to enter hypothesis and background text. It then outputs the predicted RCR score.
    ```bash
    python gordon.py
    ```
ctly learns to rank pairs of hypotheses based on RCR.

## Embeddings

The project supports two types of text embeddings:

*   **BioBERT Embeddings:** Biomedical domain-specific embeddings, efficient for biomedical text processing.
*   **Llama Embeddings:** General-purpose, high-quality embeddings from a Large Language Model, capturing broader semantic nuances.
