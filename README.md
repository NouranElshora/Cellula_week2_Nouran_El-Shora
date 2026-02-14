# Cellula Week 2 - Toxic Content Classification Project

## Overview

This repository contains the code and research for a **Toxic Content Classification** project.  
The project is designed to classify user-submitted text and image content into categories such as *Safe*, *Violent Crimes*, *Unsafe*, *Child Sexual Exploitation*, and others.

The project uses a **modular structure**, separating the image captioning, classification, and application logic, making it easy to maintain and extend.

---

## Project Components

### 1. Image Captioning (`imagecaption.py`)
- Uses models such as **BLIP-1 or BLIP-2** from Hugging Face.  
- Converts user-submitted images into text captions.  
- Encapsulated in a separate module for reusability.

### 2. Text Classification (`classifier.py`)
- Accepts text input from users or captions generated from images.  
- Supports models like:
  - **DistilBERT with LoRA** (lightweight transformer)
  - **LSTM** (traditional sequence model)
  - **LLaMA Guard** (LLM-based toxicity detection)  
- Returns the predicted toxicity category.

### 3. Application (`app.py`)
- Built using **Streamlit** for a simple interactive interface.  
- Accepts both **text input** and **image uploads** from users.  
- Passes the data through the image captioning and classification pipeline.  
- Updates a **CSV file** (`cellula toxic data.csv`) with the input and predicted category.  
- Allows viewing stored entries anytime.

### 4. Model Training (`train_model.py`)
- Contains the logic to train a **DistilBERT-based classifier**.  
- Implements:
  - **Stratified train-test split** for balanced category representation  
  - **Early stopping** to prevent overfitting  
  - Tokenization using `DistilBertTokenizerFast`  
  - Saving the trained model, tokenizer, and label mapping  
- **Note:** The repo currently contains the training code only, not trained model files.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/NouranElshora/Cellula_week2_Nouran_El-Shora.git
cd Cellula_week2_Nouran_El-Shora/Toxic content classification project

2. Create a virtual environment:

python -m venv venv
source venv/Scripts/activate   # Windows

3. Install dependencies:

pip install -r requirements.txt
