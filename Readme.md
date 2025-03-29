# Instalysis – AI-Powered Mental Health Counselor

## Overview
**Instalysis** is an innovative application leveraging advanced AI to analyze mental health through Instagram captions. It integrates multilingual transformer models (XLM-RoBERTa and mBERT) fine-tuned specifically for mental health prediction and provides empathetic conversational support via Google's Gemini AI.

---

## Project Directory Structure
```
instalysis/
├── final_fusion/
│   ├── config.json
│   ├── model.safetensors
│   ├── sentencepiece.bpe.model
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── dataset/
│   └── cleaned D.csv
├── app.py
├── instaloader_fetcher.py
├── preprocess.py
├── fusion_model.py
└── requirements.txt
```

---

## Installation and Setup

### 1. Fine-Tuning Model (Google Colab)

- Upload provided Jupyter notebook and dataset (`cleaned D.csv`) to Google Drive.
- Mount Google Drive in Colab:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- Update dataset path and model save path in the notebook.
- Run all notebook cells sequentially to fine-tune the model.

### 2. Preparing the Model
- Create a folder `final_fusion`.
- Copy the following files from Colab output to `final_fusion`:
  - `config.json`
  - `model.safetensors`
  - `sentencepiece.bpe.model`
  - `special_tokens_map.json`
  - `tokenizer_config.json`
  - `tokenizer.json`
  - `vocab.txt`

### 3. Local Environment Setup
- Create virtual environment:
  ```bash
  python -m venv venv
  ```
- Activate environment:
  - **Windows**:
    ```bash
    .\venv\Scripts\activate
    ```
  - **Linux/MacOS**:
    ```bash
    source venv/bin/activate
    ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Install SpaCy language model:
  ```bash
  python -m spacy download en_core_web_sm
  ```

### 4. API Keys
- Obtain Gemini API key from Google Generative AI.
- Generate your APIFY API key from [apify.com](https://apify.com).
- Replace placeholders in:
  - `app.py` (Gemini API)
  - `instaloader_fetcher.py` (Apify API)

---

## Running the Application

Launch Streamlit:
```bash
streamlit run app.py
```
Access via browser at:
```
http://localhost:8501
```

---

## Application Usage
- Analyze mental state from Instagram captions.
- Interact with Gemini-powered AI counselor.
- Receive personalized mental health insights and recommendations.

---

## Features
- Multilingual mental health prediction using fusion transformer model.
- Advanced text preprocessing for accurate analysis.
- Intuitive and responsive UI with Streamlit.
- Empathetic conversational AI using Gemini API.

---

## Disclaimer
Instalysis is intended solely for supportive interaction. It does not replace professional psychological advice. Always consult licensed mental health professionals for personalized support.

---

## Contributions
Contributions to enhance the project's capabilities are encouraged. Please submit pull requests or raise issues to discuss improvements.


##Detailed Instrcutions

1) To finetune the model use the colab and upload the jupyter notebook in this repo. Uplaod the dataset in this repo to your drive and mount it and then copy the path to the dataset path in the code. Other than this you also need to specify where to save the model after finetuning it.

2) Create a folder named final_fusion

3) After finetuning, load the saved model content and paste all the 6 files named config.json, model.safetensors, sentencepiece.bpe.model, special_tokens_map.json, tokenizer_config.json, tokenizer.json, vocab.txt into a folder named final_fusion

4) To run the app.py you need to first create a venv. Use command python -m venv venv
5) Then activate the venv using command ./venv/Scripts/activate
6) Then install all the required modules by running pip install -r requirement.txt
7) Then run the command python -m spacy download en_core_web_sm
8) After these run streamlit run app.py
9) Note: Do not forget to place your gemini api key and apify key in app.py and instaloader_fetcher.py


### You can generate you APIFY Account key at apify.com 



**© 2025 Instalysis**