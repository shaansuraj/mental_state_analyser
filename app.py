import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from streamlit_chat import message  

import pandas as pd
import matplotlib.pyplot as plt
import asyncio
import logging
import os
import random
import numpy as np
from collections import Counter

import torch
from transformers import AutoTokenizer
import google.generativeai as genai

# For loading safetensors
from safetensors.torch import load_file

# -- The following imports reference your custom modules:
# instaloader_fetcher: for fetch_captions()
# preprocess: for preprocess_captions()
# fusion_model: for FusionClassifier, FusionConfig
from instaloader_fetcher import fetch_captions
from preprocess import preprocess_captions
from fusion_model import FusionClassifier, FusionConfig

###############################################################################
# Environment and Logging Setup
###############################################################################
# *** Your Gemini API Key ***
GENAI_API_KEY = "AIzaSyAQDazslbfHxZ3OHKYlbU00iJrqb_YotYw"
genai.configure(api_key=GENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Load Fusion Model and Tokenizers
###############################################################################
model_path = "./final_fusion"  # Adjust if needed

config = FusionConfig(
    num_labels=5,
    xlmr_checkpoint="xlm-roberta-large",
    mbert_checkpoint="bert-base-multilingual-cased",
    xlmr_hidden_size=1024,
    mbert_hidden_size=768,
    label2id={
        "Normal": 0,
        "Happy": 1,
        "Stressed": 2,
        "Moderately Depressed": 3,
        "Severely Depressed": 4
    },
    id2label={
        0: "Normal",
        1: "Happy",
        2: "Stressed",
        3: "Moderately Depressed",
        4: "Severely Depressed"
    }
)

model = FusionClassifier(config)
state_dict = load_file(os.path.join(model_path, "model.safetensors"))
model.load_state_dict(state_dict)
model.eval()

tokenizer_xlmr = AutoTokenizer.from_pretrained("xlm-roberta-large")
tokenizer_mbert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
if tokenizer_mbert.pad_token is None:
    tokenizer_mbert.pad_token = tokenizer_mbert.eos_token

label_map = {
    0: "Normal",
    1: "Happy",
    2: "Stressed",
    3: "Moderately Depressed",
    4: "Severely Depressed"
}

###############################################################################
# Prediction Function
###############################################################################
def predict_state_of_mind(captions):
    """Predict the mental state from a list of captions using the fusion model."""
    hard_preds = []
    logits_list = []
    model.eval()
    
    for caption in captions:
        cleaned_caption = preprocess_captions([caption])[0]
        
        xlmr_inputs = tokenizer_xlmr(
            cleaned_caption,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )
        mbert_inputs = tokenizer_mbert(
            cleaned_caption,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )
        with torch.no_grad():
            outputs = model(
                xlmr_input_ids=xlmr_inputs["input_ids"],
                xlmr_attention_mask=xlmr_inputs["attention_mask"],
                mbert_input_ids=mbert_inputs["input_ids"],
                mbert_attention_mask=mbert_inputs["attention_mask"]
            )
        logits = outputs["logits"]
        hard_pred = torch.argmax(logits, dim=-1).item()
        hard_preds.append(hard_pred)
        logits_list.append(logits)
    
    vote_counts = Counter(hard_preds)
    majority_label, count = vote_counts.most_common(1)[0]
    
    # If a clear majority among the captions
    if count >= len(hard_preds) / 2:
        final_label = majority_label
    else:
        avg_logits = torch.mean(torch.cat(logits_list, dim=0), dim=0)
        final_label = torch.argmax(avg_logits).item()
    
    return label_map[final_label]

###############################################################################
# Gemini Chatbot Logic
###############################################################################
def gemini_chatbot_logic(user_input):
    """
    Uses Gemini's generative model to respond empathetically.
    1. Checks if the user's message is about mental health or a greeting.
    2. Replies in the same language if valid; otherwise instructs user to stay
       on mental health topics or greet with 'hi'/'hello'.
    """
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")

        # Quick check if the user's text is mental health or greeting
        check_prompt = (
            "Determine if the following message is about mental health issues, or emotional or a simple greeting "
            "(like 'hi', 'hello', etc.). Answer only with 'Yes' or 'No':\n"
            f"Message: '{user_input}'"
        )
        check_response = model_gemini.generate_content(check_prompt)
        if check_response.text.strip().lower().startswith("no"):
            return "Please ask mental health related questions or greet me with hi/hello."
        
        # If valid, generate an empathetic reply in the same language
        reply_prompt = (
            "Act as a compassionate mental health counselor named AI Counsellor. "
            "Respond empathetically in the same language as the user's message. "
            "User's message:\n"
            f"'{user_input}'"
        )
        response = model_gemini.generate_content(reply_prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error: Unable to generate response. Details: {e}"

def get_recommendations(state):
    """Fetch short recommendations from Gemini for the predicted mental state."""
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Provide 3 concise recommendations for someone feeling '{state}' to improve mental well-being. "
            "Be direct and supportive."
        )
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

###############################################################################
# Streamlit App
###############################################################################
st.set_page_config(page_title="Instalysis - Mental Health Counsellor", layout="wide")

# We maintain:
# - st.session_state.messages for chatbot logs
# - st.session_state.ig_analysis_done to see if user has done analysis
# - st.session_state.user_mood to store predicted mood after analysis

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ig_analysis_done" not in st.session_state:
    st.session_state.ig_analysis_done = False

if "user_mood" not in st.session_state:
    st.session_state.user_mood = None

# Basic CSS to style the chat area a bit
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f8efd4, #d9e4f5);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background: linear-gradient(135deg, #34568b, #182848);
        padding: 30px 20px;
        border-radius: 15px;
        color: #fff;
        font-size: 2.4em;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }
    .section-box {
        background-color: #ffffffBB;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .chat-container {
        width: 80%;
        margin: 0 auto;
        background-color: #fdfdfd;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="header">Instalysis - Mental Health Counsellor</div>', 
            unsafe_allow_html=True)

###############################################################################
# Instagram Credential Inputs
###############################################################################
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("Instagram Analysis")

if "auth_type" not in st.session_state:
    st.session_state.auth_type = None

col1, col2 = st.columns(2)
with col1:
    if st.button("Public", key="public_btn"):
        st.session_state.auth_type = "Public"
with col2:
    if st.button("Private", key="private_btn"):
        st.session_state.auth_type = "Private"

submit_button = False
username = ""
password = ""

if st.session_state.auth_type == "Public":
    with st.form(key="public_form"):
        username = st.text_input("Enter your Instagram username:")
        submit_button = st.form_submit_button("Analyze")
elif st.session_state.auth_type == "Private":
    with st.form(key="private_form"):
        username = st.text_input("Enter your Instagram username:")
        password = st.text_input("Enter your Instagram password:", type="password")
        submit_button = st.form_submit_button("Analyze")
else:
    st.info("Select 'Public' or 'Private' to provide Instagram credentials.")

st.markdown("</div>", unsafe_allow_html=True)

###############################################################################
# If user submitted credentials, do the analysis
###############################################################################
if submit_button and username.strip():
    async def run_fetch_captions():
        with st.spinner("Fetching captions from Instagram..."):
            return await fetch_captions(username)
    
    captions = asyncio.run(run_fetch_captions())
    
    if captions:
        st.success(f"Fetched {len(captions)} captions.")
        cleaned_captions = preprocess_captions(captions)
        st.write(f"**Cleaned Captions (first 10)**: {cleaned_captions[:10]}")
        
        # Filter empty
        cleaned_captions = [c for c in cleaned_captions if c.strip()]
        if not cleaned_captions:
            st.warning("No valid captions to analyze.")
        else:
            # Predict mental state
            user_mood = predict_state_of_mind(cleaned_captions)
            st.session_state.user_mood = user_mood
            st.session_state.ig_analysis_done = True

            st.write(f"**Predicted Mental Health Condition**: {user_mood}")

            # Show recommendations from Gemini
            recs = get_recommendations(user_mood)
            st.markdown("**Recommendations**:")
            st.markdown(recs)

            # Clear old messages
            st.session_state.messages.clear()

            # Initial greeting from AI Counsellor after analysis
            initial_msg = (
                f"Hello, I'm your AI Mental Health Counsellor. "
                f"It seems you may be feeling **{user_mood}**. "
                "I'm here to listen and help. How are you feeling about this?"
            )
            st.session_state.messages.append({
                "role": "bot",
                "content": initial_msg
            })
    else:
        st.warning("No captions returned. Check if the username is correct or private.")

###############################################################################
# Chat Section
###############################################################################
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("AI Counsellor Chat")

# If user hasn't done analysis, we can show a basic greeting
if not st.session_state.ig_analysis_done and len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "bot",
        "content": (
            "Hello, I'm your AI Mental Health Counsellor. "
            "Please analyze Instagram captions (above) so we can discuss "
            "your mental state. Or, feel free to say hi anyway!"
        )
    })

# Show chat in a container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display the messages
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "bot":
        message(msg["content"], is_user=False, key=f"bot_{i}")
    else:
        message(msg["content"], is_user=True, key=f"user_{i}")

st.markdown("</div>", unsafe_allow_html=True)

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...")
    send_btn = st.form_submit_button("Send")

if send_btn and user_input.strip():
    # Add the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Get the bot's response
    bot_response = gemini_chatbot_logic(user_input)
    # Add the bot's response
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Button to reset/clear chat
if st.button("Reset Chat"):
    st.session_state.messages.clear()
    st.session_state.ig_analysis_done = False
    st.session_state.user_mood = None

st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.markdown("""
<div style="font-size: 13px; color: #555;">
**Disclaimer**: This AI Counsellor is for supportive conversation only and is not a 
substitute for professional mental health care. If you need personalized advice, 
please consult a licensed mental health professional.
</div>
""", unsafe_allow_html=True)
