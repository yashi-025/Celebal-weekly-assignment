import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Optional: Use OpenAI if user selects it
import openai
import google.generativeai as genai

st.set_page_config(page_title="Loan Q&A Chatbot", layout="wide")

st.title("Loan Q&A Chatbot")

# Load datasets (you already have them)
@st.cache_data
def load_data():
    train_df = pd.read_csv("dataset/Training Dataset.csv")
    test_df = pd.read_csv("dataset/Test Dataset.csv")
    return train_df, test_df

train_df, test_df = load_data()
st.success("Training and Testing data loaded successfully!")

# Select model
model_choice = st.selectbox("Choose your LLM:", ["OpenAI (GPT-3.5/4)", "Google Gemini", "Claude (Anthropic)"], key="model_choice")
api_key = st.text_input(f"Enter your API key for {model_choice}", type="password")

# Upload your own file for prediction
uploaded_file = st.file_uploader("Upload file for prediction (CSV, TXT, PDF):", type=['csv', 'txt', 'pdf'])

# Create knowledge base (from training data)
@st.cache_resource
def create_vector_index(texts):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, embed_model

docs = train_df.apply(lambda row: f"Loan ID: {row['Loan_ID']}, Details: {row.drop('Loan_ID').to_dict()}", axis=1).tolist()
index, embeddings, embed_model = create_vector_index(docs)

# Function to retrieve relevant context
def retrieve_context(query, top_k=3):
    q_embed = embed_model.encode([query])
    _, I = index.search(q_embed, top_k)
    return [docs[i] for i in I[0]]

# Generate answer using selected LLM
def generate_response_llm(context, question, model_choice, api_key):
    full_prompt = f"""You are a helpful AI assistant. Use the following context to answer the question:
    
Context:
{context}

Question:
{question}

Answer:"""

    if model_choice.startswith("OpenAI"):
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.5,
        )
        return response['choices'][0]['message']['content']

    elif model_choice.startswith("Google Gemini"):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(full_prompt)
        return response.text

    elif model_choice.startswith("Claude"):
        return "Claude support is coming soon. Currently not implemented."

    else:
        return "Invalid model selection."

# Form for question input
st.subheader("Ask Questions from Training Knowledge Base")

with st.form("query_form"):
    question = st.text_input("Enter your question")
    submitted = st.form_submit_button("Get Answer")

    if submitted and question:
        with st.spinner("Retrieving answer..."):
            context_list = retrieve_context(question)
            combined_context = "\n\n".join(context_list)
            response = generate_response_llm(combined_context, question, model_choice, api_key)
            st.markdown("### Retrieved Answer:")
            st.write(response)

# Optional file preview
if uploaded_file is not None:
    st.subheader("Uploaded File Preview")
    if uploaded_file.name.endswith(".csv"):
        df_uploaded = pd.read_csv(uploaded_file)
        st.write(df_uploaded.head())
    elif uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode()
        st.text(content[:1000])
    elif uploaded_file.name.endswith(".pdf"):
        st.info("PDF parsing is not implemented yet.")

# ========== VISUALIZATION SECTION ==========
st.subheader("Dataset Insights & Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Gender Distribution")
    st.bar_chart(train_df['Gender'].value_counts())

with col2:
    st.markdown("#### Education Levels")
    st.bar_chart(train_df['Education'].value_counts())

st.markdown("#### Loan Amount Distribution")
st.line_chart(train_df['LoanAmount'].fillna(0))

# Dynamic plot for any categorical column
st.markdown("#### Explore Categorical Features")
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()
selected_col = st.selectbox("Choose a column to visualize", categorical_cols)
st.bar_chart(train_df[selected_col].value_counts())
