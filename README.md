# movie-chatbot
# Movie Chatbot – LSTM with Attention + RAG

This project implements a **movie question-answering chatbot** using a custom-trained **LSTM language model with self-attention**, combined with a **Retrieval-Augmented Generation (RAG)** pipeline over structured movie data.

The system is trained and executed in **Databricks**, with models logged and versioned via **MLflow**.

---

## Project Overview

The chatbot answers questions such as:

"Tell me about Kansas Saloon Smashers"  
"Who directed Picnic?"  
"What is the plot of The Proud Ones?"

It works in two stages:

1. **Retrieval (RAG)** – fetch relevant movie records from a Spark table.  
2. **Generation (LM)** – generate a natural language answer using a trained LSTM-based language model.

---

## Architecture

User Question  
→ Retrieval (Spark SQL)  
→ Structured Context  
→ Prompt Construction  
→ LSTM + Attention Model  
→ Generated Answer

---

## Model Design – LSTM with Self-Attention

The core language model is a **next-token prediction model** trained on movie text.

### Input Pipeline
- TextVectorization tokenizer  
- Sliding windows of length SEQ_LEN  
- Predict next token at each timestep

---

## Neural Network Architecture

Embedding → token vectors
LSTM → temporal memory
Attention → re-focus on important tokens
Concatenate → merge memory + attention
Dense → next-token logits

## Training Data

Wiki Movie Plots
Table: default.wiki_movie_plots_deduped
Cornell Movie Dialogs
Tables:
workspace.default.movie_lines
workspace.default.movie_conversations

---

## Inference – RAG Pipeline

Question → keyword retrieval
Build structured context
Inject into prompt
Generate answer token-by-token
