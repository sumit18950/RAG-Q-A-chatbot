# 📊 Loan Dataset RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that answers questions about a loan approval dataset by retrieving relevant rows and using a generative AI model (like OpenAI GPT-3.5) to produce intelligent, context-aware responses.

## 🚀 Features

- 🔍 Semantic search using FAISS and SentenceTransformers
- 🤖 LLM-based answer generation using OpenAI (GPT-3.5-turbo)
- 📂 Trained on a real-world loan prediction dataset
- 🌐 Interactive web UI using Streamlit

## 💠 Tech Stack

| Component     | Tool                                  |
|--------------|---------------------------------------|
| Embedding    | all-MiniLM-L6-v2 (SentenceTransformers) |
| Vector Store | FAISS                                 |
| LLM          | OpenAI GPT-3.5-turbo                  |
| Frontend     | Streamlit                             |
| Dataset      | [Kaggle: Loan Approval Prediction](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction) |

## 📅 File Structure

```
loan-rag-chatbot/
├── Training Dataset.csv        # Dataset used for indexing
├── app.py                      # Main Streamlit app with RAG logic
├── README.md                   # Project documentation
```

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/loan-rag-chatbot.git
cd loan-rag-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install pandas faiss-cpu sentence-transformers openai streamlit
```

### 3. Add OpenAI API Key
Set your OpenAI API key as an environment variable:

**Linux/macOS:**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=your-key-here
```

Or replace "YOUR_OPENAI_API_KEY" in `app.py`.

### 4. Run the App
```bash
streamlit run app.py
```
<img width="1172" height="381" alt="image" src="https://github.com/user-attachments/assets/5803b0bd-e3aa-4861-a5e7-ff479deaf4ef" />


## 🧪 Sample Questions

- What is the typical income of approved applicants?
- How does credit history impact loan approval?
- What are the most common rejection reasons?

## ✅ TODO

- [ ] Add support for Claude/Gemini/Mistral (Hugging Face or Anthropic)
- [ ] Dockerize the application
- [ ] Add login/auth system (optional)
- [ ] Save chat history

## 📜 License

This project is open-source and free to use for educational or non-commercial purposes.
