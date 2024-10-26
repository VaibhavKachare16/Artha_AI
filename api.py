from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define file paths and API key
EMBEDDINGS_FILE = 'embeddings.npy'
CONTEXTS_FILE = 'contexts.npy'
API_KEY = "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"
api_key_header = APIKeyHeader(name="X-API-Key")

# Initialize FastAPI
app = FastAPI()

# Request models
class QuestionRequest(BaseModel):
    question: str
    csv_file_path: str = "./uploads/mutual_funds_data.csv"

class AnswerResponse(BaseModel):
    answer: str

# Helper functions
def embed_text(text: str) -> torch.Tensor:
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1)

def cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

def load_csv_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_embeddings(embeddings: torch.Tensor, contexts: list):
    np.save(EMBEDDINGS_FILE, embeddings.numpy())
    np.save(CONTEXTS_FILE, contexts)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CONTEXTS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        contexts = np.load(CONTEXTS_FILE, allow_pickle=True).tolist()
        return torch.tensor(embeddings), contexts
    return None, None

def get_relevant_contexts(question: str, df: pd.DataFrame, top_k: int = 5) -> list:
    question_embedding = embed_text(question)
    all_context_embeddings, all_contexts = load_embeddings()

    if all_context_embeddings is None or all_contexts is None:
        all_contexts = df.apply(lambda row: f"{row['scheme_name']}, SIP: {row['min_sip']}, Lumpsum: {row['min_lumpsum']}, "
                                            f"Expense Ratio: {row['expense_ratio']}%, Fund Size: {row['fund_size_cr']} Cr, "
                                            f"Fund Age: {row['fund_age_yr']} years, Manager: {row['fund_manager']}, "
                                            f"Category: {row['category']} - {row['sub_category']}, "
                                            f"Risk Level: {row['risk_level']}, Returns 1Y/3Y/5Y: {row['returns_1yr']}%/"
                                            f"{row['returns_3yr']}%/{row['returns_5yr']}%", axis=1).tolist()

        all_context_embeddings = [embed_text(context) for context in tqdm(all_contexts, desc="Embedding contexts")]
        all_context_embeddings = torch.vstack(all_context_embeddings)
        save_embeddings(all_context_embeddings, all_contexts)
    else:
        print("Loaded existing embeddings.")

    similarities = cosine_similarity(question_embedding, all_context_embeddings)
    top_results = similarities.topk(k=top_k)
    
    indices = top_results.indices.squeeze().tolist()
    relevant_contexts = [all_contexts[i] for i in indices]
    return relevant_contexts

def generate_answer(question: str, context: str) -> str:
    return f"Answer to '{question}' based on context: {context}"

def answer_question(question: str, csv_file_path: str) -> str:
    df = load_csv_file(csv_file_path)
    relevant_contexts = get_relevant_contexts(question, df, top_k=5)
    if relevant_contexts:
        combined_context = " ".join(relevant_contexts)
        return generate_answer(question, combined_context)
    return "Not enough relevant data found to provide an answer."

# API endpoint
@app.post("/ask-question/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")
    
    try:
        answer = answer_question(request.question, request.csv_file_path)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
