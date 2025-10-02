# app/main.py
import os, json, re, time, uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import joblib
from threading import Lock

DATA_DIR = "data"
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
RISK_MODEL_FILE = "models/risk_model.joblib"
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)
app = FastAPI(title="Doctor AI - Risk Assistant (local prototype)")

# ---- Pydantic schemas ----
class Vitals(BaseModel):
    bp_systolic: Optional[float]
    bp_diastolic: Optional[float] = None
    hr: Optional[float] = None

class Labs(BaseModel):
    creatinine: Optional[float] = None
    hemoglobin: Optional[float] = None

class PredictPayload(BaseModel):
    patient_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    vitals: Optional[Vitals] = None
    labs: Optional[Labs] = None
    notes: Optional[str] = ""

class LiteratureItem(BaseModel):
    id: str
    title: str
    source: Optional[str] = ""
    text: str

class FeedbackItem(BaseModel):
    prediction_id: str
    accepted: bool
    notes: Optional[str] = ""

# ---- Globals + locks ----
index_lock = Lock()
meta_lock = Lock()
faiss_index = None
metadata = []
embed_model = None
llm_pipe = None
risk_bundle = None

# ---- Utility functions ----
def load_index_and_meta():
    global faiss_index, metadata
    if os.path.exists(INDEX_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
        print("Loaded FAISS index:", INDEX_FILE)
    else:
        faiss_index = None
        print("FAISS index not found, start with empty")

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded metadata with {len(metadata)} docs")
    else:
        metadata = []
        print("No metadata file found yet")

def save_index_and_meta():
    global faiss_index, metadata
    with index_lock:
        if faiss_index is not None:
            faiss.write_index(faiss_index, INDEX_FILE)
    with meta_lock:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

def ensure_embedding_model():
    global embed_model
    if embed_model is None:
        print("Loading embedding model (sentence-transformers)...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def ensure_llm():
    global llm_pipe
    if llm_pipe is None:
        print("Loading LLM pipeline (flan-t5-small)... This may take a bit the first time.")
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 0 if (os.environ.get("CUDA_VISIBLE_DEVICES") is not None) else -1
        llm_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device, max_length=256)

def ensure_risk_model():
    global risk_bundle
    if risk_bundle is None:
        if not os.path.exists(RISK_MODEL_FILE):
            raise FileNotFoundError("Risk model not found. Run scripts/train_risk_model.py first.")
        risk_bundle = joblib.load(RISK_MODEL_FILE)

def embed_texts(texts: List[str]) -> np.ndarray:
    ensure_embedding_model()
    embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = embs.astype("float32")
    faiss.normalize_L2(embs)
    return embs

def search_similar(text: str, top_k: int = 5):
    global faiss_index, metadata
    if faiss_index is None or len(metadata) == 0:
        return []
    q = embed_texts([text])
    with index_lock:
        D, I = faiss_index.search(q, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        doc = metadata[idx]
        results.append({"id": doc["id"], "title": doc.get("title", ""), "source": doc.get("source", ""), "snippet": doc.get("text")[:400], "score": float(score)})
    return results

def extract_features_for_model(payload: PredictPayload):
    # Build the feature vector consistent with training script
    age = payload.age if payload.age is not None else 50
    bp = payload.vitals.bp_systolic if (payload.vitals and payload.vitals.bp_systolic) else 130.0
    hr = payload.vitals.hr if (payload.vitals and payload.vitals.hr) else 75.0
    creat = payload.labs.creatinine if (payload.labs and payload.labs.creatinine) else 1.0
    hb = payload.labs.hemoglobin if (payload.labs and payload.labs.hemoglobin) else 13.5
    notes = (payload.notes or "").lower()
    diabetes = 1 if ("diabetes" in notes or "type 2" in notes) else 0
    return [age, bp, hr, creat, hb, diabetes]

def model_predict_probability(payload: PredictPayload):
    ensure_risk_model()
    feats = extract_features_for_model(payload)
    scaler = risk_bundle["scaler"]
    clf = risk_bundle["clf"]
    x = scaler.transform([feats])
    prob = clf.predict_proba(x)[0][1]
    return float(prob)

def generate_explanation(payload: PredictPayload, evidence: List[dict]):
    ensure_llm()
    # simple patient summary
    lines = []
    if payload.age is not None: lines.append(f"Age: {payload.age}")
    if payload.sex: lines.append(f"Sex: {payload.sex}")
    if payload.vitals:
        if payload.vitals.bp_systolic: lines.append(f"Systolic BP: {payload.vitals.bp_systolic}")
        if payload.vitals.hr: lines.append(f"HR: {payload.vitals.hr}")
    if payload.labs:
        if payload.labs.creatinine: lines.append(f"Creatinine: {payload.labs.creatinine}")
        if payload.labs.hemoglobin: lines.append(f"Hemoglobin: {payload.labs.hemoglobin}")
    if payload.notes:
        lines.append(f"Notes: {payload.notes[:300]}")
    patient_summary = "\n".join(lines)

    evidence_text = ""
    for e in evidence:
        evidence_text += f"[{e['id']}] {e['title']} ({e['source']}): {e['snippet']}\n"

    prompt = (
        "You are a helpful clinical assistant. Based on the patient summary and the evidence provided, "
        "write a short (2-6 sentence) explanation of the patient's short-term risk, referencing evidence by id. "
        "Do not give prescriptive treatment; only explain the likely drivers of risk.\n\n"
        "Patient summary:\n"
        f"{patient_summary}\n\n"
        "Evidence:\n"
        f"{evidence_text}\n\n"
        "Return a concise explanation. Also suggest a risk category (Low, Medium, High) on a separate line like: RISK_CATEGORY: High"
    )

    out = llm_pipe(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    # try to split out risk category if the model returns it
    cat_match = re.search(r"RISK_CATEGORY\s*:\s*(Low|Medium|High)", out, re.IGNORECASE)
    category = cat_match.group(1) if cat_match else None
    return out.strip(), category

# ---- Startup: load models/index ----
@app.on_event("startup")
def startup_event():
    load_index_and_meta()
    ensure_embedding_model()
    ensure_risk_model()
    # LLM loading can be slow; we defer loading until first use to speed startup
    print("Startup complete (embedding + risk model loaded). LLM will load on first explain call.")

# ---- API endpoints ----

@app.post("/ingest/literature")
def ingest_literature(items: List[LiteratureItem]):
    """Add new literature docs to metadata + FAISS index (appends)."""
    global metadata, faiss_index
    docs = [i.dict() for i in items]
    texts = [d["text"] for d in docs]
    embs = embed_texts(texts)
    with meta_lock:
        start_idx = len(metadata)
        metadata.extend(docs)
    with index_lock:
        if faiss_index is None:
            d = embs.shape[1]
            faiss_index = faiss.IndexFlatIP(d)
        faiss_index.add(embs)
        save_index_and_meta()
    return {"added": len(docs), "total_docs": len(metadata)}

@app.post("/predict-risk")
def predict_risk(payload: PredictPayload):
    """Main endpoint: returns classical risk score + explanation + evidence"""
    prediction_id = str(uuid.uuid4())
    # 1) classical model probability
    try:
        prob = model_predict_probability(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # 2) retrieve evidence (search using notes + summary)
    query_text = (payload.notes or "") + " " + " ".join([str(payload.age or ""), str(payload.vitals.bp_systolic if payload.vitals else "")])
    evidence = search_similar(query_text, top_k=5)

    # 3) generate explanation with LLM (may load the model lazily)
    try:
        explanation, suggested_category = generate_explanation(payload, evidence)
    except Exception as e:
        explanation = "(explanation generation failed: " + str(e) + ")"
        suggested_category = None

    # compute category by threshold
    if prob >= 0.7:
        category = "High"
    elif prob >= 0.4:
        category = "Medium"
    else:
        category = "Low"

    # response
    resp = {
        "prediction_id": prediction_id,
        "risk_score": prob,
        "risk_category": category,
        "model_suggested_category": suggested_category,
        "explanation": explanation,
        "evidence": evidence,
        "timestamp": int(time.time())
    }
    return resp

@app.get("/evidence/{doc_id}")
def get_evidence(doc_id: str):
    for d in metadata:
        if d.get("id") == doc_id:
            return d
    raise HTTPException(status_code=404, detail="Document not found")

@app.post("/feedback")
def feedback(item: FeedbackItem):
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item.dict()) + "\n")
    return {"status": "ok"}
