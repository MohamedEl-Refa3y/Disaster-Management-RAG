import os
import json
import re
import faiss
import numpy as np
import torch
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ================== CONFIG ==================
BASE = os.path.dirname(__file__)
INDEX_FILE = os.path.join(BASE, "hailstate_index.faiss")
METADATA_FILE = os.path.join(BASE, "hailstate_metadata.json")

EMBED_MODEL_NAME = "jinaai/jina-embeddings-v3"
MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
TOP_K = 5

# ================== EMBEDDING MODEL ==================
embed_model = SentenceTransformer(
    EMBED_MODEL_NAME,
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()

# ================== LOAD FAISS & METADATA ==================
index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ================== LLM ==================
@lru_cache(maxsize=1)
def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map=("auto" if torch.cuda.is_available() else None),
    )
    mdl.eval()
    return tok, mdl

_tokenizer, _model = load_llm()
generator = pipeline("text-generation", model=_model, tokenizer=_tokenizer)

# ================== HELPERS ==================
def clean_llm_output(gen: str) -> str:
    """Extract the answer between <answer> tags if present"""
    if not gen:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", gen, flags=re.S)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"<answer>(.*)", gen, flags=re.S)
    if m2:
        return m2.group(1).strip()
    return gen.strip()

def format_record_for_context(rec: dict) -> str:
    """Convert metadata record into a clean Arabic context string for the prompt"""
    global metadata

    typ = rec.get("_type", "")
    parts = [f"النوع: {typ}"]

    def add_coords(parts, rec):
        lat = rec.get('Latitude', '?')
        lon = rec.get('Longitude', '?')
        parts.append(f"الإحداثيات: {lat}, {lon}")

    if typ == "governorate":
        parts.append(f"الاسم: {rec.get('governoratename')}")
        parts.append(f"عدد السكان: {rec.get('Population', 'غير معروف')} (ذكور {rec.get('MaleCount','?')}, إناث {rec.get('FemaleCount','?')})")

        province_id = rec.get("ProvinceId") or rec.get("ID")
        if province_id:
            province_id_str = str(province_id)
            count_govs_in_province = sum(
                1 for g in metadata
                if g.get("_type") == "governorate" and str(g.get("ProvinceId")) == province_id_str
            )
            if count_govs_in_province > 0:
                parts.append(f"عدد المحافظات في هذه المنطقة: {count_govs_in_province}")

        parts.append(f"عدد الموظفين: {rec.get('GovernmentEmployeesCount', 'غير معروف')}")
        parts.append(f"عدد ملاك المنازل: {rec.get('HouseholdCount', 'غير معروف')}")
        add_coords(parts, rec)

    elif typ == "hospital":
        parts.append(f"الاسم: {rec.get('hospitalname')}")
        parts.append(f"عدد الأطباء: {rec.get('TotalDoctors', '?')}")
        parts.append(f"عدد الممرضات: {rec.get('TotalNurses', '?')}")
        parts.append(f"الأسرة الكلية: {rec.get('TotalBedsCount', '?')}")
        parts.append(f"الأسرة الشاغرة: {rec.get('VacantBedsCount', '?')}")
        parts.append(f"سيارات الإسعاف: {rec.get('AmbulanceCount', '?')}")
        add_coords(parts, rec)

    elif typ == "school":
        parts.append(f"الاسم: {rec.get('schoolname')}")
        parts.append(f"عدد الطلاب: {rec.get('StudentsMale','?')} ذكور، {rec.get('StudentsFemale','?')} إناث")
        parts.append(f"عدد المعلمين: {rec.get('TeacherCount','?')}")

        province_id = rec.get("ProvinceId")
        if province_id:
            province_id_str = str(province_id)
            count_schools_in_province = sum(
                1 for s in metadata
                if s.get("_type") == "school" and str(s.get("ProvinceId")) == province_id_str
            )
            if count_schools_in_province > 0:
                parts.append(f"عدد المدارس في هذه المحافظة: {count_schools_in_province}")

        add_coords(parts, rec)

    return " - ".join([str(p) for p in parts if p])

#=================Allocation / distance==================
import re
import math

def extract_lat_lon_and_patients(text):
    """Extract coordinates and number of patients using regex."""
    coord_match = re.search(r"\(?\s*([0-9]+\.[0-9]+)\s*,\s*([0-9]+\.[0-9]+)\s*\)?", text)
    patient_match = re.search(r"(\d+)\s*(?:مريض|مرضى|اصابات|حالة)", text)

    lat = float(coord_match.group(1)) if coord_match else None
    lon = float(coord_match.group(2)) if coord_match else None
    num_patients = int(patient_match.group(1)) if patient_match else None

    return lat, lon, num_patients


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in KM."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def search_nearest_hospital_in_metadata(lat, lon):
    """Scan metadata directly for hospitals and find the nearest one."""
    hospitals = [rec for rec in metadata if rec.get("_type") == "hospital"]
    nearest = None
    min_distance = float('inf')

    for h in hospitals:
        h_lat = h.get("Latitude")
        h_lon = h.get("Longitude")
        if not h_lat or not h_lon:
            continue

        dist = haversine(lat, lon, float(h_lat), float(h_lon))
        if dist < min_distance:
            min_distance = dist
            nearest = (h, dist)

    return nearest

# ================== UNIFIED PROMPT ==================
def build_unified_prompt(user_query: str, retrieved_contexts: list[str]) -> str:
    """
    Creates a single structured prompt to handle normal, geographic, and conditional queries.
    """
    prompt = f"""
أنت مساعد ذكي ومتخصص في تحليل البيانات الجغرافية والطبية والتعليمية والإحصائية.
تجيب فقط باللغة العربية الفصحى.
مهمتك هي تحليل سؤال المستخدم، فهم قصده (عادي أو جغرافي أو شرطي)، ثم استخراج الإجابة بدقة اعتمادًا على البيانات المقدمة فقط.

👇 اتبع الخطوات التالية بدقة:
1️⃣ - اقرأ سؤال المستخدم جيدًا، وحدد نوعه:
   - سؤال عادي: مثل "كم عدد السكان في محافظة الشنان؟"
   - سؤال جغرافي: مثل "ما هي أقرب مستشفى للموقع (41.7 , 27.5)؟"
   - سؤال شرطي: مثل "إذا كان هناك 10 مرضى في الموقع (41.7 , 27.5) ما هي المستشفى الأنسب؟"

2️⃣ - استخرج أي إحداثيات أو أعداد مرضى مذكورة في السؤال إن وجدت.

3️⃣ - استخدم البيانات التالية فقط (ولا تضف أي معلومات من خارجها) للإجابة:
{chr(10).join(retrieved_contexts)}

4️⃣ - إذا لم تجد أي إجابة واضحة أو مناسبة في البيانات، أجب فقط بـ:
غير متوفر


📝 سؤال المستخدم:
```{user_query}```

<answer>
""".strip()
    return prompt

# ================== RAG CORE ==================
def rag_query(user_query: str):
    # 1️⃣ Extract coordinates from user question
    lat, lon, num_patients = extract_lat_lon_and_patients(user_query)

    # 2️⃣ If geo → direct metadata search
    if lat and lon:
        nearest = search_nearest_hospital_in_metadata(lat, lon)
        if not nearest:
            return "⚠️ لم يتم العثور على مستشفى قريبة."

        hospital, distance_km = nearest
        context = [
            f"الاسم: {hospital['hospitalname']}",
            f"عدد الأسرة الكلية: {hospital.get('TotalBedsCount', 'غير معروف')}",
            f"عدد الأسرة الشاغرة: {hospital.get('VacantBedsCount', 'غير معروف')}",
            f"الموقع: ({hospital['Latitude']}, {hospital['Longitude']})",
            f"المسافة من الموقع المطلوب: {round(distance_km, 2)} كم"
        ]

        prompt = build_unified_prompt(user_query, context)
        gen = generator(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        return clean_llm_output(gen)

    # 3️⃣ Else → normal FAISS retrieval
    q_emb = embed_model.encode([user_query], convert_to_numpy=True)
    q_emb = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-9)
    _, I = index.search(q_emb.astype("float32"), TOP_K)
    contexts = [format_record_for_context(metadata[int(i)]) for i in I[0] if i != -1]

    if not contexts:
        return "غير متوفر."

    prompt = build_unified_prompt(user_query, contexts)
    gen = generator(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    return clean_llm_output(gen)
BASE = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE, "static")

# ================== FASTAPI SERVER ==================
app = FastAPI(
    title="RAG Chat API",
    description="Arabic RAG Model for Web & Java/C++ Clients",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return RedirectResponse("/chat")

@app.get("/chat")
def get_chat_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = rag_query(req.query)
    return {"answer": answer or "غير متوفر."}

# ================== MAIN ==================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
