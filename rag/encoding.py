import os
import pyodbc
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from decimal import Decimal
import unicodedata
import re

# ================== CONFIG ==================
SQL_SERVER = "LAPTOP-M1KPSIRM"
SQL_DATABASE = "HailState"
DRIVER = "{ODBC Driver 17 for SQL Server}"

MODEL_NAME = "jinaai/jina-embeddings-v3"
BASE = os.path.dirname(__file__)
INDEX_FILE = os.path.join(BASE, "hailstate_index.faiss")
METADATA_FILE = os.path.join(BASE, "hailstate_metadata.json")

# ================== DB CONNECTION ==================
conn_str = f"Driver={DRIVER};Server={SQL_SERVER};Database={SQL_DATABASE};Trusted_Connection=yes;"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# ================== HELPERS ==================
def ar_norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    s = s.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
    s = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def fetch_table(table, id_field, name_field, extra_fields_sql=""):
    sql = f"""
        SELECT {id_field}, {name_field} {extra_fields_sql}
        FROM {table}
        WHERE {name_field} IS NOT NULL AND {name_field} != ''
    """
    cursor.execute(sql)
    cols = [c[0] for c in cursor.description]
    results = []
    for row in cursor.fetchall():
        rec = dict(zip(cols, row))
        # 👇 نحافظ على الدقة العشرية للإحداثيات
        if "Latitude" in rec and rec["Latitude"] is not None:
            rec["Latitude"] = float(rec["Latitude"])
        if "Longitude" in rec and rec["Longitude"] is not None:
            rec["Longitude"] = float(rec["Longitude"])
        results.append(rec)
    return results

def convert_types(obj):
    # ✅ نُبقي Decimal كـ float وليس int
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(x) for x in obj]
    return obj

# ================== FETCH DATA ==================
print("📥 Fetching data from SQL...")

# Governorates
governorates = fetch_table(
    "governorates", "ID", "governoratename",
    extra_fields_sql=", Latitude, Longitude, Population, MaleCount, FemaleCount, HouseholdCount, GovernmentEmployeesCount"
)
gov_dict = {g["ID"]: g for g in governorates}
for g in governorates:
    g["_type"] = "governorate"
    g["governoratename"] = ar_norm(g["governoratename"])

# Schools
schools_raw = fetch_table(
    "Schools", "ID", "schoolname",
    extra_fields_sql=", Latitude, Longitude, StudentsFemale, StudentsMale, TeacherCount, ProvinceId"
)
schools = []
for s in schools_raw:
    s["_type"] = "school"
    s["schoolname"] = ar_norm(s["schoolname"])
    prov = gov_dict.get(s["ProvinceId"], {})
    s["GovernorateName"] = prov.get("governoratename", "غير معروف")
    s["GovernoratePopulation"] = prov.get("Population", 0)
    schools.append(s)

# Hospitals
hospitals_raw = fetch_table(
    "Hospitals", "ID", "hospitalname",
    extra_fields_sql=", Latitude, Longitude, TotalBedsCount, AmbulanceCount, PharmacyCount, ProvinceId, VacantBedsCount"
)
hospitals = []
for h in hospitals_raw:
    h["_type"] = "hospital"
    h["hospitalname"] = ar_norm(h["hospitalname"])
    prov = gov_dict.get(h["ProvinceId"], {})
    h["GovernorateName"] = prov.get("governoratename", "غير معروف")
    h["GovernoratePopulation"] = prov.get("Population", 0)
    hospitals.append(h)

# Clinics (optional)
clinics_raw = fetch_table("Clinics", "ID", "Name")
for c in clinics_raw:
    c["_type"] = "clinic"
    c["Name"] = ar_norm(c["Name"])

# HospitalClinics link
hospital_clinics = fetch_table(
    "HospitalClinics", "HospitalId", "ClinicId",
    extra_fields_sql=", NurseCount, DoctorsCount"
)
clinics_dict = {c["ID"]: c for c in clinics_raw}
for hc in hospital_clinics:
    hosp = next((h for h in hospitals if h["ID"] == hc["HospitalId"]), None)
    clinic = clinics_dict.get(hc["ClinicId"])
    if hosp and clinic:
        if "Clinics" not in hosp:
            hosp["Clinics"] = []
        hosp["Clinics"].append({
            "ClinicName": clinic["Name"],
            "NurseCount": hc.get("NurseCount", 0),
            "DoctorsCount": hc.get("DoctorsCount", 0)
        })

# Total Doctors/Nurses
for h in hospitals:
    clinics = h.get("Clinics", [])
    h["TotalDoctors"] = sum(c.get("DoctorsCount", 0) for c in clinics)
    h["TotalNurses"] = sum(c.get("NurseCount", 0) for c in clinics)

# ================== BUILD TEXTS ==================
print("🧠 Building embedding texts...")

all_items = governorates + schools + hospitals + clinics_raw
texts = []

for rec in all_items:
    typ = rec.get("_type")
    if typ == "governorate":
        txt = f"المحافظة {rec['governoratename']}، عدد السكان {rec.get('Population',0)}، الذكور {rec.get('MaleCount',0)}، الإناث {rec.get('FemaleCount',0)}."
    elif typ == "school":
        txt = f"المدرسة {rec['schoolname']} في محافظة {rec['GovernorateName']}، عدد الطلاب {rec.get('StudentsMale',0)+rec.get('StudentsFemale',0)}، عدد المعلمين {rec.get('TeacherCount',0)}."
    elif typ == "hospital":
        txt = f"المستشفى {rec['hospitalname']} في محافظة {rec['GovernorateName']}، الأسرة {rec.get('TotalBedsCount',0)}، الشاغرة {rec.get('VacantBedsCount',0)}، سيارات الإسعاف {rec.get('AmbulanceCount',0)}، الصيدليات {rec.get('PharmacyCount',0)}، عدد الأطباء {rec.get('TotalDoctors',0)}، عدد الممرضات {rec.get('TotalNurses',0)}، يقع في خط العرض {rec.get('Latitude','')} وخط الطول {rec.get('Longitude','')}."
    elif typ == "clinic":
        txt = f"العيادة {rec['Name']}."
    else:
        txt = ""
    texts.append(txt)

# ================== EMBEDDING ==================
print("🔸 Encoding texts...")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)

# ================== FAISS INDEX ==================
print("📊 Building FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype("float32"))
faiss.write_index(index, INDEX_FILE)

# ================== SAVE METADATA ==================
metadata = [convert_types(rec) for rec in all_items]
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Saved index to {INDEX_FILE}")
print(f"✅ Saved metadata to {METADATA_FILE}")
print(f"Total records indexed: {len(metadata)}")

