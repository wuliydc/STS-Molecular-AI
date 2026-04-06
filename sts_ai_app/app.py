"""
STS-Molecular-AI: FastAPI backend
Run: uvicorn app:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pickle, numpy as np, os

app = FastAPI(title="STS-Molecular-AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root():
    index = os.path.join(static_dir, 'index.html')
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "STS-Molecular-AI API", "docs": "/docs"}

# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'sts_ai_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
model   = bundle['model']
le      = bundle['label_encoder']
feat_names = bundle['feature_names']

FISH_GENES  = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS = ['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6',
               'COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES   = ['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

TARGETABLE = {
    'ALK':         'ALK inhibitor (crizotinib/alectinib)',
    'NTRK':        'TRK inhibitor (larotrectinib/entrectinib)',
    'RET':         'RET inhibitor (selpercatinib)',
    'ROS1':        'ROS1 inhibitor (crizotinib)',
    'FGFR':        'FGFR inhibitor (erdafitinib)',
    'BRAF':        'BRAF inhibitor (vemurafenib)',
    'TMB-H':       'Immune checkpoint inhibitor',
    'MSI-H':       'Immune checkpoint inhibitor',
    # Trabectedin: FDA-approved for unresectable/metastatic liposarcoma and leiomyosarcoma
    'FUS-DDIT3':   'Trabectedin (FDA-approved; DDIT3-rearranged myxoid liposarcoma)',
    'EWSR1-DDIT3': 'Trabectedin (FDA-approved; DDIT3-rearranged myxoid liposarcoma)',
}

class PatientInput(BaseModel):
    age: float = 50.0
    sex: str = "Female"
    fish_result: str = ""        # "阳性" / "阴性" / ""
    fish_gene: str = ""          # e.g. "DDIT3"
    rna_result: str = ""
    rna_fusion: str = ""         # e.g. "FUS-DDIT3"
    dna_result: str = ""
    dna_mutations: str = ""      # e.g. "TP53(mut)/MDM2(amp)"
    tmb_high: bool = False
    msi_high: bool = False

def build_feat_vector(inp: PatientInput):
    feats = {f: 0.0 for f in feat_names}
    feats['age']       = inp.age
    feats['sex_male']  = 1.0 if inp.sex.lower() in ['male','男'] else 0.0
    feats['fish_positive'] = 1.0 if inp.fish_result == '阳性' else 0.0
    feats['fish_negative'] = 1.0 if inp.fish_result == '阴性' else 0.0
    feats['fish_done']     = 1.0 if inp.fish_result else 0.0
    for g in FISH_GENES:
        feats['fish_' + g] = 1.0 if g in inp.fish_gene and inp.fish_result == '阳性' else 0.0
    feats['rna_positive'] = 1.0 if inp.rna_result == '阳性' else 0.0
    feats['rna_negative'] = 1.0 if inp.rna_result == '阴性' else 0.0
    feats['rna_done']     = 1.0 if inp.rna_result else 0.0
    for f in RNA_FUSIONS:
        feats['fusion_' + f.replace('-','_')] = 1.0 if f in inp.rna_fusion else 0.0
    feats['dna_positive'] = 1.0 if inp.dna_result == '阳性' else 0.0
    feats['dna_negative'] = 1.0 if inp.dna_result == '阴性' else 0.0
    feats['dna_done']     = 1.0 if inp.dna_result else 0.0
    for g in DNA_GENES:
        feats['dna_' + g] = 1.0 if g in inp.dna_mutations else 0.0
    feats['tmb_high'] = 1.0 if inp.tmb_high else 0.0
    feats['msi_high'] = 1.0 if inp.msi_high else 0.0
    return np.array([feats[f] for f in feat_names]).reshape(1, -1)

def get_targets(inp: PatientInput):
    targets = []
    for gene in ['ALK','NTRK','RET','ROS1','FGFR']:
        if gene in inp.rna_fusion and inp.rna_result == '阳性':
            targets.append({'gene': gene, 'drug': TARGETABLE[gene]})
    # Trabectedin for DDIT3-rearranged liposarcoma (FDA-approved)
    if inp.rna_result == '阳性':
        for fusion_key in ['FUS-DDIT3', 'EWSR1-DDIT3']:
            if fusion_key in inp.rna_fusion:
                targets.append({'gene': fusion_key, 'drug': TARGETABLE[fusion_key]})
    if 'BRAF' in inp.dna_mutations:
        targets.append({'gene': 'BRAF', 'drug': TARGETABLE['BRAF']})
    if inp.tmb_high:
        targets.append({'gene': 'TMB-H', 'drug': TARGETABLE['TMB-H']})
    if inp.msi_high:
        targets.append({'gene': 'MSI-H', 'drug': TARGETABLE['MSI-H']})
    return targets

def get_recommendation(inp: PatientInput, top_diagnosis: str, confidence: float):
    recs = []
    if inp.fish_result == '阴性' and not inp.rna_result:
        recs.append('RNA-NGS recommended: FISH negative, fusion gene may be missed')
    if confidence < 0.6 and not inp.dna_result:
        recs.append('DNA-NGS recommended: low diagnostic confidence, mutation profiling may help')
    if inp.rna_result == '阳性' and inp.fish_result == '阳性':
        recs.append('Both FISH and RNA-NGS positive: diagnosis confirmed, DNA-NGS optional')
    if inp.fish_result == '阳性' and inp.rna_result == '阴性':
        recs.append('Discordant (FISH+/RNA-): consider intergenic fusion or RNA quality issue')
    if inp.fish_result == '阴性' and inp.rna_result == '阳性':
        recs.append('Discordant (FISH-/RNA+): FISH probe design gap, RNA-NGS result preferred')
    if not recs:
        recs.append('Current testing strategy is appropriate')
    return recs

@app.post("/predict")
def predict(inp: PatientInput):
    X = build_feat_vector(inp)
    probs = model.predict_proba(X)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    diagnoses = [
        {'diagnosis': le.classes_[i], 'probability': round(float(probs[i]), 4)}
        for i in top3_idx
    ]
    top_diag   = diagnoses[0]['diagnosis']
    confidence = diagnoses[0]['probability']
    targets    = get_targets(inp)
    recs       = get_recommendation(inp, top_diag, confidence)
    return {
        'top_diagnosis':    top_diag,
        'confidence':       round(confidence, 4),
        'differential':     diagnoses,
        'therapeutic_targets': targets,
        'testing_recommendation': recs,
        'model_version':    '1.0.0',
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "Logistic Regression",
            "n_classes": len(le.classes_), "classes": list(le.classes_)}
