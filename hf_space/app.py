import gradio as gr
import pickle, numpy as np, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sts_ai_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
model      = bundle['model']
le         = bundle['label_encoder']
feat_names = bundle['feature_names']

FISH_GENES  = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS = ['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6',
               'COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES   = ['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

TARGETABLE = {
    'ALK':'ALK inhibitor (crizotinib)',
    'NTRK':'TRK inhibitor (larotrectinib)',
    'RET':'RET inhibitor (selpercatinib)',
    'ROS1':'ROS1 inhibitor (crizotinib)',
    'FGFR':'FGFR inhibitor (erdafitinib)',
    'BRAF':'BRAF inhibitor (vemurafenib)',
    'TMB-H':'Immune checkpoint inhibitor',
    'MSI-H':'Immune checkpoint inhibitor',
    'FUS-DDIT3':'Trabectedin (FDA-approved, myxoid liposarcoma)',
    'EWSR1-DDIT3':'Trabectedin (FDA-approved, myxoid liposarcoma)',
}

def predict(age, sex, fish_result, fish_gene,
            rna_result, rna_fusion,
            dna_result, dna_mutations, tmb_high, msi_high):

    feats = {f: 0.0 for f in feat_names}
    try:    feats['age'] = float(age)
    except: feats['age'] = 50.0
    feats['sex_male']      = 1.0 if sex == 'Male' else 0.0
    feats['fish_positive'] = 1.0 if fish_result == 'Positive' else 0.0
    feats['fish_negative'] = 1.0 if fish_result == 'Negative' else 0.0
    feats['fish_done']     = 1.0 if fish_result else 0.0
    for g in FISH_GENES:
        feats['fish_'+g] = 1.0 if g in (fish_gene or '') and fish_result == 'Positive' else 0.0
    feats['rna_positive'] = 1.0 if rna_result == 'Positive' else 0.0
    feats['rna_negative'] = 1.0 if rna_result == 'Negative' else 0.0
    feats['rna_done']     = 1.0 if rna_result else 0.0
    for f in RNA_FUSIONS:
        feats['fusion_'+f.replace('-','_')] = 1.0 if f in (rna_fusion or '') else 0.0
    feats['dna_positive'] = 1.0 if dna_result == 'Positive' else 0.0
    feats['dna_negative'] = 1.0 if dna_result == 'Negative' else 0.0
    feats['dna_done']     = 1.0 if dna_result else 0.0
    for g in DNA_GENES:
        feats['dna_'+g] = 1.0 if g in (dna_mutations or '') else 0.0
    feats['tmb_high'] = 1.0 if tmb_high else 0.0
    feats['msi_high'] = 1.0 if msi_high else 0.0

    X     = np.array([feats[f] for f in feat_names]).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    top3  = np.argsort(probs)[::-1][:3]

    out = "=== DIFFERENTIAL DIAGNOSIS ===\n"
    for rank, i in enumerate(top3):
        out += f"{rank+1}. {le.classes_[i]} ({probs[i]:.1%})\n"

    out += "\n=== THERAPEUTIC TARGETS ===\n"
    found = False
    for gene in ['ALK','NTRK','RET','ROS1','FGFR']:
        if gene in (rna_fusion or '') and rna_result == 'Positive':
            out += f"- {gene} fusion: {TARGETABLE[gene]}\n"; found = True
    for fk in ['FUS-DDIT3','EWSR1-DDIT3']:
        if fk in (rna_fusion or '') and rna_result == 'Positive':
            out += f"- {fk}: {TARGETABLE[fk]}\n"; found = True
    if 'BRAF' in (dna_mutations or ''):
        out += f"- BRAF: {TARGETABLE['BRAF']}\n"; found = True
    if tmb_high:
        out += f"- TMB-High: {TARGETABLE['TMB-H']}\n"; found = True
    if msi_high:
        out += f"- MSI-High: {TARGETABLE['MSI-H']}\n"; found = True
    if not found:
        out += "No actionable targets identified\n"

    out += "\n=== TESTING RECOMMENDATION ===\n"
    conf = float(probs[top3[0]])
    if fish_result == 'Negative' and not rna_result:
        out += "RNA-NGS recommended (FISH negative)\n"
    elif conf < 0.6 and not dna_result:
        out += "DNA-NGS recommended (low confidence)\n"
    elif fish_result == 'Positive' and rna_result == 'Negative':
        out += "Discordant FISH+/RNA-: check RNA quality\n"
    elif fish_result == 'Negative' and rna_result == 'Positive':
        out += "Discordant FISH-/RNA+: prefer RNA-NGS result\n"
    elif fish_result == 'Positive' and rna_result == 'Positive':
        out += "Concordant positive: diagnosis confirmed\n"
    else:
        out += "Current strategy appropriate\n"

    out += "\n[For research use only. Not for clinical diagnosis.]"
    return out


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age (years)", value=50),
        gr.Radio(["Female","Male"], label="Sex", value="Female"),
        gr.Dropdown(["","Positive","Negative"], label="FISH Result", value=""),
        gr.Dropdown(["","DDIT3","MDM2","SS18","EWSR1","ALK","NTRK1","NTRK2","NTRK3","TFE3","CMET","ROS1"],
                    label="FISH Gene", value=""),
        gr.Dropdown(["","Positive","Negative"], label="RNA-NGS Result", value=""),
        gr.Dropdown(["","FUS-DDIT3","EWSR1-DDIT3","SS18-SSX1","SS18-SSX2",
                     "NAB2-STAT6","COL1A1-PDGFB","EML4-ALK","ETV6-NTRK3",
                     "ASPSCR1-TFE3","HEY1-NCOA2"],
                    label="Fusion Partner", value=""),
        gr.Dropdown(["","Positive","Negative"], label="DNA-NGS Result", value=""),
        gr.Textbox(label="DNA Mutations", placeholder="e.g. TP53(mut)/MDM2(amp)"),
        gr.Checkbox(label="TMB-High", value=False),
        gr.Checkbox(label="MSI-High", value=False),
    ],
    outputs=gr.Textbox(label="Results", lines=20),
    title="STS-Molecular-AI",
    description="Soft Tissue Sarcoma Molecular Diagnostic Decision Support | AUC=0.780 | For research use only",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
