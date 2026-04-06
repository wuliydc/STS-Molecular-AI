import numpy as np, shap, csv
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

valid = [r for r in data if r['肿瘤类型'] not in ['待明确','','良性肿瘤'] and r['检测方法组合'] != '']
tumor_counts = Counter(r['肿瘤类型'] for r in valid)
top_tumors = [t for t, c in tumor_counts.most_common() if c >= 15]
valid = [r for r in valid if r['肿瘤类型'] in top_tumors]

FISH_GENES  = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS = ['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6',
               'COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES   = ['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

def build_features(row):
    feats = {}
    try:    feats['age'] = float(row['年龄'])
    except: feats['age'] = 50.0
    feats['sex_male']     = 1 if row['性别'] == '男' else 0
    fish_res              = row['FISH结果']
    feats['fish_positive']= 1 if fish_res == '阳性' else 0
    feats['fish_negative']= 1 if fish_res == '阴性' else 0
    feats['fish_done']    = 1 if fish_res else 0
    for g in FISH_GENES:
        feats['fish_' + g] = 1 if g in row.get('检测方法组合','') and fish_res == '阳性' else 0
    rna_res    = row['RNA_NGS结果']
    rna_fusion = row['融合伴侣']
    feats['rna_positive'] = 1 if rna_res == '阳性' else 0
    feats['rna_negative'] = 1 if rna_res == '阴性' else 0
    feats['rna_done']     = 1 if rna_res else 0
    for f in RNA_FUSIONS:
        key = 'fusion_' + f.replace('-', '_')
        feats[key] = 1 if f in rna_fusion else 0
    dna_res = row['DNA_NGS结果']
    dna_mut = row['DNA突变']
    feats['dna_positive'] = 1 if dna_res == '阳性' else 0
    feats['dna_negative'] = 1 if dna_res == '阴性' else 0
    feats['dna_done']     = 1 if dna_res else 0
    for g in DNA_GENES:
        feats['dna_' + g] = 1 if g in dna_mut else 0
    feats['tmb_high'] = 1 if 'TMB-H' in row.get('治疗靶点','') else 0
    feats['msi_high'] = 1 if 'MSI-H' in row.get('治疗靶点','') else 0
    return feats

rows_feat  = [build_features(r) for r in valid]
feat_names = list(rows_feat[0].keys())
X = np.array([[r[f] for f in feat_names] for r in rows_feat])
le = LabelEncoder()
y  = le.fit_transform([r['肿瘤类型'] for r in valid])

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X, y)
sv = shap.TreeExplainer(rf).shap_values(X[:20])
print('shap_values type:', type(sv))
if isinstance(sv, list):
    print('list len:', len(sv), 'each shape:', np.array(sv[0]).shape)
else:
    print('array shape:', np.array(sv).shape)
print('feat_names len:', len(feat_names), 'X cols:', X.shape[1])
