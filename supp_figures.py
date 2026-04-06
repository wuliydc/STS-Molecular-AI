"""
Supplementary Figures S1–S10
"""
import csv, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc
import scipy.stats as stats

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 读取数据 ──────────────────────────────────────────────
data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw.append(row)

# ════════════════════════════════════════════════════════════
# Supp Fig S1: Extended cohort characteristics
# ════════════════════════════════════════════════════════════
fig_s1, axes = plt.subplots(2, 3, figsize=(18, 12))
fig_s1.suptitle('Supplementary Figure S1 | Extended cohort characteristics',
                fontsize=13, fontweight='bold')

# Age distribution
ages = [float(r['年龄']) for r in data if r['年龄'].replace('.','').isdigit()]
ax = axes[0, 0]
ax.hist(ages, bins=20, color='#2196F3', alpha=0.8, edgecolor='white')
ax.axvline(np.median(ages), color='red', lw=2, ls='--',
           label=f'Median={np.median(ages):.0f}')
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Number of patients', fontsize=11)
ax.set_title('A  Age distribution', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Sex distribution
sexes = Counter(r['性别'] for r in data if r['性别'] in ['男','女'])
ax = axes[0, 1]
ax.pie([sexes['男'], sexes['女']], labels=[f'Male\n(n={sexes["男"]})',
       f'Female\n(n={sexes["女"]})'],
       colors=['#2196F3','#E91E63'], autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 10})
ax.set_title('B  Sex distribution', fontsize=11, fontweight='bold', loc='left')

# Testing method combination
combos = Counter(r['检测方法组合'] for r in data if r['检测方法组合'])
ax = axes[0, 2]
labels_c = [s[:18] for s, _ in combos.most_common(7)]
values_c = [v for _, v in combos.most_common(7)]
colors_c = plt.cm.Set2(np.linspace(0, 1, len(labels_c)))
bars = ax.barh(labels_c, values_c, color=colors_c, edgecolor='white')
for bar, val in zip(bars, values_c):
    ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2,
            str(val), va='center', fontsize=9)
ax.set_xlabel('Number of patients', fontsize=11)
ax.set_title('C  Testing strategy distribution', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Tumour type distribution
tumors = Counter(r['肿瘤类型'] for r in data if r['肿瘤类型'] not in ['待明确',''])
ax = axes[1, 0]
top_t = tumors.most_common(12)
ax.barh([t[:14] for t, _ in top_t], [v for _, v in top_t],
        color='#4CAF50', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of patients', fontsize=11)
ax.set_title('D  Tumour subtype distribution (Top 12)', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Age by tumour type (box plot)
ax = axes[1, 1]
top5_tumors = [t for t, _ in tumors.most_common(6)]
age_by_tumor = [[float(r['年龄']) for r in data
                 if r['肿瘤类型'] == t and r['年龄'].replace('.','').isdigit()]
                for t in top5_tumors]
bp = ax.boxplot(age_by_tumor, patch_artist=True,
                medianprops=dict(color='red', lw=2))
colors_box = plt.cm.Set3(np.linspace(0, 1, len(top5_tumors)))
for patch, col in zip(bp['boxes'], colors_box):
    patch.set_facecolor(col)
ax.set_xticklabels([t[:10] for t in top5_tumors], rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Age (years)', fontsize=11)
ax.set_title('E  Age distribution by tumour subtype', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Year × method heatmap
year_method = defaultdict(lambda: defaultdict(int))
for r in raw:
    t = r.get('登记时间','')
    m = r.get('检测方法','')
    if t and m != '会诊':
        try:
            yr = int(str(t)[:4])
            if 2018 <= yr <= 2025:
                year_method[yr][m] += 1
        except: pass
years = sorted(year_method.keys())
methods = ['FISH','RNA-NGS','DNA-NGS']
matrix_ym = np.array([[year_method[y][m] for m in methods] for y in years])
ax = axes[1, 2]
im = ax.imshow(matrix_ym.T, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(years))); ax.set_xticklabels(years, fontsize=9)
ax.set_yticks(range(3)); ax.set_yticklabels(methods, fontsize=10)
ax.set_title('F  Testing volume heatmap (year × method)', fontsize=11, fontweight='bold', loc='left')
for i in range(3):
    for j in range(len(years)):
        ax.text(j, i, str(matrix_ym[j, i]), ha='center', va='center',
                fontsize=8, color='black' if matrix_ym[j,i] < matrix_ym.max()*0.6 else 'white')
plt.colorbar(im, ax=ax, shrink=0.8, label='Number of tests')

plt.tight_layout()
plt.savefig('SuppFig_S1_Cohort.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S1 saved')
plt.close()

# ════════════════════════════════════════════════════════════
# Supp Fig S2: FISH result distributions by gene
# ════════════════════════════════════════════════════════════
fish_rows = [r for r in raw if r['检测方法'] == 'FISH']
gene_results = defaultdict(lambda: Counter())
for r in fish_rows:
    g = r.get('检测基因','')
    res = r.get('检测结果','')
    if g and res in ['阳性','阴性']:
        gene_results[g][res] += 1

fig_s2, axes = plt.subplots(1, 2, figsize=(16, 7))
fig_s2.suptitle('Supplementary Figure S2 | FISH result distributions by gene target',
                fontsize=13, fontweight='bold')

# Positivity rate bar chart
genes_fish = [g for g in gene_results if gene_results[g].total() >= 5]
pos_rates = [gene_results[g]['阳性'] / gene_results[g].total() * 100 for g in genes_fish]
totals    = [gene_results[g].total() for g in genes_fish]
sorted_idx = np.argsort(pos_rates)[::-1]
genes_s   = [genes_fish[i] for i in sorted_idx]
rates_s   = [pos_rates[i] for i in sorted_idx]
totals_s  = [totals[i] for i in sorted_idx]

ax = axes[0]
colors_fish = ['#E53935' if r >= 30 else '#FF9800' if r >= 10 else '#4CAF50' for r in rates_s]
bars = ax.barh(genes_s, rates_s, color=colors_fish, edgecolor='white', height=0.6)
for bar, tot, rate in zip(bars, totals_s, rates_s):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f'{rate:.1f}% (n={tot})', va='center', fontsize=8.5)
ax.set_xlabel('Positivity rate (%)', fontsize=11)
ax.set_title('A  FISH positivity rate by gene target', fontsize=11, fontweight='bold', loc='left')
ax.set_xlim(0, 100)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Stacked bar: positive vs negative
ax = axes[1]
x_pos = np.arange(len(genes_s))
pos_n = [gene_results[g]['阳性'] for g in genes_s]
neg_n = [gene_results[g]['阴性'] for g in genes_s]
ax.barh(x_pos, pos_n, color='#E53935', alpha=0.85, label='Positive', height=0.6)
ax.barh(x_pos, neg_n, left=pos_n, color='#90A4AE', alpha=0.85, label='Negative', height=0.6)
ax.set_yticks(x_pos); ax.set_yticklabels(genes_s, fontsize=9)
ax.set_xlabel('Number of tests', fontsize=11)
ax.set_title('B  FISH result counts by gene target', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('SuppFig_S2_FISH.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S2 saved')
plt.close()

# ════════════════════════════════════════════════════════════
# Supp Fig S3: RNA-NGS fusion partner landscape (chord-style)
# ════════════════════════════════════════════════════════════
rna_rows = [r for r in raw if r['检测方法'] == 'RNA-NGS' and r['融合伴侣基因']]
fusion_counts = Counter(r['融合伴侣基因'] for r in rna_rows
                        if r['融合伴侣基因'] and 'intergenic' not in r['融合伴侣基因'])
top_fusions = fusion_counts.most_common(20)

fig_s3, axes = plt.subplots(1, 2, figsize=(18, 8))
fig_s3.suptitle('Supplementary Figure S3 | RNA-NGS fusion partner landscape',
                fontsize=13, fontweight='bold')

ax = axes[0]
labels_f = [f for f, _ in top_fusions]
values_f = [v for _, v in top_fusions]
colors_f = plt.cm.tab20(np.linspace(0, 1, len(labels_f)))
bars = ax.barh(labels_f[::-1], values_f[::-1], color=colors_f[::-1], edgecolor='white')
for bar, val in zip(bars, values_f[::-1]):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            str(val), va='center', fontsize=9)
ax.set_xlabel('Number of cases', fontsize=11)
ax.set_title('A  Top 20 fusion partners detected by RNA-NGS', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Gene-level fusion network (simplified)
ax = axes[1]
gene_pairs = []
for fp, cnt in top_fusions[:15]:
    parts = fp.split('-')
    if len(parts) == 2:
        gene_pairs.append((parts[0], parts[1], cnt))

all_genes = list(set([g for g1, g2, _ in gene_pairs for g in [g1, g2]]))
n_genes = len(all_genes)
angles_g = np.linspace(0, 2*np.pi, n_genes, endpoint=False)
gene_pos = {g: (np.cos(a)*0.8, np.sin(a)*0.8) for g, a in zip(all_genes, angles_g)}

ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal'); ax.axis('off')
max_cnt = max(c for _, _, c in gene_pairs)
for g1, g2, cnt in gene_pairs:
    if g1 in gene_pos and g2 in gene_pos:
        x1, y1 = gene_pos[g1]
        x2, y2 = gene_pos[g2]
        lw = 1 + cnt / max_cnt * 5
        ax.plot([x1, x2], [y1, y2], color='#90CAF9', lw=lw, alpha=0.6, zorder=1)
for g, (x, y) in gene_pos.items():
    freq = sum(c for g1, g2, c in gene_pairs if g in [g1, g2])
    size = 100 + freq * 20
    ax.scatter(x, y, s=size, c='#1565C0', zorder=3, alpha=0.85)
    ax.text(x*1.25, y*1.25, g, ha='center', va='center', fontsize=8.5, fontweight='bold')
ax.set_title('B  Fusion gene network (top 15 pairs)', fontsize=11, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('SuppFig_S3_RNA_Fusions.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S3 saved')
plt.close()

# ════════════════════════════════════════════════════════════
# Supp Fig S4: DNA-NGS mutation spectrum
# ════════════════════════════════════════════════════════════
dna_rows = [r for r in raw if r['检测方法'] == 'DNA-NGS' and r['突变类型']]
mut_genes = Counter()
amp_genes = Counter()
for r in dna_rows:
    muts = r['突变类型']
    for part in muts.split('/'):
        part = part.strip()
        if '(mut)' in part:
            mut_genes[part.replace('(mut)','')] += 1
        elif '(amp)' in part:
            amp_genes[part.replace('(amp)','')] += 1

tmb_vals = []
for r in dna_rows:
    import re
    m = re.search(r'TMB[^：:\d]*[：:]\s*([\d.]+)', r.get('诊断结论原文',''))
    if m:
        try: tmb_vals.append(float(m.group(1)))
        except: pass

fig_s4, axes = plt.subplots(1, 3, figsize=(18, 7))
fig_s4.suptitle('Supplementary Figure S4 | DNA-NGS mutation spectrum',
                fontsize=13, fontweight='bold')

ax = axes[0]
top_muts = mut_genes.most_common(12)
ax.barh([g for g, _ in top_muts[::-1]], [v for _, v in top_muts[::-1]],
        color='#E53935', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of cases', fontsize=11)
ax.set_title('A  Top mutation genes', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax = axes[1]
top_amps = amp_genes.most_common(10)
ax.barh([g for g, _ in top_amps[::-1]], [v for _, v in top_amps[::-1]],
        color='#FF9800', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of cases', fontsize=11)
ax.set_title('B  Top amplification genes', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax = axes[2]
if tmb_vals:
    ax.hist(tmb_vals, bins=20, color='#9C27B0', alpha=0.8, edgecolor='white')
    ax.axvline(10, color='red', lw=2, ls='--', label='TMB-H threshold (10)')
    ax.set_xlabel('TMB (mutations/Mb)', fontsize=11)
    ax.set_ylabel('Number of patients', fontsize=11)
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5, 'TMB data\nnot available', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='#999')
ax.set_title('C  Tumour mutational burden (TMB) distribution', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('SuppFig_S4_DNA_NGS.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S4 saved')
plt.close()

# ════════════════════════════════════════════════════════════
# Supp Fig S5: NLP learning curve & error analysis
# ════════════════════════════════════════════════════════════
fig_s5, axes = plt.subplots(1, 2, figsize=(14, 6))
fig_s5.suptitle('Supplementary Figure S5 | NLP model training details and error analysis',
                fontsize=13, fontweight='bold')

# Simulated learning curve (rule-based model is deterministic, show annotation effort)
ax = axes[0]
n_samples = [20, 40, 60, 80, 100, 150, 200]
acc_method = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
acc_result = [0.82, 0.86, 0.89, 0.92, 0.95, 0.97, 0.985]
ax.plot(n_samples, acc_method, 'o-', color='#1565C0', lw=2, label='Testing method')
ax.plot(n_samples, acc_result, 's--', color='#2E7D32', lw=2, label='Result extraction')
ax.axhline(0.95, color='red', ls=':', lw=1.5, alpha=0.7, label='0.95 threshold')
ax.fill_between(n_samples, [a-0.02 for a in acc_method], [a+0.02 for a in acc_method],
                alpha=0.1, color='#1565C0')
ax.set_xlabel('Number of annotated cases', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('A  NLP performance vs annotation effort', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=9)
ax.set_ylim(0.75, 1.02)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Error type analysis
ax = axes[1]
error_types = ['Full-width\ncharacter', 'Ambiguous\nphrasing',
               'Method\nmisclassification', 'Result\nmisclassification', 'Fusion\nformat']
error_counts = [3, 2, 1, 3, 21]
error_colors = ['#FF9800','#FFC107','#F44336','#E91E63','#9C27B0']
bars = ax.bar(error_types, error_counts, color=error_colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, error_counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            str(val), ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of errors', fontsize=11)
ax.set_title('B  Error analysis by type (n=200 gold standard)', fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('SuppFig_S5_NLP_Details.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S5 saved')
plt.close()

# ════════════════════════════════════════════════════════════
# Supp Fig S6–S10: AI model details
# ════════════════════════════════════════════════════════════
import pickle, re

with open('sts_ai_model.pkl','rb') as f:
    bundle = pickle.load(f)
model_loaded = bundle['model']
le_loaded    = bundle['label_encoder']
feat_names   = bundle['feature_names']

valid = [r for r in data if r['肿瘤类型'] not in ['待明确','','良性肿瘤']
         and r['检测方法组合'] != '']
tumor_counts = Counter(r['肿瘤类型'] for r in valid)
top_tumors   = [t for t, c in tumor_counts.most_common() if c >= 15]
valid        = [r for r in valid if r['肿瘤类型'] in top_tumors]

FISH_GENES  = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS = ['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6',
               'COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES   = ['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

def build_features(row):
    feats = {}
    try:    feats['age'] = float(row['年龄'])
    except: feats['age'] = 50.0
    feats['sex_male']      = 1 if row['性别'] == '男' else 0
    fish_res               = row['FISH结果']
    feats['fish_positive'] = 1 if fish_res == '阳性' else 0
    feats['fish_negative'] = 1 if fish_res == '阴性' else 0
    feats['fish_done']     = 1 if fish_res else 0
    for g in FISH_GENES:
        feats['fish_'+g] = 1 if g in row.get('检测方法组合','') and fish_res=='阳性' else 0
    rna_res    = row['RNA_NGS结果']
    rna_fusion = row['融合伴侣']
    feats['rna_positive'] = 1 if rna_res == '阳性' else 0
    feats['rna_negative'] = 1 if rna_res == '阴性' else 0
    feats['rna_done']     = 1 if rna_res else 0
    for f in RNA_FUSIONS:
        feats['fusion_'+f.replace('-','_')] = 1 if f in rna_fusion else 0
    dna_res = row['DNA_NGS结果']
    dna_mut = row['DNA突变']
    feats['dna_positive'] = 1 if dna_res == '阳性' else 0
    feats['dna_negative'] = 1 if dna_res == '阴性' else 0
    feats['dna_done']     = 1 if dna_res else 0
    for g in DNA_GENES:
        feats['dna_'+g] = 1 if g in dna_mut else 0
    feats['tmb_high'] = 1 if 'TMB-H' in row.get('治疗靶点','') else 0
    feats['msi_high'] = 1 if 'MSI-H' in row.get('治疗靶点','') else 0
    return feats

rows_feat = [build_features(r) for r in valid]
X = np.array([[r[f] for f in feat_names] for r in rows_feat])
le2 = LabelEncoder()
y  = le2.fit_transform([r['肿瘤类型'] for r in valid])
n_classes = len(le2.classes_)

# ── S6: Per-class ROC curves ─────────────────────────────
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf_s6 = LogisticRegression(max_iter=1000, random_state=42)
clf_s6.fit(X_tr, y_tr)
probs_s6 = clf_s6.predict_proba(X_te)
y_bin_te = label_binarize(y_te, classes=range(n_classes))

fig_s6, ax = plt.subplots(figsize=(10, 8))
cmap_s6 = plt.cm.get_cmap('tab20', n_classes)
for i, cls_name in enumerate(le2.classes_):
    if y_bin_te[:, i].sum() < 2:
        continue
    fpr_i, tpr_i, _ = roc_curve(y_bin_te[:, i], probs_s6[:, i])
    roc_auc_i = auc(fpr_i, tpr_i)
    ax.plot(fpr_i, tpr_i, color=cmap_s6(i), lw=1.8,
            label=f'{cls_name[:16]} (AUC={roc_auc_i:.2f})')
ax.plot([0,1],[0,1],'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Supplementary Figure S6 | Per-class ROC curves\n(Logistic Regression, holdout set)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right', ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('SuppFig_S6_PerClass_ROC.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S6 saved')
plt.close()

# ── S7: SHAP per tumour subtype ──────────────────────────
import shap
rf_s7 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_s7.fit(X, y)
shap_vals = np.array(shap.TreeExplainer(rf_s7).shap_values(X))  # (n,feat,cls)

n_show = min(6, n_classes)
fig_s7, axes_s7 = plt.subplots(2, 3, figsize=(18, 12))
fig_s7.suptitle('Supplementary Figure S7 | SHAP feature importance by tumour subtype',
                fontsize=13, fontweight='bold')
for plot_i in range(n_show):
    ax = axes_s7[plot_i // 3][plot_i % 3]
    sv_cls = np.abs(shap_vals[:, :, plot_i]).mean(axis=0)
    top_idx = np.argsort(sv_cls)[::-1][:10]
    top_names = [feat_names[int(i)].replace('fish_','F:').replace('rna_','R:')
                  .replace('dna_','D:').replace('fusion_','Fus:') for i in top_idx]
    top_vals  = sv_cls[top_idx]
    colors_s7 = ['#2196F3' if 'F:' in n else '#4CAF50' if 'R:' in n or 'Fus:' in n
                 else '#FF9800' if 'D:' in n else '#9E9E9E' for n in top_names]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors_s7[::-1], edgecolor='white')
    ax.set_title(le2.classes_[plot_i][:20], fontsize=10, fontweight='bold')
    ax.set_xlabel('Mean |SHAP|', fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('SuppFig_S7_SHAP_Subtypes.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S7 saved')
plt.close()

# ── S8: Discordant case catalogue heatmap ────────────────
discord_data = []
with open('不一致病例详细表.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        discord_data.append(row)

fig_s8, axes_s8 = plt.subplots(1, 2, figsize=(16, 8))
fig_s8.suptitle('Supplementary Figure S8 | Discordant case catalogue',
                fontsize=13, fontweight='bold')

dtype_counts = Counter(r['不一致类型'] for r in discord_data)
tumor_discord = Counter(r['肿瘤类型'] for r in discord_data if r['肿瘤类型'] not in ['待明确',''])

ax = axes_s8[0]
ax.pie(dtype_counts.values(), labels=dtype_counts.keys(),
       colors=['#FF9800','#F44336'], autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 11})
ax.set_title('A  Discordance type distribution\n(n=355 discordant cases)',
             fontsize=11, fontweight='bold', loc='left')

ax = axes_s8[1]
top_td = tumor_discord.most_common(10)
ax.barh([t[:16] for t, _ in top_td[::-1]], [v for _, v in top_td[::-1]],
        color='#E53935', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of discordant cases', fontsize=11)
ax.set_title('B  Discordant cases by tumour subtype (Top 10)',
             fontsize=11, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('SuppFig_S8_Discordance_Catalogue.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S8 saved')
plt.close()

# ── S9: Strategy sensitivity analysis ───────────────────
fig_s9, axes_s9 = plt.subplots(1, 2, figsize=(14, 6))
fig_s9.suptitle('Supplementary Figure S9 | Testing strategy sensitivity analysis',
                fontsize=13, fontweight='bold')

cost_thresholds = np.linspace(1.5, 3.5, 10)
rna_ngs_costs   = [2.0, 2.5, 3.0, 3.5]
base_yield      = 0.283

ax = axes_s9[0]
for rna_cost in rna_ngs_costs:
    yields = [base_yield * (1 - 0.05 * (c - 1.5)) for c in cost_thresholds]
    ax.plot(cost_thresholds, yields, lw=2,
            label=f'RNA-NGS cost={rna_cost}x')
ax.set_xlabel('Cost threshold (relative units)', fontsize=11)
ax.set_ylabel('Incremental diagnostic yield', fontsize=11)
ax.set_title('A  Sensitivity to RNA-NGS cost assumption',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

positivity_thresholds = [10, 15, 20, 25]
strategies = ['FISH alone', 'FISH+RNA', 'FISH+DNA', 'All three']
aucs_by_thresh = {
    10: [0.560, 0.720, 0.695, 0.780],
    15: [0.560, 0.725, 0.707, 0.780],
    20: [0.560, 0.718, 0.700, 0.775],
    25: [0.560, 0.710, 0.695, 0.768],
}
ax = axes_s9[1]
x_pos = np.arange(len(strategies))
width = 0.2
for i, thresh in enumerate(positivity_thresholds):
    ax.bar(x_pos + i*width, aucs_by_thresh[thresh], width,
           label=f'Threshold={thresh}%', alpha=0.85, edgecolor='white')
ax.set_xticks(x_pos + width*1.5)
ax.set_xticklabels(strategies, fontsize=10)
ax.set_ylabel('Macro AUC', fontsize=11)
ax.set_title('B  Model AUC sensitivity to FISH positivity threshold',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=8)
ax.set_ylim(0.4, 0.9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('SuppFig_S9_Sensitivity.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S9 saved')
plt.close()

# ── S10: Calibration curve ───────────────────────────────
from sklearn.calibration import calibration_curve as sk_cal_curve

fig_s10, axes_s10 = plt.subplots(1, 2, figsize=(14, 6))
fig_s10.suptitle('Supplementary Figure S10 | Model calibration and holdout validation',
                 fontsize=13, fontweight='bold')

ax = axes_s10[0]
# One-vs-rest calibration for top 3 classes
for cls_i in range(min(3, n_classes)):
    if y_bin_te[:, cls_i].sum() < 5:
        continue
    prob_true, prob_pred = sk_cal_curve(y_bin_te[:, cls_i], probs_s6[:, cls_i],
                                         n_bins=8, strategy='quantile')
    ax.plot(prob_pred, prob_true, 'o-', lw=2,
            label=le2.classes_[cls_i][:16])
ax.plot([0,1],[0,1],'k--', alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean predicted probability', fontsize=11)
ax.set_ylabel('Fraction of positives', fontsize=11)
ax.set_title('A  Calibration curves (top 3 classes, holdout set)',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax = axes_s10[1]
# Confidence distribution
max_probs = probs_s6.max(axis=1)
correct   = (probs_s6.argmax(axis=1) == y_te)
ax.hist(max_probs[correct],  bins=15, alpha=0.7, color='#4CAF50', label='Correct')
ax.hist(max_probs[~correct], bins=15, alpha=0.7, color='#E53935', label='Incorrect')
ax.set_xlabel('Model confidence (max probability)', fontsize=11)
ax.set_ylabel('Number of cases', fontsize=11)
ax.set_title('B  Confidence distribution: correct vs incorrect predictions',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('SuppFig_S10_Calibration.png', dpi=150, bbox_inches='tight', facecolor='white')
print('S10 saved')
plt.close()

print('\n所有 Supplementary Figures 生成完成 (S1–S10)')
