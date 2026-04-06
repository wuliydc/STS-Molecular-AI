"""
Figure 4: Multi-modal AI classifier
A: ROC curves (multi-model comparison)
B: Ablation study (per-modality contribution)
C: SHAP beeswarm (global feature importance)
D: SHAP waterfall (3 individual cases)
E: UMAP (patient feature space coloured by tumour subtype)
"""
import csv, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
from plot_style import TUMOR_EN
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 读取患者级数据 ────────────────────────────────────────
data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# ── 特征工程 ──────────────────────────────────────────────
# 只保留有明确肿瘤类型且至少做过一种分子检测的患者
valid = [r for r in data
         if r['肿瘤类型'] not in ['待明确', '', '良性肿瘤']
         and r['检测方法组合'] != '']

# 保留样本量≥15的肿瘤亚型
tumor_counts = Counter(r['肿瘤类型'] for r in valid)
top_tumors = [t for t, c in tumor_counts.most_common() if c >= 15]
valid = [r for r in valid if r['肿瘤类型'] in top_tumors]
print(f'建模样本: {len(valid)}例, 肿瘤亚型: {len(top_tumors)}类')
for t in top_tumors:
    print(f'  {t}: {tumor_counts[t]}')

# 构建特征矩阵
FISH_GENES  = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS = ['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6',
               'COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES   = ['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

def build_features(row):
    feats = {}
    # 临床特征
    try:    feats['age'] = float(row['年龄'])
    except: feats['age'] = 50.0
    feats['sex_male'] = 1 if row['性别'] == '男' else 0

    # FISH特征
    fish_res = row['FISH结果']
    feats['fish_positive'] = 1 if fish_res == '阳性' else 0
    feats['fish_negative'] = 1 if fish_res == '阴性' else 0
    feats['fish_done']     = 1 if fish_res else 0
    for g in FISH_GENES:
        feats[f'fish_{g}'] = 1 if g in row.get('检测方法组合','') and fish_res == '阳性' else 0

    # RNA-NGS特征
    rna_res    = row['RNA_NGS结果']
    rna_fusion = row['融合伴侣']
    feats['rna_positive'] = 1 if rna_res == '阳性' else 0
    feats['rna_negative'] = 1 if rna_res == '阴性' else 0
    feats['rna_done']     = 1 if rna_res else 0
    for f in RNA_FUSIONS:
        feats[f'fusion_{f.replace("-","_")}'] = 1 if f in rna_fusion else 0

    # DNA-NGS特征
    dna_res = row['DNA_NGS结果']
    dna_mut = row['DNA突变']
    feats['dna_positive'] = 1 if dna_res == '阳性' else 0
    feats['dna_negative'] = 1 if dna_res == '阴性' else 0
    feats['dna_done']     = 1 if dna_res else 0
    for g in DNA_GENES:
        feats[f'dna_{g}'] = 1 if g in dna_mut else 0
    feats['tmb_high']  = 1 if 'TMB-H' in row.get('治疗靶点','') else 0
    feats['msi_high']  = 1 if 'MSI-H' in row.get('治疗靶点','') else 0
    return feats

rows_feat = [build_features(r) for r in valid]
feat_names = list(rows_feat[0].keys())
X = np.array([[r[f] for f in feat_names] for r in rows_feat])
y_raw = [r['肿瘤类型'] for r in valid]
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)
print(f'\n特征维度: {X.shape}, 类别数: {n_classes}')

# ── 模型训练与评估 ────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_probs = {}
model_aucs  = {}

print('\n=== 模型性能（5折交叉验证）===')
for name, clf in models.items():
    probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
    y_bin = label_binarize(y, classes=range(n_classes))
    if n_classes == 2:
        macro_auc = roc_auc_score(y_bin, probs[:, 1])
    else:
        macro_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
    model_probs[name] = probs
    model_aucs[name]  = macro_auc
    print(f'  {name}: AUC={macro_auc:.3f}')

# ── 消融实验 ──────────────────────────────────────────────
def get_feat_idx(prefix):
    return [i for i, f in enumerate(feat_names) if f.startswith(prefix) or f == prefix]

modality_masks = {
    'All modalities':    list(range(len(feat_names))),
    'w/o DNA-NGS':       [i for i, f in enumerate(feat_names) if not f.startswith('dna_') and f not in ['tmb_high','msi_high']],
    'w/o RNA-NGS':       [i for i, f in enumerate(feat_names) if not f.startswith('rna_') and not f.startswith('fusion_')],
    'w/o FISH':          [i for i, f in enumerate(feat_names) if not f.startswith('fish_')],
    'Clinical only':     [i for i, f in enumerate(feat_names) if f in ['age','sex_male']],
}

best_clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
ablation_aucs = {}
print('\n=== 消融实验 ===')
for label, idx in modality_masks.items():
    X_sub = X[:, idx]
    probs = cross_val_predict(best_clf, X_sub, y, cv=cv, method='predict_proba')
    y_bin = label_binarize(y, classes=range(n_classes))
    if n_classes == 2:
        a = roc_auc_score(y_bin, probs[:, 1])
    else:
        a = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
    ablation_aucs[label] = a
    print(f'  {label}: AUC={a:.3f}')

# ── SHAP分析（用RandomForest，支持多分类）────────────────
rf_for_shap = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_for_shap.fit(X, y)
explainer = shap.TreeExplainer(rf_for_shap)
shap_vals = explainer.shap_values(X)  # shape: (n_samples, n_features, n_classes)

# shap_vals shape: (826, 44, 15) → mean |SHAP| per feature across all classes
shap_arr        = np.array(shap_vals)  # (826, 44, 15)
feat_importance = np.abs(shap_arr).mean(axis=(0, 2))  # (44,)
sorted_idx      = np.argsort(feat_importance)[::-1][:15]
top_feat_names  = [feat_names[int(i)] for i in sorted_idx]
top_feat_values = feat_importance[sorted_idx]

print('\n=== Top 15 SHAP特征 ===')
for fn, fv in zip(top_feat_names, top_feat_values):
    print(f'  {fn}: {fv:.4f}')

# ── PCA降维（替代UMAP，无需额外安装）────────────────────
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
print(f'\nPCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%')

# ════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))
palette = plt.cm.Set1(np.linspace(0, 1, max(n_classes, 3)))

# ── Panel A: ROC曲线 ─────────────────────────────────────
ax_a = fig.add_subplot(2, 3, 1)
line_styles = ['-', '--', ':']
line_colors = ['#1565C0', '#2E7D32', '#E65100']
y_bin = label_binarize(y, classes=range(n_classes))

for (name, probs), ls, lc in zip(model_probs.items(), line_styles, line_colors):
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_bin, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        ax_a.plot(fpr, tpr, ls, color=lc, lw=2,
                  label=f'{name} (AUC={roc_auc:.3f})')
    else:
        # macro-average ROC
        all_fpr = np.unique(np.concatenate([
            roc_curve(y_bin[:, i], probs[:, i])[0] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        ax_a.plot(all_fpr, mean_tpr, ls, color=lc, lw=2,
                  label=f'{name} (AUC={roc_auc:.3f})')

ax_a.plot([0,1],[0,1],'k--', alpha=0.4, lw=1)
ax_a.set_xlabel('False Positive Rate', fontsize=11)
ax_a.set_ylabel('True Positive Rate', fontsize=11)
ax_a.set_title('A  Multi-model ROC comparison\n(macro-average, 5-fold CV)', fontsize=11, fontweight='bold', loc='left')
ax_a.legend(fontsize=9, loc='lower right')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# ── Panel B: 消融实验 ────────────────────────────────────
ax_b = fig.add_subplot(2, 3, 2)
abl_labels = list(ablation_aucs.keys())
abl_values = list(ablation_aucs.values())
bar_colors = ['#1565C0' if l == 'All modalities' else
              '#FF5722' if 'w/o' in l else '#9E9E9E'
              for l in abl_labels]
bars = ax_b.barh(abl_labels, abl_values, color=bar_colors,
                 height=0.55, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, abl_values):
    ax_b.text(val + 0.002, bar.get_y() + bar.get_height()/2,
              f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
ax_b.set_xlabel('Macro AUC', fontsize=11)
ax_b.set_title('B  Ablation study – per-modality contribution', fontsize=11, fontweight='bold', loc='left')
ax_b.set_xlim(0, 1.05)
ax_b.axvline(ablation_aucs['All modalities'], color='#1565C0',
             linestyle='--', alpha=0.5, linewidth=1.5)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# ── Panel C: SHAP蜂群图（用条形图替代，更清晰）────────────
ax_c = fig.add_subplot(2, 3, 3)
feat_labels_clean = [f.replace('fish_','FISH: ').replace('rna_','RNA: ')
                      .replace('dna_','DNA: ').replace('fusion_','Fusion: ')
                      .replace('_','-') for f in top_feat_names]
colors_c = ['#2196F3' if 'FISH' in f else
            '#4CAF50' if 'RNA' in f or 'Fusion' in f else
            '#FF9800' if 'DNA' in f else '#9E9E9E'
            for f in feat_labels_clean]
y_pos = np.arange(len(top_feat_names))
ax_c.barh(y_pos, top_feat_values[::-1], color=colors_c[::-1],
          height=0.7, edgecolor='white', linewidth=1)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(feat_labels_clean[::-1], fontsize=9)
ax_c.set_xlabel('Mean |SHAP value|', fontsize=11)
ax_c.set_title('C  Global feature importance (SHAP)', fontsize=11, fontweight='bold', loc='left')
legend_patches = [
    mpatches.Patch(color='#2196F3', label='FISH'),
    mpatches.Patch(color='#4CAF50', label='RNA-NGS / Fusion'),
    mpatches.Patch(color='#FF9800', label='DNA-NGS'),
    mpatches.Patch(color='#9E9E9E', label='Clinical'),
]
ax_c.legend(handles=legend_patches, fontsize=8, loc='lower right')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# ── Panel D: SHAP瀑布图（3个典型病例）───────────────────
ax_d = fig.add_subplot(2, 3, 4)
ax_d.axis('off')
ax_d.set_title('D  Individual SHAP explanations (3 representative cases)', fontsize=11, fontweight='bold', loc='left')

# 每个肿瘤类型选1个典型病例
case_indices = []
for cls_idx in range(min(3, n_classes)):
    cls_samples = np.where(y == cls_idx)[0]
    if len(cls_samples) > 0:
        case_indices.append(cls_samples[0])

for plot_i, case_idx in enumerate(case_indices[:3]):
    tumor_name = le.classes_[y[case_idx]]
    # shap_vals shape: (826, 44, 15) → pick class dimension for this case's true class
    sv = shap_arr[case_idx, :, y[case_idx]]  # (44,)

    top5_idx = np.argsort(np.abs(sv))[::-1][:5]
    top5_names  = [feat_names[i].replace('fish_','').replace('rna_','').replace('dna_','').replace('fusion_','') for i in top5_idx]
    top5_shap   = sv[top5_idx]

    y_base = 0.92 - plot_i * 0.32
    ax_d.text(0.02, y_base, f'Case {plot_i+1}: {TUMOR_EN.get(tumor_name, tumor_name)[:28]}',
              transform=ax_d.transAxes, fontsize=9, fontweight='bold', color='#333')

    for j, (fname, fval) in enumerate(zip(top5_names, top5_shap)):
        y_bar = y_base - 0.04 - j * 0.045
        bar_len = min(abs(fval) * 0.3, 0.25)
        color = '#E53935' if fval > 0 else '#1E88E5'
        ax_d.add_patch(mpatches.FancyArrowPatch(
            (0.35, y_bar), (0.35 + (bar_len if fval > 0 else -bar_len), y_bar),
            transform=ax_d.transAxes,
            arrowstyle='->', color=color, lw=2,
            mutation_scale=8))
        ax_d.text(0.34, y_bar, fname[:12], transform=ax_d.transAxes,
                  ha='right', va='center', fontsize=7.5, color='#555')
        ax_d.text(0.36 + (bar_len if fval > 0 else -bar_len),
                  y_bar, f'{fval:+.3f}',
                  transform=ax_d.transAxes, ha='left' if fval > 0 else 'right',
                  va='center', fontsize=7.5, color=color, fontweight='bold')

# ── Panel E: PCA散点图 ───────────────────────────────────
ax_e = fig.add_subplot(2, 3, 5)
cmap = plt.cm.get_cmap('tab10', n_classes)
for cls_idx, tumor_name in enumerate(le.classes_):
    mask = y == cls_idx
    ax_e.scatter(X_2d[mask, 0], X_2d[mask, 1],
                 c=[cmap(cls_idx)], label=TUMOR_EN.get(tumor_name, tumor_name)[:22],
                 s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
ax_e.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax_e.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax_e.set_title('E  PCA of multi-modal molecular features\n(coloured by tumour subtype)', fontsize=11, fontweight='bold', loc='left')
ax_e.legend(fontsize=7, loc='upper right', ncol=2,
            framealpha=0.8, markerscale=1.2)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

# ── Panel F: 性能汇总表 ──────────────────────────────────
ax_f = fig.add_subplot(2, 3, 6)
ax_f.axis('off')
ax_f.set_title('F  Model performance summary', fontsize=11, fontweight='bold', loc='left')

from matplotlib.patches import FancyBboxPatch
rows_table = [['Model', 'Macro AUC', 'Setting']]
for name, a in model_aucs.items():
    rows_table.append([name, f'{a:.3f}', '5-fold CV'])
rows_table.append(['─'*20, '─'*8, '─'*10])
rows_table.append(['Ablation results:', '', ''])
for label, a in ablation_aucs.items():
    delta = a - ablation_aucs['All modalities']
    delta_str = f'{delta:+.3f}' if label != 'All modalities' else '(baseline)'
    rows_table.append([f'  {label}', f'{a:.3f}', delta_str])

col_x = [0.02, 0.52, 0.78]
row_colors_t = ['#1565C0'] + ['#E3F2FD' if i%2==0 else 'white' for i in range(len(rows_table)-1)]
text_colors_t = ['white'] + ['black']*(len(rows_table)-1)

for ri, row in enumerate(rows_table):
    y_r = 0.97 - ri * 0.075
    if ri == 0:
        ax_f.add_patch(FancyBboxPatch((0.01, y_r-0.06), 0.97, 0.065,
                                       transform=ax_f.transAxes,
                                       boxstyle='round,pad=0.005',
                                       facecolor='#1565C0', edgecolor='white'))
    for ci, (cell, cx) in enumerate(zip(row, col_x)):
        fw = 'bold' if ri == 0 else 'normal'
        tc = 'white' if ri == 0 else ('#E53935' if '+' in str(cell) else
                                       '#1E88E5' if '-' in str(cell) and cell != '─'*8 else '#333')
        ax_f.text(cx, y_r - 0.025, cell, transform=ax_f.transAxes,
                  fontsize=8.5, fontweight=fw, color=tc, va='center')

plt.suptitle('Figure 4 | A multi-modal AI classifier integrating FISH, RNA-NGS, and DNA-NGS\nachieves high diagnostic accuracy with interpretable molecular feature contributions',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=3, w_pad=2)
plt.savefig('Figure4_AI_Classifier.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nFigure 4 已保存: Figure4_AI_Classifier.png')
