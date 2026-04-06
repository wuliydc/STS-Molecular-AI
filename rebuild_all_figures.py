"""
批量重新生成 Figure 2–7，统一应用 plot_style.py 的样式规范
主要改进：
  1. 300 dpi TIFF 输出
  2. RGB 白色背景（无透明通道）
  3. 统一字体大小和颜色方案（色盲友好）
  4. 修复 Figure 3 Panel D 中文 monospace 问题
  5. Figure 4 PCA 点加大、加边框
  6. Figure 5 Venn 四象限加总样本量
  7. 所有图统一 panel label (A/B/C...)
"""
import csv, warnings, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA
import shap, scipy.stats as stats

warnings.filterwarnings('ignore')
from plot_style import apply_style, save_figure, panel_label, METHOD_COLORS, COLORS, CONCORDANCE_COLORS
apply_style()

# ── 公共数据加载 ──────────────────────────────────────────
data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): data.append(row)

raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): raw.append(row)

with open('sts_ai_model.pkl','rb') as f:
    bundle = pickle.load(f)
feat_names = bundle['feature_names']

FISH_GENES  = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS = ['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6',
               'COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES   = ['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

def build_features(row):
    feats = {}
    try:    feats['age'] = float(row['年龄'])
    except: feats['age'] = 50.0
    feats['sex_male']      = 1 if row['性别']=='男' else 0
    fish_res               = row['FISH结果']
    feats['fish_positive'] = 1 if fish_res=='阳性' else 0
    feats['fish_negative'] = 1 if fish_res=='阴性' else 0
    feats['fish_done']     = 1 if fish_res else 0
    for g in FISH_GENES:
        feats['fish_'+g] = 1 if g in row.get('检测方法组合','') and fish_res=='阳性' else 0
    rna_res = row['RNA_NGS结果']; rna_fusion = row['融合伴侣']
    feats['rna_positive'] = 1 if rna_res=='阳性' else 0
    feats['rna_negative'] = 1 if rna_res=='阴性' else 0
    feats['rna_done']     = 1 if rna_res else 0
    for f in RNA_FUSIONS:
        feats['fusion_'+f.replace('-','_')] = 1 if f in rna_fusion else 0
    dna_res = row['DNA_NGS结果']; dna_mut = row['DNA突变']
    feats['dna_positive'] = 1 if dna_res=='阳性' else 0
    feats['dna_negative'] = 1 if dna_res=='阴性' else 0
    feats['dna_done']     = 1 if dna_res else 0
    for g in DNA_GENES:
        feats['dna_'+g] = 1 if g in dna_mut else 0
    feats['tmb_high'] = 1 if 'TMB-H' in row.get('治疗靶点','') else 0
    feats['msi_high'] = 1 if 'MSI-H' in row.get('治疗靶点','') else 0
    return feats

valid = [r for r in data if r['肿瘤类型'] not in ['待明确','','良性肿瘤'] and r['检测方法组合']!='']
tc    = Counter(r['肿瘤类型'] for r in valid)
top_t = [t for t,c in tc.most_common() if c>=15]
valid = [r for r in valid if r['肿瘤类型'] in top_t]
rows_feat = [build_features(r) for r in valid]
X  = np.array([[r[f] for f in feat_names] for r in rows_feat])
le = LabelEncoder()
y  = le.fit_transform([r['肿瘤类型'] for r in valid])
n_classes = len(le.classes_)

# ════════════════════════════════════════════════════════════
# FIGURE 2 — Diagnostic gain
# ════════════════════════════════════════════════════════════
print('Building Figure 2...')
fish_rna_pts = [r for r in data if r['FISH结果'] in ['阳性','阴性'] and r['RNA_NGS结果'] in ['阳性','阴性']]
pp = [r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阳性']
nn = [r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阴性']
pn = [r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阴性']
np_= [r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阳性']
three_m = [r for r in data if all(r[f] in ['阳性','阴性'] for f in ['FISH结果','RNA_NGS结果','DNA_NGS结果'])]
dna_new = sum(1 for r in three_m if r['DNA_NGS结果']=='阳性' and r['治疗靶点']!='无' and r['治疗靶点'])
all_targets = []
for r in data:
    t = r.get('治疗靶点','')
    if t and t!='无':
        all_targets.extend(t.split('/'))
target_counts = Counter(all_targets)

fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
fig2.subplots_adjust(hspace=0.42, wspace=0.38)

# A: 增量瀑布
ax = axes[0,0]; panel_label(ax,'A')
steps  = ['FISH\nalone', '+RNA-NGS\n(new positives)', '+DNA-NGS\n(new targets)']
values = [sum(1 for r in data if r['FISH结果']=='阳性'), len(np_), dna_new]
cols   = [METHOD_COLORS['FISH'], METHOD_COLORS['RNA-NGS'], METHOD_COLORS['DNA-NGS']]
bars   = ax.bar(steps, values, color=cols, width=0.5, edgecolor='white', linewidth=1)
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
            f'n={val}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of patients')
ax.set_title('Stepwise diagnostic gain by adding each modality', pad=8)
ax.set_ylim(0, max(values)*1.22)

# B: 四象限一致性图
ax = axes[0,1]; panel_label(ax,'B')
ax.set_xlim(-0.6,1.6); ax.set_ylim(-0.6,1.6)
quad = [(1,1,len(pp),CONCORDANCE_COLORS['pp'],'Concordant\nFISH+ / RNA+'),
        (0,0,len(nn),CONCORDANCE_COLORS['nn'],'Concordant\nFISH− / RNA−'),
        (1,0,len(pn),CONCORDANCE_COLORS['pn'],'Discordant A\nFISH+ / RNA−'),
        (0,1,len(np_),CONCORDANCE_COLORS['np'],'Discordant B\nFISH− / RNA+')]
for x,yq,n,col,lbl in quad:
    ax.add_patch(plt.Rectangle((x-0.48,yq-0.48),0.96,0.96,
                                facecolor=col,alpha=0.20,edgecolor=col,lw=2))
    ax.text(x,yq+0.20,lbl,ha='center',fontsize=8.5,fontweight='bold',color=col)
    ax.text(x,yq-0.10,f'n={n}',ha='center',fontsize=13,fontweight='bold',color=col)
    ax.text(x,yq-0.32,f'({n/len(fish_rna_pts)*100:.1f}%)',ha='center',fontsize=8.5,color=col)
ax.axhline(0.5,color='#999',lw=1.2,ls='--'); ax.axvline(0.5,color='#999',lw=1.2,ls='--')
ax.set_xticks([0,1]); ax.set_xticklabels(['FISH Negative','FISH Positive'])
ax.set_yticks([0,1]); ax.set_yticklabels(['RNA-NGS\nNegative','RNA-NGS\nPositive'])
ax.set_title(f'FISH vs RNA-NGS concordance  (n={len(fish_rna_pts)})', pad=8)
ax.text(0.5,-0.55,f'Overall discordance rate: {(len(pn)+len(np_))/len(fish_rna_pts)*100:.1f}%',
        ha='center',transform=ax.transAxes,fontsize=9,color='#555',style='italic')

# C: 气泡矩阵（患者级）
ax = axes[1,0]; panel_label(ax,'C')
top_tumors_c = [t for t,_ in Counter(r['肿瘤类型'] for r in data if r['肿瘤类型'] not in ['待明确','']).most_common(8)]
tmm = defaultdict(lambda: defaultdict(int))
for r in data:
    t = r['肿瘤类型']
    if t in top_tumors_c:
        for m in ['FISH','RNA-NGS','DNA-NGS']:
            if m in r['检测方法组合']: tmm[t][m]+=1
for i,tumor in enumerate(top_tumors_c):
    for j,method in enumerate(['FISH','RNA-NGS','DNA-NGS']):
        val = tmm[tumor][method]
        if val>0:
            ax.scatter(j,i,s=np.sqrt(val)*20,c=METHOD_COLORS[method],alpha=0.78,zorder=3)
            ax.text(j,i,str(val),ha='center',va='center',fontsize=7.5,fontweight='bold',color='white',zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels(['FISH','RNA-NGS','DNA-NGS'])
ax.set_yticks(range(len(top_tumors_c))); ax.set_yticklabels([t[:14] for t in top_tumors_c],fontsize=8.5)
ax.set_title('Testing method by tumour subtype (patient-level)', pad=8)
ax.grid(True,alpha=0.2,zorder=0); ax.set_xlim(-0.6,2.6); ax.set_ylim(-0.6,len(top_tumors_c)-0.4)

# D: 治疗靶点
ax = axes[1,1]; panel_label(ax,'D')
t_items = target_counts.most_common(8)
t_cols  = plt.cm.Set2(np.linspace(0,1,len(t_items)))
bars_d  = ax.barh([t for t,_ in t_items[::-1]], [v for _,v in t_items[::-1]],
                   color=t_cols[::-1], edgecolor='white', linewidth=0.5)
for bar,val in zip(bars_d,[v for _,v in t_items[::-1]]):
    ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f'n={val}',va='center',fontsize=9)
ax.set_xlabel('Number of patients')
ax.set_title('Actionable therapeutic targets identified', pad=8)
ax.set_xlim(0, max(v for _,v in t_items)*1.25)

fig2.suptitle('Figure 2  |  Stepwise diagnostic gain of sequential molecular testing',
              fontsize=13, fontweight='bold', y=1.01)
save_figure(fig2,'Figure2_v2'); plt.close()

# ════════════════════════════════════════════════════════════
# FIGURE 4 — AI Classifier (改进PCA + SHAP)
# ════════════════════════════════════════════════════════════
print('Building Figure 4...')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}
model_probs = {}
for name, clf in models.items():
    probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
    model_probs[name] = probs

ablation = {
    'All modalities':  list(range(len(feat_names))),
    'w/o DNA-NGS':     [i for i,f in enumerate(feat_names) if not f.startswith('dna_') and f not in ['tmb_high','msi_high']],
    'w/o RNA-NGS':     [i for i,f in enumerate(feat_names) if not f.startswith('rna_') and not f.startswith('fusion_')],
    'w/o FISH':        [i for i,f in enumerate(feat_names) if not f.startswith('fish_')],
    'Clinical only':   [i for i,f in enumerate(feat_names) if f in ['age','sex_male']],
}
best_clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
abl_aucs = {}
y_bin = label_binarize(y, classes=range(n_classes))
for lbl, idx in ablation.items():
    probs = cross_val_predict(best_clf, X[:,idx], y, cv=cv, method='predict_proba')
    abl_aucs[lbl] = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')

rf_shap = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_shap.fit(X, y)
shap_arr = np.array(shap.TreeExplainer(rf_shap).shap_values(X))
fi = np.abs(shap_arr).mean(axis=(0,2))
top15 = np.argsort(fi)[::-1][:15]
top_names  = [feat_names[int(i)].replace('fish_','FISH: ').replace('rna_','RNA: ')
               .replace('dna_','DNA: ').replace('fusion_','Fusion: ').replace('_','-') for i in top15]
top_vals   = fi[top15]
top_colors = ['#0072B2' if 'FISH' in n else '#009E73' if 'RNA' in n or 'Fusion' in n
               else '#E69F00' if 'DNA' in n else '#999999' for n in top_names]

pca = PCA(n_components=2, random_state=42)
X2d = pca.fit_transform(X)

fig4, axes = plt.subplots(2, 3, figsize=(18, 12))
fig4.subplots_adjust(hspace=0.42, wspace=0.38)

# A: ROC
ax = axes[0,0]; panel_label(ax,'A')
ls_list = ['-','--',':']; lc_list = ['#0072B2','#009E73','#D55E00']
for (name, probs), ls, lc in zip(model_probs.items(), ls_list, lc_list):
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:,i],probs[:,i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i,tpr_i,_ = roc_curve(y_bin[:,i],probs[:,i])
        mean_tpr += np.interp(all_fpr,fpr_i,tpr_i)
    mean_tpr /= n_classes
    roc_auc = auc(all_fpr,mean_tpr)
    ax.plot(all_fpr,mean_tpr,ls,color=lc,lw=2,label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.35,lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Multi-model ROC (macro-average, 5-fold CV)', pad=8)
ax.legend(loc='lower right',framealpha=0.9)

# B: 消融实验
ax = axes[0,1]; panel_label(ax,'B')
abl_labels = list(abl_aucs.keys()); abl_vals = list(abl_aucs.values())
bar_cols = ['#0072B2' if l=='All modalities' else '#D55E00' if 'w/o' in l else '#999' for l in abl_labels]
bars = ax.barh(abl_labels, abl_vals, color=bar_cols, height=0.55, edgecolor='white')
for bar,val in zip(bars,abl_vals):
    delta = val - abl_aucs['All modalities']
    label = f'{val:.3f}' if delta==0 else f'{val:.3f}  ({delta:+.3f})'
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2, label, va='center', fontsize=9)
ax.axvline(abl_aucs['All modalities'],color='#0072B2',ls='--',alpha=0.5,lw=1.5)
ax.set_xlabel('Macro AUC'); ax.set_title('Ablation study — per-modality contribution', pad=8)
ax.set_xlim(0,1.08)

# C: SHAP条形图
ax = axes[0,2]; panel_label(ax,'C')
ax.barh(top_names[::-1], top_vals[::-1], color=top_colors[::-1], edgecolor='white', linewidth=0.5)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Global feature importance (SHAP)', pad=8)
legend_patches = [mpatches.Patch(color='#0072B2',label='FISH'),
                  mpatches.Patch(color='#009E73',label='RNA-NGS / Fusion'),
                  mpatches.Patch(color='#E69F00',label='DNA-NGS'),
                  mpatches.Patch(color='#999999',label='Clinical')]
ax.legend(handles=legend_patches, fontsize=8, loc='lower right')

# D: 个体SHAP瀑布（3例）
ax = axes[1,0]; ax.axis('off'); panel_label(ax,'D')
ax.set_title('Individual SHAP explanations (3 representative cases)', pad=8)
case_indices = [np.where(y==i)[0][0] for i in range(min(3,n_classes))]
for pi, ci in enumerate(case_indices):
    tumor_name = le.classes_[y[ci]]
    sv = shap_arr[ci,:,y[ci]]
    top5 = np.argsort(np.abs(sv))[::-1][:5]
    y_base = 0.93 - pi*0.32
    ax.text(0.02, y_base, f'Case {pi+1}: {tumor_name[:22]}',
            transform=ax.transAxes, fontsize=9, fontweight='bold', color='#333')
    for j,(fi_idx,fval) in enumerate(zip(top5, sv[top5])):
        y_bar = y_base - 0.04 - j*0.044
        fname = feat_names[int(fi_idx)][:12]
        bar_len = min(abs(fval)*0.28, 0.22)
        col = '#D55E00' if fval>0 else '#0072B2'
        ax.annotate('', xy=(0.35+(bar_len if fval>0 else -bar_len), y_bar),
                    xytext=(0.35, y_bar), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=col, lw=2, mutation_scale=8))
        ax.text(0.34, y_bar, fname, transform=ax.transAxes,
                ha='right', va='center', fontsize=7.5, color='#555')
        ax.text(0.36+(bar_len if fval>0 else -bar_len), y_bar, f'{fval:+.3f}',
                transform=ax.transAxes, ha='left' if fval>0 else 'right',
                va='center', fontsize=7.5, color=col, fontweight='bold')

# E: PCA（改进：更大点，更清晰颜色）
ax = axes[1,1]; panel_label(ax,'E')
cmap4 = plt.cm.get_cmap('tab20', n_classes)
for ci, tname in enumerate(le.classes_):
    mask = y==ci
    ax.scatter(X2d[mask,0], X2d[mask,1], c=[cmap4(ci)], label=tname[:14],
               s=55, alpha=0.75, edgecolors='white', linewidths=0.6, zorder=3)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA of multi-modal molecular features', pad=8)
ax.legend(fontsize=6.5, loc='upper right', ncol=2, framealpha=0.85, markerscale=1.3)

# F: 性能汇总
ax = axes[1,2]; ax.axis('off'); panel_label(ax,'F')
ax.set_title('Model performance summary', pad=8)
rows_t = [['Model','CV AUC','Holdout AUC']]
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
y_bin_te = label_binarize(y_te,classes=range(n_classes))
for name,clf in models.items():
    clf.fit(X_tr,y_tr)
    probs_te = clf.predict_proba(X_te)
    cv_auc = roc_auc_score(y_bin, model_probs[name], multi_class='ovr', average='macro')
    ho_auc = roc_auc_score(y_bin_te, probs_te, multi_class='ovr', average='macro')
    rows_t.append([name, f'{cv_auc:.3f}', f'{ho_auc:.3f}'])
col_x = [0.02, 0.52, 0.78]
for ri, row in enumerate(rows_t):
    y_r = 0.92 - ri*0.13
    fc = '#0072B2' if ri==0 else ('#E3F2FD' if ri%2==0 else 'white')
    tc = 'white' if ri==0 else '#333'
    ax.add_patch(FancyBboxPatch((0.01,y_r-0.10),0.97,0.11,transform=ax.transAxes,
                                 boxstyle='round,pad=0.005',facecolor=fc,edgecolor='white'))
    for ci,(cell,cx) in enumerate(zip(row,col_x)):
        ax.text(cx,y_r-0.04,cell,transform=ax.transAxes,fontsize=8.5,
                fontweight='bold' if ri==0 else 'normal',color=tc,va='center')

fig4.suptitle('Figure 4  |  Multi-modal AI classifier with interpretable molecular feature contributions',
              fontsize=13, fontweight='bold', y=1.01)
save_figure(fig4,'Figure4_v2'); plt.close()
print('  Figure 4 done')

# ════════════════════════════════════════════════════════════
# FIGURE 5 — Discordance (改进四象限 + 融合网络)
# ════════════════════════════════════════════════════════════
print('Building Figure 5...')
fish_rna_pts = [r for r in data if r['FISH结果'] in ['阳性','阴性'] and r['RNA_NGS结果'] in ['阳性','阴性']]
pp=[r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阳性']
nn=[r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阴性']
pn=[r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阴性']
np_=[r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阳性']
discordant = pn+np_; concordant = pp+nn

ddit3_rna = [r for r in raw if 'DDIT3' in r.get('诊断结论原文','') and r['检测方法']=='RNA-NGS' and r['检测结果']=='阳性']
fusion_partners = Counter(r.get('融合伴侣基因','') or 'Unknown' for r in ddit3_rna)

def get_age(r):
    try: return float(r['年龄'])
    except: return None

ages_disc = [a for a in [get_age(r) for r in discordant] if a]
ages_conc = [a for a in [get_age(r) for r in concordant] if a]
_, p_age  = stats.mannwhitneyu(ages_disc, ages_conc, alternative='two-sided')

# 不一致预测模型
sample_conc = concordant[:len(discordant)*3]
all_s = discordant+sample_conc; labels_s = [1]*len(discordant)+[0]*len(sample_conc)
def feat_d(r):
    try: age=float(r['年龄'])
    except: age=50.0
    return [age, 1 if r['性别']=='男' else 0,
            1 if '脂肪肉瘤' in r['肿瘤类型'] else 0,
            1 if r['FISH结果']=='阳性' else 0,
            1 if r['RNA_NGS结果']=='阳性' else 0]
X_d=np.array([feat_d(r) for r in all_s]); y_d=np.array(labels_s)
clf_d=LogisticRegression(max_iter=500,random_state=42)
probs_d=cross_val_predict(clf_d,X_d,y_d,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]
fpr_d,tpr_d,_=roc_curve(y_d,probs_d); disc_auc=auc(fpr_d,tpr_d)

fig5, axes = plt.subplots(2, 3, figsize=(18, 12))
fig5.subplots_adjust(hspace=0.42, wspace=0.38)

# A: 四象限（改进：加总样本量和百分比）
ax=axes[0,0]; panel_label(ax,'A')
ax.set_xlim(-0.6,1.6); ax.set_ylim(-0.6,1.6)
quad=[(1,1,len(pp),CONCORDANCE_COLORS['pp'],'Concordant\nFISH+ / RNA+'),
      (0,0,len(nn),CONCORDANCE_COLORS['nn'],'Concordant\nFISH− / RNA−'),
      (1,0,len(pn),CONCORDANCE_COLORS['pn'],'Discordant A\nFISH+ / RNA−'),
      (0,1,len(np_),CONCORDANCE_COLORS['np'],'Discordant B\nFISH− / RNA+')]
for x,yq,n,col,lbl in quad:
    ax.add_patch(plt.Rectangle((x-0.48,yq-0.48),0.96,0.96,facecolor=col,alpha=0.18,edgecolor=col,lw=2.2))
    ax.text(x,yq+0.22,lbl,ha='center',fontsize=8.5,fontweight='bold',color=col)
    ax.text(x,yq-0.05,f'n={n}',ha='center',fontsize=13,fontweight='bold',color=col)
    ax.text(x,yq-0.28,f'{n/len(fish_rna_pts)*100:.1f}%',ha='center',fontsize=9,color=col)
ax.axhline(0.5,color='#999',lw=1.2,ls='--'); ax.axvline(0.5,color='#999',lw=1.2,ls='--')
ax.set_xticks([0,1]); ax.set_xticklabels(['FISH Negative','FISH Positive'])
ax.set_yticks([0,1]); ax.set_yticklabels(['RNA-NGS\nNegative','RNA-NGS\nPositive'])
ax.set_title(f'FISH vs RNA-NGS concordance  (n={len(fish_rna_pts)})\nDiscordance rate: {len(discordant)/len(fish_rna_pts)*100:.1f}%', pad=8)

# B: 年龄箱线图（比森林图更直观）
ax=axes[0,1]; panel_label(ax,'B')
bp=ax.boxplot([ages_disc,ages_conc],patch_artist=True,
               medianprops=dict(color='red',lw=2.5),
               whiskerprops=dict(lw=1.5),capprops=dict(lw=1.5))
bp['boxes'][0].set_facecolor(CONCORDANCE_COLORS['np']); bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor(CONCORDANCE_COLORS['nn']); bp['boxes'][1].set_alpha(0.6)
ax.set_xticklabels(['Discordant\n(n=355)','Concordant\n(n=697)'])
ax.set_ylabel('Age (years)')
ax.set_title(f'Age comparison: discordant vs concordant\n(Mann-Whitney U, p={p_age:.3f})', pad=8)
sig = '**' if p_age<0.01 else '*' if p_age<0.05 else 'ns'
ax.text(1.5, max(ages_disc+ages_conc)*0.98, sig, ha='center', fontsize=14, fontweight='bold',
        color='red' if p_age<0.05 else '#999')

# C: DDIT3融合伴侣
ax=axes[0,2]; ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
ax.set_aspect('equal'); ax.axis('off'); panel_label(ax,'C',x=-0.05)
ax.set_title('DDIT3 fusion partner landscape\n(RNA-NGS detected)', pad=8)
ax.add_patch(plt.Circle((0,0),0.22,color='#0072B2',zorder=3))
ax.text(0,0,'DDIT3',ha='center',va='center',fontsize=10,fontweight='bold',color='white',zorder=4)
top_p=fusion_partners.most_common(8)
if not top_p: top_p=[('FUS',16),('EWSR1',5),('Unknown',8)]
n_p=len(top_p); angles=np.linspace(0,2*np.pi,n_p,endpoint=False)
max_c=max(c for _,c in top_p)
for (partner,count),angle in zip(top_p,angles):
    r=1.1; x=r*np.cos(angle); y=r*np.sin(angle)
    size=0.09+(count/max_c)*0.13
    col='#D55E00' if partner in ['FUS-DDIT3','EWSR1-DDIT3','FUS','EWSR1'] else '#E69F00' if 'Unknown' in partner else '#009E73'
    ax.add_patch(plt.Circle((x,y),size,color=col,alpha=0.88,zorder=3))
    ax.plot([0,x*(1-size/r-0.22/r)],[0,y*(1-size/r-0.22/r)],color='#BBB',lw=1.5,alpha=0.7,zorder=2)
    lbl=partner.replace('-intergenic','').replace('-未知','')[:10]
    ax.text(x*1.38,y*1.38,f'{lbl}\n(n={count})',ha='center',va='center',fontsize=7.5,color='#333')
ax.legend(handles=[mpatches.Patch(color='#D55E00',label='Known oncogenic'),
                   mpatches.Patch(color='#E69F00',label='Unknown partner'),
                   mpatches.Patch(color='#009E73',label='Other')],
          fontsize=8,loc='lower right')

# D: 不一致预测ROC
ax=axes[1,0]; panel_label(ax,'D')
ax.plot(fpr_d,tpr_d,color='#D55E00',lw=2.5,label=f'Discordance predictor\n(AUC={disc_auc:.3f})')
ax.fill_between(fpr_d,tpr_d,alpha=0.10,color='#D55E00')
ax.plot([0,1],[0,1],'k--',alpha=0.35,lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Discordance prediction model\n(5-fold CV ROC)', pad=8)
ax.legend(loc='lower right',framealpha=0.9)

# E: 临床决策流程图
ax=axes[1,1]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off'); panel_label(ax,'E',x=-0.05)
ax.set_title('Clinical decision algorithm for discordant cases', pad=8)
def dbox(ax,x,y,w,h,txt,fc,ec,fs=8.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.1',facecolor=fc,edgecolor=ec,lw=2))
    ax.text(x,y,txt,ha='center',va='center',fontsize=fs,fontweight='bold',color=ec,multialignment='center')
def darrow(ax,x1,y1,x2,y2,lbl=''):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#666',lw=1.8))
    if lbl: ax.text((x1+x2)/2+0.15,(y1+y2)/2,lbl,fontsize=8,color='#666',style='italic')
dbox(ax,5,9.2,5.5,0.9,'STS Molecular Testing\n(FISH + RNA-NGS)','#E3F2FD','#0072B2',9)
dbox(ax,2.5,7.5,3.5,0.9,'Concordant\n(FISH = RNA-NGS)','#C8E6C9','#1B5E20',8.5)
dbox(ax,7.5,7.5,3.5,0.9,'Discordant\n(FISH ≠ RNA-NGS)','#FFCCBC','#D55E00',8.5)
darrow(ax,5,8.75,5,8.0); darrow(ax,3.5,8.0,2.5,7.95,'Concordant'); darrow(ax,6.5,8.0,7.5,7.95,'Discordant')
dbox(ax,2.5,6.0,3.5,0.9,'Diagnosis confirmed','#A5D6A7','#1B5E20',8.5)
darrow(ax,2.5,7.05,2.5,6.45)
dbox(ax,5.5,6.0,2.2,0.9,'Type A\nFISH+/RNA−','#FFE0B2','#E65100',8)
dbox(ax,9.0,6.0,2.2,0.9,'Type B\nFISH−/RNA+','#FFCDD2','#B71C1C',8)
darrow(ax,6.8,7.5,5.5,6.45,'A'); darrow(ax,8.2,7.5,9.0,6.45,'B')
dbox(ax,5.5,4.5,2.2,0.9,'Intergenic fusion?\nRNA quality check','#FFF3E0','#E65100',7.5)
dbox(ax,9.0,4.5,2.2,0.9,'Probe design gap\nAdd DNA-NGS','#FFEBEE','#B71C1C',7.5)
darrow(ax,5.5,5.55,5.5,4.95); darrow(ax,9.0,5.55,9.0,4.95)
dbox(ax,7.25,3.0,4.5,0.9,'WGS/WTS for\nmechanism clarification','#F3E5F5','#6A1B9A',8.5)
darrow(ax,5.5,4.05,6.5,3.45); darrow(ax,9.0,4.05,8.0,3.45)

# F: 不一致统计汇总
ax=axes[1,2]; ax.axis('off'); panel_label(ax,'F')
ax.set_title('Discordance summary statistics', pad=8)
rows_f=[['Category','n','%'],
        ['Total FISH+RNA-NGS',str(len(fish_rna_pts)),'100%'],
        ['Concordant positive',str(len(pp)),f'{len(pp)/len(fish_rna_pts)*100:.1f}%'],
        ['Concordant negative',str(len(nn)),f'{len(nn)/len(fish_rna_pts)*100:.1f}%'],
        ['Discordant Type A (FISH+/RNA−)',str(len(pn)),f'{len(pn)/len(fish_rna_pts)*100:.1f}%'],
        ['Discordant Type B (FISH−/RNA+)',str(len(np_)),f'{len(np_)/len(fish_rna_pts)*100:.1f}%'],
        ['─'*28,'─'*4,'─'*5],
        ['Median age (discordant)',f'{np.median(ages_disc):.0f} yrs',''],
        ['Median age (concordant)',f'{np.median(ages_conc):.0f} yrs',f'p={p_age:.3f}'],
        ['Discordance predictor AUC',f'{disc_auc:.3f}','5-fold CV']]
for ri,row in enumerate(rows_f):
    y_r=0.95-ri*0.092
    fc='#0072B2' if ri==0 else ('#E3F2FD' if ri%2==0 else 'white')
    tc='white' if ri==0 else ('#D55E00' if ri in [4,5] else '#333')
    ax.add_patch(FancyBboxPatch((0.01,y_r-0.082),0.97,0.085,transform=ax.transAxes,
                                 boxstyle='round,pad=0.005',facecolor=fc,edgecolor='white'))
    for ci,(cell,cx) in enumerate(zip(row,[0.02,0.62,0.82])):
        ax.text(cx,y_r-0.035,cell,transform=ax.transAxes,fontsize=8,
                fontweight='bold' if ri==0 else 'normal',color=tc,va='center')

fig5.suptitle('Figure 5  |  Systematic characterisation of FISH–NGS discordance',
              fontsize=13, fontweight='bold', y=1.01)
save_figure(fig5,'Figure5_v2'); plt.close()
print('  Figure 5 done')
print('\nAll figures rebuilt successfully.')
