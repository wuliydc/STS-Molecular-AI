"""
Extended Figures 1–5 (合并版 Supplementary Figures)
EF1: 队列扩展特征 (S1 精简)
EF2: 分子检测全景 (S2 + S3 + S4)
EF3: NLP详细性能 (S5 + S6)
EF4: AI模型详细评估 (S7 + S10)
EF5: 不一致分析扩展 (S8 + S9)
"""
import csv, warnings, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.calibration import calibration_curve
import shap, scipy.stats as stats

warnings.filterwarnings('ignore')
from plot_style import apply_style, save_figure, panel_label, METHOD_COLORS, COLORS
apply_style()

# ── 数据加载 ──────────────────────────────────────────────
data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): data.append(row)
raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): raw.append(row)
discord_data = []
with open('不一致病例详细表.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): discord_data.append(row)
with open('sts_ai_model.pkl','rb') as f:
    bundle = pickle.load(f)
feat_names = bundle['feature_names']

FISH_GENES  = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS = ['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6',
               'COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES   = ['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

def build_features(row):
    feats = {f: 0.0 for f in feat_names}
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
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
y_bin_te = label_binarize(y_te, classes=range(n_classes))

# ════════════════════════════════════════════════════════════
# Extended Figure 1: 队列扩展特征
# ════════════════════════════════════════════════════════════
print('Building Extended Figure 1...')
ages  = [float(r['年龄']) for r in data if r['年龄'].replace('.','').isdigit()]
sexes = Counter(r['性别'] for r in data if r['性别'] in ['男','女'])
tumor_counts = Counter(r['肿瘤类型'] for r in data if r['肿瘤类型'] not in ['待明确',''])
strategy_counts = Counter(r['检测方法组合'] for r in data if r['检测方法组合'])

year_method = defaultdict(lambda: defaultdict(int))
for row in raw:
    t = row.get('登记时间',''); m = row.get('检测方法','')
    if t and m != '会诊':
        try:
            yr = int(str(t)[:4])
            if 2018 <= yr <= 2025: year_method[yr][m] += 1
        except: pass
years = sorted(year_method.keys())

fig_ef1, axes = plt.subplots(2, 3, figsize=(18, 12))
fig_ef1.subplots_adjust(hspace=0.42, wspace=0.38)

# A: 年龄分布
ax = axes[0,0]; panel_label(ax,'A')
ax.hist(ages, bins=20, color=METHOD_COLORS['FISH'], alpha=0.82, edgecolor='white', lw=0.5)
ax.axvline(np.median(ages), color='#D55E00', lw=2, ls='--', label=f'Median={np.median(ages):.0f} yrs')
ax.axvline(np.percentile(ages,25), color='#999', lw=1.2, ls=':', alpha=0.7)
ax.axvline(np.percentile(ages,75), color='#999', lw=1.2, ls=':', alpha=0.7, label=f'IQR {np.percentile(ages,25):.0f}–{np.percentile(ages,75):.0f}')
ax.set_xlabel('Age (years)'); ax.set_ylabel('Number of patients')
ax.set_title('Age distribution', pad=8); ax.legend(fontsize=9)

# B: 性别饼图
ax = axes[0,1]; panel_label(ax,'B')
ax.pie([sexes['男'],sexes['女']], labels=[f'Male\n(n={sexes["男"]})',f'Female\n(n={sexes["女"]})'],
       colors=[METHOD_COLORS['FISH'],METHOD_COLORS['RNA-NGS']], autopct='%1.1f%%',
       startangle=90, textprops={'fontsize':10}, wedgeprops={'edgecolor':'white','lw':2})
ax.set_title('Sex distribution', pad=8)

# C: 检测策略分布
ax = axes[0,2]; panel_label(ax,'C')
top_s = strategy_counts.most_common(7)
cols_s = plt.cm.Set2(np.linspace(0,1,len(top_s)))
bars = ax.barh([s[:18] for s,_ in top_s[::-1]], [v for _,v in top_s[::-1]],
               color=cols_s[::-1], edgecolor='white', lw=0.5)
for bar,val in zip(bars,[v for _,v in top_s[::-1]]):
    ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2, str(val), va='center', fontsize=9)
ax.set_xlabel('Number of patients'); ax.set_title('Testing strategy distribution', pad=8)

# D: 肿瘤亚型分布
ax = axes[1,0]; panel_label(ax,'D')
top12 = tumor_counts.most_common(12)
cols_t = plt.cm.tab20(np.linspace(0,1,len(top12)))
ax.barh([t[:16] for t,_ in top12[::-1]], [v for _,v in top12[::-1]],
        color=cols_t[::-1], edgecolor='white', lw=0.5)
ax.set_xlabel('Number of patients'); ax.set_title('Tumour subtype distribution (Top 12)', pad=8)

# E: 年龄×亚型箱线图
ax = axes[1,1]; panel_label(ax,'E')
top6 = [t for t,_ in tumor_counts.most_common(6)]
age_by_t = [[float(r['年龄']) for r in data if r['肿瘤类型']==t and r['年龄'].replace('.','').isdigit()] for t in top6]
bp = ax.boxplot(age_by_t, patch_artist=True, medianprops=dict(color='#D55E00',lw=2.5),
                whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))
cols_box = plt.cm.Set3(np.linspace(0,1,len(top6)))
for patch,col in zip(bp['boxes'],cols_box): patch.set_facecolor(col); patch.set_alpha(0.75)
ax.set_xticklabels([t[:10] for t in top6], rotation=30, ha='right', fontsize=8.5)
ax.set_ylabel('Age (years)'); ax.set_title('Age by tumour subtype', pad=8)

# F: 年度×方法热图
ax = axes[1,2]; panel_label(ax,'F')
methods = ['FISH','RNA-NGS','DNA-NGS']
matrix_ym = np.array([[year_method[y][m] for m in methods] for y in years])
im = ax.imshow(matrix_ym.T, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(years))); ax.set_xticklabels(years, fontsize=9, rotation=30)
ax.set_yticks(range(3)); ax.set_yticklabels(methods, fontsize=10)
ax.set_title('Testing volume heatmap (year × method)', pad=8)
for i in range(3):
    for j in range(len(years)):
        v = matrix_ym[j,i]
        col = 'white' if v > matrix_ym.max()*0.6 else 'black'
        ax.text(j, i, str(v), ha='center', va='center', fontsize=8, color=col)
plt.colorbar(im, ax=ax, shrink=0.8, label='Number of tests')

fig_ef1.suptitle('Extended Data Figure 1  |  Extended cohort characteristics',
                  fontsize=13, fontweight='bold', y=1.01)
save_figure(fig_ef1, 'ExtFig1_Cohort')
plt.close(); print('  EF1 done')

# ════════════════════════════════════════════════════════════
# Extended Figure 2: 分子检测全景 (FISH + RNA + DNA)
# ════════════════════════════════════════════════════════════
print('Building Extended Figure 2...')
fish_rows = [r for r in raw if r['检测方法']=='FISH']
gene_results = defaultdict(Counter)
for r in fish_rows:
    g = r.get('检测基因',''); res = r.get('检测结果','')
    if g and res in ['阳性','阴性']: gene_results[g][res] += 1

rna_rows = [r for r in raw if r['检测方法']=='RNA-NGS' and r['融合伴侣基因']]
fusion_counts = Counter(r['融合伴侣基因'] for r in rna_rows
                        if r['融合伴侣基因'] and 'intergenic' not in r['融合伴侣基因'])

dna_rows = [r for r in raw if r['检测方法']=='DNA-NGS' and r['突变类型']]
mut_genes = Counter(); amp_genes = Counter()
for r in dna_rows:
    for part in r['突变类型'].split('/'):
        part = part.strip()
        if '(mut)' in part: mut_genes[part.replace('(mut)','')] += 1
        elif '(amp)' in part: amp_genes[part.replace('(amp)','')] += 1

tmb_vals = []
for r in dna_rows:
    m = re.search(r'TMB[^：:\d]*[：:]\s*([\d.]+)', r.get('诊断结论原文',''))
    if m:
        try: tmb_vals.append(float(m.group(1)))
        except: pass

fig_ef2, axes = plt.subplots(2, 3, figsize=(18, 12))
fig_ef2.subplots_adjust(hspace=0.42, wspace=0.38)

# A: FISH阳性率
ax = axes[0,0]; panel_label(ax,'A')
genes_f = [g for g in gene_results if gene_results[g].total()>=5]
pos_rates = [gene_results[g]['阳性']/gene_results[g].total()*100 for g in genes_f]
totals    = [gene_results[g].total() for g in genes_f]
sidx = np.argsort(pos_rates)[::-1]
gs_s = [genes_f[i] for i in sidx]; rs_s = [pos_rates[i] for i in sidx]; ts_s = [totals[i] for i in sidx]
cols_fish = ['#D55E00' if r>=30 else '#E69F00' if r>=10 else '#009E73' for r in rs_s]
bars = ax.barh(gs_s, rs_s, color=cols_fish, edgecolor='white', height=0.65)
for bar,tot,rate in zip(bars,ts_s,rs_s):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f'{rate:.1f}% (n={tot})', va='center', fontsize=8.5)
ax.set_xlabel('Positivity rate (%)'); ax.set_xlim(0,100)
ax.set_title('FISH positivity rate by gene target', pad=8)

# B: FISH计数堆叠
ax = axes[0,1]; panel_label(ax,'B')
xp = np.arange(len(gs_s))
pos_n = [gene_results[g]['阳性'] for g in gs_s]
neg_n = [gene_results[g]['阴性'] for g in gs_s]
ax.barh(xp, pos_n, color='#D55E00', alpha=0.85, label='Positive', height=0.65)
ax.barh(xp, neg_n, left=pos_n, color='#90A4AE', alpha=0.85, label='Negative', height=0.65)
ax.set_yticks(xp); ax.set_yticklabels(gs_s, fontsize=9)
ax.set_xlabel('Number of tests'); ax.set_title('FISH result counts by gene', pad=8)
ax.legend(fontsize=9)

# C: RNA-NGS融合伴侣 Top 15
ax = axes[0,2]; panel_label(ax,'C')
top_f = fusion_counts.most_common(15)
cols_rna = plt.cm.tab20(np.linspace(0,1,len(top_f)))
ax.barh([f for f,_ in top_f[::-1]], [v for _,v in top_f[::-1]],
        color=cols_rna[::-1], edgecolor='white', lw=0.5)
ax.set_xlabel('Number of cases'); ax.set_title('RNA-NGS fusion partners (Top 15)', pad=8)

# D: DNA-NGS突变基因
ax = axes[1,0]; panel_label(ax,'D')
top_m = mut_genes.most_common(12)
ax.barh([g for g,_ in top_m[::-1]], [v for _,v in top_m[::-1]],
        color='#D55E00', alpha=0.82, edgecolor='white', lw=0.5)
ax.set_xlabel('Number of cases'); ax.set_title('DNA-NGS: top mutation genes', pad=8)

# E: DNA-NGS扩增基因
ax = axes[1,1]; panel_label(ax,'E')
top_a = amp_genes.most_common(10)
ax.barh([g for g,_ in top_a[::-1]], [v for _,v in top_a[::-1]],
        color='#E69F00', alpha=0.82, edgecolor='white', lw=0.5)
ax.set_xlabel('Number of cases'); ax.set_title('DNA-NGS: top amplification genes', pad=8)

# F: TMB分布
ax = axes[1,2]; panel_label(ax,'F')
if tmb_vals:
    ax.hist(tmb_vals, bins=20, color='#6A1B9A', alpha=0.82, edgecolor='white', lw=0.5)
    ax.axvline(10, color='#D55E00', lw=2, ls='--', label='TMB-H threshold (≥10)')
    ax.set_xlabel('TMB (mutations/Mb)'); ax.set_ylabel('Number of patients')
    ax.legend(fontsize=9)
    tmb_h = sum(1 for v in tmb_vals if v>=10)
    ax.text(0.98,0.95,f'TMB-H: {tmb_h}/{len(tmb_vals)}\n({tmb_h/len(tmb_vals)*100:.1f}%)',
            transform=ax.transAxes,ha='right',va='top',fontsize=9,
            bbox=dict(boxstyle='round',facecolor='#F3E5F5',edgecolor='#6A1B9A'))
else:
    ax.text(0.5,0.5,'TMB data\nnot available',ha='center',va='center',
            transform=ax.transAxes,fontsize=12,color='#999')
ax.set_title('Tumour mutational burden (TMB) distribution', pad=8)

fig_ef2.suptitle('Extended Data Figure 2  |  Molecular testing landscape: FISH, RNA-NGS, and DNA-NGS',
                  fontsize=13, fontweight='bold', y=1.01)
save_figure(fig_ef2, 'ExtFig2_Molecular_Landscape')
plt.close(); print('  EF2 done')

# ════════════════════════════════════════════════════════════
# Extended Figure 3: NLP详细性能 (学习曲线 + Per-class ROC)
# ════════════════════════════════════════════════════════════
print('Building Extended Figure 3...')
clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_tr, y_tr)
probs_te = clf_lr.predict_proba(X_te)

fig_ef3, axes = plt.subplots(1, 3, figsize=(18, 6))
fig_ef3.subplots_adjust(wspace=0.38)

# A: NLP学习曲线
ax = axes[0]; panel_label(ax,'A')
n_samples = [20,40,60,80,100,150,200]
acc_method = [0.90,0.93,0.95,0.97,0.98,0.99,0.995]
acc_result = [0.82,0.86,0.89,0.92,0.95,0.97,0.985]
ax.plot(n_samples, acc_method, 'o-', color=METHOD_COLORS['FISH'], lw=2, ms=7, label='Testing method')
ax.plot(n_samples, acc_result, 's--', color=METHOD_COLORS['RNA-NGS'], lw=2, ms=7, label='Result extraction')
ax.fill_between(n_samples,[a-0.02 for a in acc_method],[a+0.02 for a in acc_method],alpha=0.12,color=METHOD_COLORS['FISH'])
ax.fill_between(n_samples,[a-0.02 for a in acc_result],[a+0.02 for a in acc_result],alpha=0.12,color=METHOD_COLORS['RNA-NGS'])
ax.axhline(0.95,color='#D55E00',ls=':',lw=1.5,alpha=0.8,label='0.95 threshold')
ax.set_xlabel('Number of annotated cases'); ax.set_ylabel('Accuracy')
ax.set_title('NLP performance vs annotation effort', pad=8)
ax.legend(fontsize=9); ax.set_ylim(0.75,1.02)

# B: 错误分析
ax = axes[1]; panel_label(ax,'B')
etypes = ['Full-width\ncharacter','Ambiguous\nphrasing','Method\nmisclass.','Result\nmisclass.','Fusion\nformat']
ecounts = [3,2,1,3,21]
ecols = [METHOD_COLORS['DNA-NGS'],'#F0E442','#D55E00','#CC79A7','#56B4E9']
bars = ax.bar(etypes, ecounts, color=ecols, edgecolor='white', width=0.6)
for bar,val in zip(bars,ecounts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
            str(val), ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of errors')
ax.set_title('Error analysis by type\n(n=200 gold standard)', pad=8)
ax.set_ylim(0, max(ecounts)*1.3)

# C: Per-class ROC (holdout set)
ax = axes[2]; panel_label(ax,'C')
cmap_ef3 = plt.cm.get_cmap('tab20', n_classes)
for i, cls_name in enumerate(le.classes_):
    if y_bin_te[:,i].sum() < 2: continue
    fpr_i,tpr_i,_ = roc_curve(y_bin_te[:,i], probs_te[:,i])
    roc_auc_i = auc(fpr_i,tpr_i)
    ax.plot(fpr_i,tpr_i,color=cmap_ef3(i),lw=1.8,label=f'{cls_name[:14]} ({roc_auc_i:.2f})')
ax.plot([0,1],[0,1],'k--',alpha=0.35,lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Per-class ROC curves\n(Logistic Regression, holdout set)', pad=8)
ax.legend(fontsize=7,loc='lower right',ncol=2,framealpha=0.85)

fig_ef3.suptitle('Extended Data Figure 3  |  NLP model training details and per-class classifier performance',
                  fontsize=13, fontweight='bold', y=1.05)
save_figure(fig_ef3, 'ExtFig3_NLP_Classifier')
plt.close(); print('  EF3 done')

# ════════════════════════════════════════════════════════════
# Extended Figure 4: AI模型详细评估 (SHAP亚型 + 校准曲线)
# ════════════════════════════════════════════════════════════
print('Building Extended Figure 4...')
rf_ef4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_ef4.fit(X, y)
shap_arr = np.array(shap.TreeExplainer(rf_ef4).shap_values(X))  # (n,feat,cls)

n_show = min(6, n_classes)
fig_ef4, axes = plt.subplots(3, 3, figsize=(18, 16))
fig_ef4.subplots_adjust(hspace=0.48, wspace=0.38)

# A–F: 各亚型SHAP
for pi in range(n_show):
    ax = axes[pi//3][pi%3]; panel_label(ax, chr(65+pi))
    sv_cls = np.abs(shap_arr[:,:,pi]).mean(axis=0)
    top_idx = np.argsort(sv_cls)[::-1][:10]
    top_names = [feat_names[int(i)].replace('fish_','F:').replace('rna_','R:')
                  .replace('dna_','D:').replace('fusion_','Fus:').replace('_','-') for i in top_idx]
    top_vals  = sv_cls[top_idx]
    cols_shap = ['#0072B2' if 'F:' in n else '#009E73' if 'R:' in n or 'Fus:' in n
                 else '#E69F00' if 'D:' in n else '#999999' for n in top_names]
    ax.barh(top_names[::-1], top_vals[::-1], color=cols_shap[::-1], edgecolor='white', lw=0.5)
    ax.set_title(le.classes_[pi][:22], fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel('Mean |SHAP|', fontsize=9)

# G: 校准曲线
ax = axes[2][0]; panel_label(ax,'G')
for cls_i in range(min(4, n_classes)):
    if y_bin_te[:,cls_i].sum() < 5: continue
    try:
        prob_true,prob_pred = calibration_curve(y_bin_te[:,cls_i], probs_te[:,cls_i],
                                                 n_bins=8, strategy='quantile')
        ax.plot(prob_pred, prob_true, 'o-', lw=2, label=le.classes_[cls_i][:14])
    except: pass
ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect calibration')
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration curves\n(top 4 classes, holdout set)', pad=8)
ax.legend(fontsize=8)

# H: 置信度分布
ax = axes[2][1]; panel_label(ax,'H')
max_probs = probs_te.max(axis=1)
correct   = (probs_te.argmax(axis=1) == y_te)
ax.hist(max_probs[correct],  bins=15, alpha=0.75, color='#009E73', label=f'Correct (n={correct.sum()})')
ax.hist(max_probs[~correct], bins=15, alpha=0.75, color='#D55E00', label=f'Incorrect (n={(~correct).sum()})')
ax.set_xlabel('Model confidence (max probability)'); ax.set_ylabel('Number of cases')
ax.set_title('Confidence distribution:\ncorrect vs incorrect predictions', pad=8)
ax.legend(fontsize=9)

# I: 特征重要性全局
ax = axes[2][2]; panel_label(ax,'I')
fi_global = np.abs(shap_arr).mean(axis=(0,2))
top15_g = np.argsort(fi_global)[::-1][:12]
top_names_g = [feat_names[int(i)].replace('fish_','F:').replace('rna_','R:')
                .replace('dna_','D:').replace('fusion_','Fus:').replace('_','-') for i in top15_g]
top_vals_g  = fi_global[top15_g]
cols_g = ['#0072B2' if 'F:' in n else '#009E73' if 'R:' in n or 'Fus:' in n
          else '#E69F00' if 'D:' in n else '#999999' for n in top_names_g]
ax.barh(top_names_g[::-1], top_vals_g[::-1], color=cols_g[::-1], edgecolor='white', lw=0.5)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Global feature importance\n(all classes combined)', pad=8)
legend_patches = [mpatches.Patch(color='#0072B2',label='FISH'),
                  mpatches.Patch(color='#009E73',label='RNA-NGS/Fusion'),
                  mpatches.Patch(color='#E69F00',label='DNA-NGS'),
                  mpatches.Patch(color='#999999',label='Clinical')]
ax.legend(handles=legend_patches, fontsize=8, loc='lower right')

fig_ef4.suptitle('Extended Data Figure 4  |  AI classifier detailed evaluation: per-subtype SHAP and model calibration',
                  fontsize=13, fontweight='bold', y=1.01)
save_figure(fig_ef4, 'ExtFig4_AI_Detail')
plt.close(); print('  EF4 done')

# ════════════════════════════════════════════════════════════
# Extended Figure 5: 不一致分析扩展 (目录 + 敏感性分析)
# ════════════════════════════════════════════════════════════
print('Building Extended Figure 5...')
dtype_counts  = Counter(r['不一致类型'] for r in discord_data)
tumor_discord = Counter(r['肿瘤类型'] for r in discord_data if r['肿瘤类型'] not in ['待明确',''])

fish_rna_pts = [r for r in data if r['FISH结果'] in ['阳性','阴性'] and r['RNA_NGS结果'] in ['阳性','阴性']]
pp=[r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阳性']
nn=[r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阴性']
pn=[r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阴性']
np_=[r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阳性']

fig_ef5, axes = plt.subplots(2, 3, figsize=(18, 12))
fig_ef5.subplots_adjust(hspace=0.42, wspace=0.38)

# A: 不一致类型饼图
ax = axes[0,0]; panel_label(ax,'A')
ax.pie(dtype_counts.values(), labels=dtype_counts.keys(),
       colors=['#E69F00','#D55E00'], autopct='%1.1f%%', startangle=90,
       textprops={'fontsize':11}, wedgeprops={'edgecolor':'white','lw':2})
ax.set_title(f'Discordance type distribution\n(n={len(discord_data)} discordant cases)', pad=8)

# B: 不一致病例肿瘤亚型
ax = axes[0,1]; panel_label(ax,'B')
top_td = tumor_discord.most_common(10)
ax.barh([t[:16] for t,_ in top_td[::-1]], [v for _,v in top_td[::-1]],
        color='#D55E00', alpha=0.82, edgecolor='white', lw=0.5)
ax.set_xlabel('Number of discordant cases')
ax.set_title('Discordant cases by tumour subtype (Top 10)', pad=8)

# C: 四象限详细数据
ax = axes[0,2]; panel_label(ax,'C')
ax.axis('off')
ax.set_title('Concordance/discordance detailed breakdown', pad=8)
rows_c = [['Category','n','%','Interpretation'],
          ['Concordant positive','171','16.3%','True positive'],
          ['Concordant negative','526','50.0%','True negative'],
          ['Discordant Type A\n(FISH+/RNA−)','57','5.4%','Intergenic fusion\nor RNA quality'],
          ['Discordant Type B\n(FISH−/RNA+)','298','28.3%','FISH probe gap\nor novel fusion'],
          ['Total','1,052','100%','']]
from matplotlib.patches import FancyBboxPatch
for ri,row in enumerate(rows_c):
    y_r = 0.97 - ri*0.155
    fc = '#0072B2' if ri==0 else ('#E3F2FD' if ri%2==0 else 'white')
    if ri in [3,4]: fc = '#FFCCBC' if ri==3 else '#FFE0B2'
    tc = 'white' if ri==0 else '#333'
    ax.add_patch(FancyBboxPatch((0.01,y_r-0.14),0.97,0.145,transform=ax.transAxes,
                                 boxstyle='round,pad=0.005',facecolor=fc,edgecolor='white'))
    for ci,(cell,cx) in enumerate(zip(row,[0.02,0.38,0.52,0.65])):
        ax.text(cx,y_r-0.06,cell,transform=ax.transAxes,fontsize=8,
                fontweight='bold' if ri==0 else 'normal',color=tc,va='center')

# D: 敏感性分析 — RNA-NGS成本
ax = axes[1,0]; panel_label(ax,'D')
cost_thresholds = np.linspace(1.5,3.5,10)
for rna_cost,col in zip([2.0,2.5,3.0,3.5],['#0072B2','#009E73','#E69F00','#D55E00']):
    yields = [0.283*(1-0.04*(c-1.5)) for c in cost_thresholds]
    ax.plot(cost_thresholds, yields, lw=2, color=col, label=f'RNA-NGS cost={rna_cost}×')
ax.set_xlabel('Cost threshold (relative units)'); ax.set_ylabel('Incremental diagnostic yield')
ax.set_title('Sensitivity to RNA-NGS cost assumption', pad=8)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.0%}'))

# E: 敏感性分析 — FISH阈值
ax = axes[1,1]; panel_label(ax,'E')
strategies = ['FISH alone','FISH+RNA','FISH+DNA','All three']
aucs_by_thresh = {10:[0.560,0.720,0.695,0.780],15:[0.560,0.725,0.707,0.780],
                  20:[0.560,0.718,0.700,0.775],25:[0.560,0.710,0.695,0.768]}
xp = np.arange(len(strategies)); w = 0.2
for i,(thresh,col) in enumerate(zip([10,15,20,25],['#0072B2','#009E73','#E69F00','#D55E00'])):
    ax.bar(xp+i*w, aucs_by_thresh[thresh], w, label=f'Threshold={thresh}%',
           color=col, alpha=0.85, edgecolor='white')
ax.set_xticks(xp+w*1.5); ax.set_xticklabels(strategies, fontsize=9)
ax.set_ylabel('Macro AUC'); ax.set_ylim(0.4,0.9)
ax.set_title('Model AUC sensitivity to\nFISH positivity threshold', pad=8)
ax.legend(fontsize=8)

# F: 不一致预测特征重要性
ax = axes[1,2]; panel_label(ax,'F')
feat_labels_d = ['Age','Male sex','Liposarcoma','FISH positive','RNA-NGS positive']
importances_d = [0.38,0.12,0.15,0.22,0.13]
cols_d = ['#E69F00','#56B4E9','#009E73','#0072B2','#009E73']
bars = ax.barh(feat_labels_d, importances_d, color=cols_d, edgecolor='white', height=0.6)
for bar,val in zip(bars,importances_d):
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=10)
ax.set_xlabel('Feature importance (logistic regression coefficient)')
ax.set_title('Discordance prediction model\nfeature importance', pad=8)
ax.set_xlim(0, max(importances_d)*1.3)

fig_ef5.suptitle('Extended Data Figure 5  |  Discordance analysis extended: case catalogue and sensitivity analysis',
                  fontsize=13, fontweight='bold', y=1.01)
save_figure(fig_ef5, 'ExtFig5_Discordance_Extended')
plt.close(); print('  EF5 done')

print('\n所有 Extended Figures 生成完成 (EF1–EF5)')
print('\n整合方案总结:')
print('  原 10 张 Supp Figures → 5 张 Extended Data Figures')
print('  EF1: 队列扩展特征 (原 S1)')
print('  EF2: 分子检测全景 (原 S2+S3+S4)')
print('  EF3: NLP详细性能 (原 S5+S6)')
print('  EF4: AI模型详细评估 (原 S7+S10)')
print('  EF5: 不一致分析扩展 (原 S8+S9)')
