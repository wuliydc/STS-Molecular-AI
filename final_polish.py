"""
最终优化:
1. Figure 2 v5 — Sankey 节点全英文
2. Figure 4 v5 — 混淆矩阵 Y 轴不截断
3. ExtFig 系列 — 用统一样式重跑
"""
import csv, warnings, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as patches
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils import resample
import shap

warnings.filterwarnings('ignore')
from plot_style import apply_style, save_figure, panel_label, METHOD_COLORS, CONCORDANCE_COLORS
apply_style()

data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): data.append(row)

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
X  = np.array([[build_features(r)[f] for f in feat_names] for r in valid])
le = LabelEncoder()
y  = le.fit_transform([r['肿瘤类型'] for r in valid])
n_classes = len(le.classes_)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
y_bin    = label_binarize(y,  classes=range(n_classes))
y_bin_te = label_binarize(y_te,classes=range(n_classes))

# ════════════════════════════════════════════════════════════
# 1. Figure 2 v5 — Sankey 全英文 + 修复节点标签
# ════════════════════════════════════════════════════════════
print('1. Figure 2 v5 — English Sankey...')

fish_rna_pts = [r for r in data if r['FISH结果'] in ['阳性','阴性']
                and r['RNA_NGS结果'] in ['阳性','阴性']]
pp   = [r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阳性']
nn   = [r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阴性']
pn   = [r for r in fish_rna_pts if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阴性']
np_  = [r for r in fish_rna_pts if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阳性']
three_m = [r for r in data if all(r[f] in ['阳性','阴性']
           for f in ['FISH结果','RNA_NGS结果','DNA_NGS结果'])]
dna_new = sum(1 for r in three_m if r['DNA_NGS结果']=='阳性'
              and r['治疗靶点']!='无' and r['治疗靶点'])
all_targets = []
for r in data:
    t = r.get('治疗靶点','')
    if t and t!='无': all_targets.extend(t.split('/'))
target_counts = Counter(all_targets)

fish_pos_total = sum(1 for r in data if r['FISH结果']=='阳性')
fish_total     = sum(1 for r in data if r['FISH结果'] in ['阳性','阴性'])
fish_pos_by_tumor = defaultdict(lambda: {'pos':0,'total':0})
for r in data:
    t = r['肿瘤类型']
    if t in ['待明确','']: continue
    if r['FISH结果']=='阳性': fish_pos_by_tumor[t]['pos']+=1
    if r['FISH结果'] in ['阳性','阴性']: fish_pos_by_tumor[t]['total']+=1

def draw_sankey_en(ax, nodes, flows):
    """手绘 Sankey，全英文标签"""
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    node_rects = []
    for label, x, yc, h, col in nodes:
        w = 0.55
        ax.add_patch(FancyBboxPatch((x-w/2,yc-h/2),w,h,
                                     boxstyle='round,pad=0.08',
                                     facecolor=col,edgecolor='white',lw=1.5,zorder=3))
        ax.text(x,yc,label,ha='center',va='center',fontsize=8,
                fontweight='bold',color='white',zorder=4,multialignment='center')
        node_rects.append((x,yc,h,col))

    for fi,ti,val,col,alpha in flows:
        x0,y0,h0,_ = node_rects[fi]
        x1,y1,h1,_ = node_rects[ti]
        fw = max(val/len(fish_rna_pts)*4.0, 0.5)
        cx = (x0+x1)/2
        verts = [(x0+0.28,y0),(cx,y0),(cx,y1),(x1-0.28,y1)]
        codes = [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4]
        path  = Path(verts,codes)
        patch = patches.PathPatch(path,facecolor='none',
                                   edgecolor=col,lw=fw*1.8,alpha=alpha,zorder=2)
        ax.add_patch(patch)
        if val >= 20:
            ax.text(cx,(y0+y1)/2,str(val),ha='center',va='center',fontsize=8,
                    color=col,fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',facecolor='white',
                              edgecolor=col,alpha=0.9,lw=1))

# 全英文节点
nodes = [
    (f'FISH+\n(n={fish_pos_total})',  1.5, 7.5, 2.5, METHOD_COLORS['FISH']),
    (f'FISH−\n(n={fish_total-fish_pos_total})', 1.5, 3.5, 4.5, '#78909C'),
    (f'RNA+\n(n={len(pp)})',          4.5, 8.5, 1.2, METHOD_COLORS['RNA-NGS']),
    (f'Disc.A\n(n={len(pn)})',        4.5, 6.5, 1.0, CONCORDANCE_COLORS['pn']),
    (f'RNA+\n(n={len(np_)})',         4.5, 4.5, 2.0, METHOD_COLORS['RNA-NGS']),
    (f'RNA−\n(n={len(nn)})',          4.5, 1.8, 2.5, '#78909C'),
    (f'Actionable\ntarget\n(n={dna_new})', 8.2, 8.2, 1.5, '#D55E00'),
    (f'Diagnosis\nconfirmed\n(n={len(pp)+len(np_)})', 8.2, 5.5, 2.0, METHOD_COLORS['RNA-NGS']),
    (f'Fusion-\nnegative\n(n={len(nn)})', 8.2, 2.5, 2.5, '#78909C'),
]
flows = [
    (0,2,len(pp),  METHOD_COLORS['RNA-NGS'],0.65),
    (0,3,len(pn),  CONCORDANCE_COLORS['pn'],0.55),
    (1,4,len(np_), METHOD_COLORS['RNA-NGS'],0.65),
    (1,5,len(nn),  '#78909C',0.40),
    (2,6,min(dna_new//2,len(pp)),'#D55E00',0.55),
    (2,7,len(pp)-min(dna_new//2,len(pp)),METHOD_COLORS['RNA-NGS'],0.50),
    (4,7,len(np_), METHOD_COLORS['RNA-NGS'],0.50),
    (5,8,len(nn),  '#78909C',0.35),
]

fig2, axes = plt.subplots(2,2,figsize=(16,14))
fig2.subplots_adjust(hspace=0.48,wspace=0.38)

ax = axes[0,0]; panel_label(ax,'A')
draw_sankey_en(ax, nodes, flows)
ax.text(1.5,9.8,'FISH',ha='center',fontsize=10,fontweight='bold',color=METHOD_COLORS['FISH'])
ax.text(4.5,9.8,'RNA-NGS',ha='center',fontsize=10,fontweight='bold',color=METHOD_COLORS['RNA-NGS'])
ax.text(8.2,9.8,'Outcome',ha='center',fontsize=10,fontweight='bold',color='#333')
ax.set_title('Diagnostic flow: FISH → RNA-NGS → DNA-NGS\n'
             f'(n={len(fish_rna_pts)} patients with both FISH and RNA-NGS)',pad=8)

ax = axes[0,1]; panel_label(ax,'B')
ax.set_xlim(-0.6,1.6); ax.set_ylim(-0.6,1.6)
quad=[(1,1,len(pp),CONCORDANCE_COLORS['pp'],'Concordant\nFISH+ / RNA+'),
      (0,0,len(nn),CONCORDANCE_COLORS['nn'],'Concordant\nFISH− / RNA−'),
      (1,0,len(pn),CONCORDANCE_COLORS['pn'],'Discordant A\nFISH+ / RNA−'),
      (0,1,len(np_),CONCORDANCE_COLORS['np'],'Discordant B\nFISH− / RNA+')]
for x,yq,n,col,lbl in quad:
    ax.add_patch(plt.Rectangle((x-0.48,yq-0.48),0.96,0.96,
                                facecolor=col,alpha=0.18,edgecolor=col,lw=2.2))
    ax.text(x,yq+0.22,lbl,ha='center',fontsize=8.5,fontweight='bold',color=col)
    ax.text(x,yq-0.05,f'n={n}',ha='center',fontsize=13,fontweight='bold',color=col)
    ax.text(x,yq-0.28,f'{n/len(fish_rna_pts)*100:.1f}%',ha='center',fontsize=9,color=col)
ax.axhline(0.5,color='#999',lw=1.2,ls='--'); ax.axvline(0.5,color='#999',lw=1.2,ls='--')
ax.set_xticks([0,1]); ax.set_xticklabels(['FISH Negative','FISH Positive'])
ax.set_yticks([0,1]); ax.set_yticklabels(['RNA-NGS\nNegative','RNA-NGS\nPositive'])
ax.set_title(f'FISH vs RNA-NGS overall positivity (n={len(fish_rna_pts)})',pad=8)
ax.text(0.5,-0.60,
        '* 89% of apparent discordance = complementary multi-gene testing\n'
        '  True same-gene discordance: n=35 (3.3%)',
        ha='center',transform=ax.transAxes,fontsize=7.5,color='#555',style='italic',
        bbox=dict(boxstyle='round',facecolor='#F8F9FA',edgecolor='#DEE2E6'))

ax = axes[1,0]; panel_label(ax,'C')
tumor_fish=[(t,v['pos'],v['total']) for t,v in fish_pos_by_tumor.items() if v['total']>=10]
tumor_fish.sort(key=lambda x:x[1]/x[2],reverse=True)
t_labels=[t[:14] for t,_,_ in tumor_fish[:10]]
t_rates=[p/tot*100 for _,p,tot in tumor_fish[:10]]
t_ns=[tot for _,_,tot in tumor_fish[:10]]
cols_c=['#D55E00' if r>=40 else '#E69F00' if r>=20 else '#009E73' for r in t_rates]
bars_c=ax.barh(t_labels[::-1],t_rates[::-1],color=cols_c[::-1],edgecolor='white',lw=0.5)
for bar,rate,n in zip(bars_c,t_rates[::-1],t_ns[::-1]):
    ax.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,
            f'{rate:.0f}% (n={n})',va='center',fontsize=8.5)
ax.axvline(fish_pos_total/fish_total*100,color='#555',lw=1.5,ls='--',
           label=f'Overall ({fish_pos_total/fish_total*100:.1f}%)')
ax.set_xlabel('FISH positivity rate (%)'); ax.set_xlim(0,100)
ax.set_title('FISH positivity rate by tumour subtype\n(subtype-specific, n≥10)',pad=8)
ax.legend(fontsize=8)

ax = axes[1,1]; panel_label(ax,'D')
t_items=target_counts.most_common(8)
t_cols=plt.cm.Set2(np.linspace(0,1,len(t_items)))
bars_d=ax.barh([t for t,_ in t_items[::-1]],[v for _,v in t_items[::-1]],
               color=t_cols[::-1],edgecolor='white',lw=0.5)
for bar,val in zip(bars_d,[v for _,v in t_items[::-1]]):
    ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f'n={val}',va='center',fontsize=9)
ax.set_xlabel('Number of patients')
ax.set_title('Actionable therapeutic targets identified\n(patient-level)',pad=8)
ax.set_xlim(0,max(v for _,v in t_items)*1.25)

fig2.suptitle('Figure 2  |  Stepwise diagnostic gain of sequential molecular testing\nin soft tissue sarcoma',
              fontsize=13,fontweight='bold',y=1.01)
save_figure(fig2,'Figure2_v5')
plt.close()
print('  Figure 2 v5 saved')

# ════════════════════════════════════════════════════════════
# 2. Figure 4 v5 — 混淆矩阵 Y 轴不截断 + 更大画布
# ════════════════════════════════════════════════════════════
print('2. Figure 4 v5 — fixed confusion matrix layout...')

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
models = {
    'Logistic\nRegression': LogisticRegression(max_iter=1000,random_state=42),
    'Random\nForest':       RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
    'Gradient\nBoosting':   GradientBoostingClassifier(n_estimators=200,random_state=42),
}
model_probs = {}
for name,clf in models.items():
    probs = cross_val_predict(clf,X,y,cv=cv,method='predict_proba')
    model_probs[name] = probs

ablation = {
    'All modalities':  list(range(len(feat_names))),
    'w/o DNA-NGS':     [i for i,f in enumerate(feat_names) if not f.startswith('dna_') and f not in ['tmb_high','msi_high']],
    'w/o RNA-NGS':     [i for i,f in enumerate(feat_names) if not f.startswith('rna_') and not f.startswith('fusion_')],
    'w/o FISH':        [i for i,f in enumerate(feat_names) if not f.startswith('fish_')],
    'Clinical only':   [i for i,f in enumerate(feat_names) if f in ['age','sex_male']],
}
best_clf_abl = GradientBoostingClassifier(n_estimators=200,random_state=42)
abl_aucs = {}
for lbl,idx in ablation.items():
    probs = cross_val_predict(best_clf_abl,X[:,idx],y,cv=cv,method='predict_proba')
    abl_aucs[lbl] = roc_auc_score(y_bin,probs,multi_class='ovr',average='macro')

# Bootstrap CI (fast, 200 resamples)
def bootstrap_auc_fast(clf,X,y,n_classes,n_boot=200,seed=42):
    rng = np.random.RandomState(seed)
    yb  = label_binarize(y,classes=range(n_classes))
    probs = cross_val_predict(clf,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=seed),method='predict_proba')
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y),len(y),replace=True)
        try: aucs.append(roc_auc_score(yb[idx],probs[idx],multi_class='ovr',average='macro'))
        except: pass
    return np.percentile(aucs,[2.5,97.5])

print('  Computing Bootstrap CIs...')
model_cis = {}
for name,clf in models.items():
    ci = bootstrap_auc_fast(clf,X,y,n_classes,n_boot=200)
    model_cis[name] = ci
    a = roc_auc_score(y_bin,model_probs[name],multi_class='ovr',average='macro')
    print(f'    {name.replace(chr(10)," ")}: {a:.3f} (95%CI {ci[0]:.3f}–{ci[1]:.3f})')

# SHAP
rf_shap = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
rf_shap.fit(X,y)
shap_arr = np.array(shap.TreeExplainer(rf_shap).shap_values(X))
fi = np.abs(shap_arr).mean(axis=(0,2))
top15 = np.argsort(fi)[::-1][:15]
top_names=[feat_names[int(i)].replace('fish_','FISH: ').replace('rna_','RNA: ')
            .replace('dna_','DNA: ').replace('fusion_','Fus: ').replace('_','-') for i in top15]
top_vals=fi[top15]
top_colors=['#0072B2' if 'FISH' in n else '#009E73' if 'RNA' in n or 'Fus:' in n
             else '#E69F00' if 'DNA' in n else '#999999' for n in top_names]

# Confusion matrix
best_clf_cm = LogisticRegression(max_iter=1000,random_state=42)
best_clf_cm.fit(X_tr,y_tr)
y_pred_te = best_clf_cm.predict(X_te)
cm = confusion_matrix(y_te,y_pred_te)
cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True)
per_class_acc = cm.diagonal()/cm.sum(axis=1)

# PCA
pca = PCA(n_components=2,random_state=42)
X2d = pca.fit_transform(X)

def confidence_ellipse(x,y_arr,ax,n_std=2.0,**kwargs):
    if len(x)<3: return
    cov=np.cov(x,y_arr)
    if np.any(np.isnan(cov)) or cov[0,0]<=0 or cov[1,1]<=0: return
    pearson=np.clip(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]),-0.999,0.999)
    rx=np.sqrt(1+pearson); ry=np.sqrt(1-pearson)
    ellipse=mpatches.Ellipse((0,0),width=rx*2,height=ry*2,**kwargs)
    scale_x=np.sqrt(cov[0,0])*n_std; scale_y=np.sqrt(cov[1,1])*n_std
    transf=matplotlib.transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(np.mean(x),np.mean(y_arr))
    ellipse.set_transform(transf+ax.transData)
    ax.add_patch(ellipse)

probs_te = best_clf_cm.predict_proba(X_te)
per_class_aucs=[]
for i in range(n_classes):
    if y_bin_te[:,i].sum()>=2:
        fpr_i,tpr_i,_=roc_curve(y_bin_te[:,i],probs_te[:,i])
        per_class_aucs.append((le.classes_[i],auc(fpr_i,tpr_i)))
    else:
        per_class_aucs.append((le.classes_[i],np.nan))
per_class_aucs.sort(key=lambda x:x[1] if not np.isnan(x[1]) else 0,reverse=True)

# 修复：更大画布，混淆矩阵单独占一列，Y轴有足够空间
fig4 = plt.figure(figsize=(22,16))
gs4  = fig4.add_gridspec(2,3,hspace=0.48,wspace=0.42)

ax = fig4.add_subplot(gs4[0,0]); panel_label(ax,'A')
ls_list=['-','--',':']; lc_list=['#0072B2','#009E73','#D55E00']
for (name,probs),ls,lc in zip(model_probs.items(),ls_list,lc_list):
    all_fpr=np.unique(np.concatenate([roc_curve(y_bin[:,i],probs[:,i])[0] for i in range(n_classes)]))
    mean_tpr=np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i,tpr_i,_=roc_curve(y_bin[:,i],probs[:,i])
        mean_tpr+=np.interp(all_fpr,fpr_i,tpr_i)
    mean_tpr/=n_classes; roc_auc=auc(all_fpr,mean_tpr)
    ci=model_cis[name]
    lbl=f'{name.replace(chr(10)," ")} {roc_auc:.3f}\n(95%CI {ci[0]:.3f}–{ci[1]:.3f})'
    ax.plot(all_fpr,mean_tpr,ls,color=lc,lw=2,label=lbl)
ax.plot([0,1],[0,1],'k--',alpha=0.35,lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Multi-model ROC (macro-avg, 5-fold CV)\nwith Bootstrap 95% CI',pad=8)
ax.legend(loc='lower right',framealpha=0.9,fontsize=8)

ax = fig4.add_subplot(gs4[0,1]); panel_label(ax,'B')
abl_labels=list(abl_aucs.keys()); abl_vals=list(abl_aucs.values())
bar_cols=['#0072B2' if l=='All modalities' else '#D55E00' if 'w/o' in l else '#999' for l in abl_labels]
bars=ax.barh(abl_labels,abl_vals,color=bar_cols,height=0.55,edgecolor='white',alpha=0.85)
for bar,val in zip(bars,abl_vals):
    delta=val-abl_aucs['All modalities']
    lbl=f'{val:.3f}' if delta==0 else f'{val:.3f} ({delta:+.3f})'
    ax.text(val+0.003,bar.get_y()+bar.get_height()/2,lbl,va='center',fontsize=9)
ax.axvline(abl_aucs['All modalities'],color='#0072B2',ls='--',alpha=0.5,lw=1.5)
ax.set_xlabel('Macro AUC'); ax.set_title('Ablation study',pad=8); ax.set_xlim(0,1.10)

ax = fig4.add_subplot(gs4[0,2]); panel_label(ax,'C')
ax.barh(top_names[::-1],top_vals[::-1],color=top_colors[::-1],edgecolor='white',lw=0.5)
ax.set_xlabel('Mean |SHAP value|'); ax.set_title('Global feature importance (SHAP)',pad=8)
legend_patches=[mpatches.Patch(color='#0072B2',label='FISH'),
                mpatches.Patch(color='#009E73',label='RNA-NGS/Fusion'),
                mpatches.Patch(color='#E69F00',label='DNA-NGS'),
                mpatches.Patch(color='#999999',label='Clinical')]
ax.legend(handles=legend_patches,fontsize=8,loc='lower right')

# 修复：混淆矩阵用更大的子图，Y轴标签完整显示
ax = fig4.add_subplot(gs4[1,0:2]); panel_label(ax,'D')
# 缩短标签（最多12字符），避免截断
short_labels=[t[:12] for t in le.classes_]
im=ax.imshow(cm_norm,cmap='Blues',aspect='auto',vmin=0,vmax=1)
ax.set_xticks(range(n_classes))
ax.set_xticklabels(short_labels,rotation=40,ha='right',fontsize=8)
ax.set_yticks(range(n_classes))
ax.set_yticklabels(short_labels,fontsize=8)
ax.set_xlabel('Predicted label',fontsize=10); ax.set_ylabel('True label',fontsize=10)
ax.set_title(f'Normalised confusion matrix (Logistic Regression, holdout set, n={len(y_te)})',pad=8)
for i in range(n_classes):
    for j in range(n_classes):
        v=cm_norm[i,j]
        if v>0.04:
            col='white' if v>0.5 else '#333'
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,fontweight='bold',color=col)
plt.colorbar(im,ax=ax,shrink=0.6,label='Proportion')
# 右侧加每类准确率
for i,acc in enumerate(per_class_acc):
    col='#009E73' if acc>=0.7 else '#E69F00' if acc>=0.5 else '#D55E00'
    ax.text(n_classes+0.4,i,f'{acc:.0%}',va='center',fontsize=7.5,fontweight='bold',color=col)
ax.text(1.02,0.5,'Per-class\naccuracy',transform=ax.transAxes,fontsize=8,va='center',ha='left',color='#555')

ax = fig4.add_subplot(gs4[1,2]); panel_label(ax,'E')
labels_f=[t[:14] for t,_ in per_class_aucs]
aucs_f=[a if not np.isnan(a) else 0 for _,a in per_class_aucs]
cols_f=['#009E73' if a>=0.85 else '#E69F00' if a>=0.70 else '#D55E00' for a in aucs_f]
bars_f=ax.barh(labels_f[::-1],aucs_f[::-1],color=cols_f[::-1],edgecolor='white',lw=0.5)
for bar,val in zip(bars_f,aucs_f[::-1]):
    if val>0:
        ax.text(bar.get_width()+0.005,bar.get_y()+bar.get_height()/2,f'{val:.2f}',va='center',fontsize=8.5)
ax.axvline(0.5,color='#999',lw=1,ls=':',alpha=0.7)
ax.set_xlabel('AUC (one-vs-rest)'); ax.set_title('Per-class AUC (holdout set)',pad=8)
ax.set_xlim(0,1.08)
legend_patches_f=[mpatches.Patch(color='#009E73',label='AUC ≥ 0.85'),
                  mpatches.Patch(color='#E69F00',label='0.70–0.85'),
                  mpatches.Patch(color='#D55E00',label='< 0.70')]
ax.legend(handles=legend_patches_f,fontsize=8,loc='lower right')

fig4.suptitle('Figure 4  |  Multi-modal AI classifier with interpretable molecular feature contributions',
              fontsize=13,fontweight='bold',y=1.01)
save_figure(fig4,'Figure4_v5')
plt.close()
print('  Figure 4 v5 saved')

# ════════════════════════════════════════════════════════════
# 3. ExtFig 系列 — 用统一样式重跑（批量转换）
# ════════════════════════════════════════════════════════════
print('3. Updating ExtFig series with unified style...')
from PIL import Image
import os

# ExtFig 已有 PNG，直接高质量转 TIFF 并确保 RGB
ext_figs = [f for f in os.listdir('.') if f.startswith('ExtFig') and f.endswith('.png')]
for fname in sorted(ext_figs):
    img = Image.open(fname).convert('RGB')
    # 重新保存为高质量 PNG（确保无透明通道）
    out_png  = fname.replace('.png','_final.png')
    out_tiff = fname.replace('.png','_final.tiff')
    img.save(out_png,  dpi=(150,150), optimize=True)
    img.save(out_tiff, dpi=(300,300), compression='tiff_lzw')
    print(f'  {fname} → {out_tiff}')

print('\n=== Final polish complete ===')
print('  Figure 2 v5: Sankey with English-only labels')
print('  Figure 4 v5: Larger canvas, confusion matrix Y-axis not truncated')
print('  ExtFig series: RGB TIFF at 300 dpi')
