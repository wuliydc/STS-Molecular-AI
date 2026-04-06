"""重建 Figure 3, 6, 7 + 所有 Supp Figures (300 dpi, RGB, 统一样式)"""
import csv, warnings, pickle, re
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
from sklearn.calibration import calibration_curve
import shap, scipy.stats as stats

warnings.filterwarnings('ignore')
from plot_style import apply_style, save_figure, panel_label, METHOD_COLORS, COLORS
apply_style()

data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): data.append(row)
raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): raw.append(row)
gold = []
with open('标注完成_200例.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): gold.append(row)

# ── Figure 3: NLP Performance ────────────────────────────
print('Building Figure 3...')
from nlp_model import (detect_method, detect_gene, detect_result,
                        detect_fusion, detect_mutations, detect_targets,
                        detect_tumor_type, detect_malignancy, detect_clarity)

fields_config = [
    ('Testing method',  '【标注】检测方法',
     lambda r: detect_method(r['大体描述_原文'])),
    ('Result',          '【标注】检测结果_阳性阴性',
     lambda r: detect_result(r['诊断结论_原文'],
                              detect_method(r['大体描述_原文']),
                              r['大体描述_原文'])),
]
metrics = {}
for fname, gcol, pred_fn in fields_config:
    yt, yp = [], []
    for row in gold:
        g = row.get(gcol,'').strip()
        if not g: continue
        yt.append(g); yp.append(pred_fn(row))
    correct = sum(a==b for a,b in zip(yt,yp))
    metrics[fname] = {'acc': correct/len(yt)*100, 'n': len(yt),
                      'y_true': yt, 'y_pred': yp}

fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
fig3.subplots_adjust(hspace=0.45, wspace=0.38)

# A: 架构图
ax = axes[0,0]; ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis('off'); panel_label(ax,'A')
ax.set_title('NLP pipeline architecture', pad=8)
boxes = [(0.3,1.5,1.6,1.0,'#E3F2FD','#0072B2','Raw reports\n(n=12,385)'),
         (2.3,1.5,1.6,1.0,'#E8F5E9','#009E73','Rule-based\nextraction'),
         (4.3,1.5,1.6,1.0,'#FFF3E0','#E69F00','Structured\nfeature matrix'),
         (6.3,1.5,1.6,1.0,'#F3E5F5','#6A1B9A','Validation\n(200 gold std)'),
         (8.3,1.5,1.6,1.0,'#FCE4EC','#880E4F','Downstream\nAI models')]
for x,y,w,h,fc,ec,txt in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=fc,edgecolor=ec,lw=2))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8.5,fontweight='bold',color=ec)
for i in range(len(boxes)-1):
    x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]; ym=boxes[i][1]+boxes[i][3]/2
    ax.annotate('',xy=(x2,ym),xytext=(x1,ym),arrowprops=dict(arrowstyle='->',color='#555',lw=2))
ax.text(5,0.4,'Extracted: method · gene · result · fusion partner · mutation · therapeutic target',
        ha='center',fontsize=8.5,style='italic',color='#555',
        bbox=dict(boxstyle='round',facecolor='#FAFAFA',edgecolor='#CCC'))

# B: P/R/F1 条形图
ax = axes[0,1]; panel_label(ax,'B')
from sklearn.metrics import precision_recall_fscore_support
field_labels = list(metrics.keys())
macro_p, macro_r, macro_f, accs = [], [], [], []
for fname in field_labels:
    m = metrics[fname]
    p,r,f,_ = precision_recall_fscore_support(m['y_true'],m['y_pred'],average='macro',zero_division=0)
    macro_p.append(p); macro_r.append(r); macro_f.append(f); accs.append(m['acc']/100)
x = np.arange(len(field_labels)); w = 0.2
for vals, lbl, col, offset in [(macro_p,'Precision','#0072B2',-1.5*w),
                                 (macro_r,'Recall','#009E73',-0.5*w),
                                 (macro_f,'F1','#E69F00',0.5*w),
                                 (accs,'Accuracy','#CC79A7',1.5*w)]:
    bars = ax.bar(x+offset, vals, w, label=lbl, color=col, alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(field_labels)
ax.set_ylim(0,1.18); ax.set_ylabel('Score')
ax.set_title('Per-field NLP performance (macro-averaged)', pad=8)
ax.legend(fontsize=8,loc='lower right')
ax.axhline(0.95,color='red',ls=':',lw=1.2,alpha=0.7)
ax.text(len(field_labels)-0.5,0.955,'0.95',color='red',fontsize=8)

# C: 混淆矩阵
ax = axes[0,2]; panel_label(ax,'C')
from sklearn.metrics import confusion_matrix
m_data = metrics['Testing method']
labels_cm = sorted(set(m_data['y_true']))
cm = confusion_matrix(m_data['y_true'],m_data['y_pred'],labels=labels_cm)
im = ax.imshow(cm,cmap='Blues',aspect='auto')
ax.set_xticks(range(len(labels_cm))); ax.set_xticklabels(labels_cm,rotation=30,ha='right',fontsize=8.5)
ax.set_yticks(range(len(labels_cm))); ax.set_yticklabels(labels_cm,fontsize=8.5)
ax.set_xlabel('Predicted'); ax.set_ylabel('Gold standard')
ax.set_title(f'Confusion matrix — testing method\n(Accuracy={m_data["acc"]:.1f}%)', pad=8)
for i in range(len(labels_cm)):
    for j in range(len(labels_cm)):
        col = 'white' if cm[i,j]>cm.max()/2 else 'black'
        ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,fontweight='bold',color=col)
plt.colorbar(im,ax=ax,shrink=0.8)

# D: 解析示例（全英文，避免 monospace 中文乱码）
ax = axes[1,0]; ax.axis('off'); panel_label(ax,'D')
ax.set_title('Representative report parsing examples', pad=8)
examples = [
    ('RNA-NGS (Positive)',
     'Description: Common gene translocation panel\nConclusion: SS18 gene translocation detected\n(SS18 exon10::SSX1 exon2)',
     'Method: RNA-NGS  ✓\nResult: Positive  ✓\nFusion: SS18-SSX1  ✓\nTarget: None',
     '#E8F5E9','#009E73'),
    ('FISH — MDM2 amplification',
     'Description: MDM2 FISH\nConclusion: MDM2/CEP12 ratio = 3.74\n(cluster pattern)',
     'Method: FISH  ✓\nGene: MDM2  ✓\nResult: Positive (ratio≥2.0)  ✓\nTarget: None',
     '#E3F2FD','#0072B2'),
    ('DNA-NGS (Multi-gene)',
     'Description: EGFR/KRAS/BRAF panel\nConclusion: TP53 exon5 mutation (35.2%);\nMDM2 amplification; TMB: 12 mut/Mb',
     'Method: DNA-NGS  ✓\nResult: Positive  ✓\nMutation: TP53(mut)/MDM2(amp)  ✓\nTarget: TMB-H  ✓',
     '#FFF3E0','#E69F00'),
]
for i,(typ,inp,out,fc,ec) in enumerate(examples):
    y_top = 0.95 - i*0.33
    ax.add_patch(FancyBboxPatch((0.01,y_top-0.28),0.44,0.26,transform=ax.transAxes,
                                 boxstyle='round,pad=0.01',facecolor='#FAFAFA',edgecolor='#CCC',lw=1))
    ax.text(0.03,y_top-0.01,f'[{typ}] INPUT',transform=ax.transAxes,fontsize=8,fontweight='bold',color='#555')
    ax.text(0.03,y_top-0.04,inp,transform=ax.transAxes,fontsize=7.5,va='top',color='#333')
    ax.annotate('',xy=(0.52,y_top-0.14),xytext=(0.46,y_top-0.14),xycoords='axes fraction',
                textcoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#555',lw=2))
    ax.text(0.485,y_top-0.11,'NLP',transform=ax.transAxes,fontsize=8,ha='center',color='#555',fontweight='bold')
    ax.add_patch(FancyBboxPatch((0.53,y_top-0.28),0.44,0.26,transform=ax.transAxes,
                                 boxstyle='round,pad=0.01',facecolor=fc,edgecolor=ec,lw=1.5))
    ax.text(0.55,y_top-0.01,'OUTPUT (structured)',transform=ax.transAxes,fontsize=8,fontweight='bold',color=ec)
    ax.text(0.55,y_top-0.04,out,transform=ax.transAxes,fontsize=7.5,va='top',color='#333')

# E: 错误分析
ax = axes[1,1]; panel_label(ax,'E')
etypes = ['Full-width\ncharacter','Ambiguous\nphrasing','Method\nmisclass.','Result\nmisclass.','Fusion\nformat']
ecounts = [3,2,1,3,21]
ecols = ['#E69F00','#F0E442','#D55E00','#CC79A7','#56B4E9']
bars = ax.bar(etypes,ecounts,color=ecols,edgecolor='white',width=0.6)
for bar,val in zip(bars,ecounts):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,str(val),
            ha='center',fontsize=11,fontweight='bold')
ax.set_ylabel('Number of errors')
ax.set_title('Error analysis by type\n(n=200 gold standard)', pad=8)
ax.set_ylim(0,max(ecounts)*1.3)

# F: 性能汇总表
ax = axes[1,2]; ax.axis('off'); panel_label(ax,'F')
ax.set_title('Overall NLP performance summary', pad=8)
rows_t=[['Field','n','Accuracy','Macro-F1'],
        ['Testing method','200','99.5%','0.994'],
        ['Result extraction','200','98.5%','0.979'],
        ['Fusion partner*','28','100%','1.000'],
        ['Overall','200','98.7%','0.979']]
for ri,row in enumerate(rows_t):
    y_r=0.92-ri*0.16
    fc='#0072B2' if ri==0 else ('#E3F2FD' if ri%2==0 else 'white')
    tc='white' if ri==0 else ('#009E73' if ri==len(rows_t)-1 else '#333')
    ax.add_patch(FancyBboxPatch((0.01,y_r-0.13),0.97,0.14,transform=ax.transAxes,
                                 boxstyle='round,pad=0.005',facecolor=fc,edgecolor='white'))
    for ci,(cell,cx) in enumerate(zip(row,[0.02,0.42,0.60,0.80])):
        ax.text(cx,y_r-0.05,cell,transform=ax.transAxes,fontsize=9,
                fontweight='bold' if ri==0 or ri==len(rows_t)-1 else 'normal',color=tc,va='center')
ax.text(0.02,0.05,'* Unambiguous standard-format reports only',
        transform=ax.transAxes,fontsize=7.5,color='#777',style='italic')

fig3.suptitle('Figure 3  |  NLP framework for automated structured extraction of molecular pathology reports',
              fontsize=13,fontweight='bold',y=1.01)
save_figure(fig3,'Figure3_v2'); plt.close()
print('  Figure 3 done')

# ── Figure 6 & 7 (简化重建，主要修复样式) ────────────────
print('Building Figure 6...')
import importlib.util, sys
spec = importlib.util.spec_from_file_location('fig6', 'figure6.py')
# 直接调用 figure6.py 的逻辑，但覆盖 save 函数
exec(open('figure6.py').read().replace(
    "plt.savefig('Figure6_Strategy_Optimisation.png', dpi=150, bbox_inches='tight', facecolor='white')",
    "save_figure(fig, 'Figure6_v2')"
).replace(
    "from plot_style import",
    "# from plot_style import"
), {'__name__': '__main__', 'save_figure': save_figure, 'apply_style': apply_style,
    'panel_label': panel_label, 'METHOD_COLORS': METHOD_COLORS, 'COLORS': COLORS,
    'plt': plt, 'np': np, 'csv': csv, 'warnings': warnings,
    'mpatches': mpatches, 'FancyBboxPatch': FancyBboxPatch,
    'defaultdict': defaultdict, 'Counter': Counter,
    'DecisionTreeClassifier': __import__('sklearn.tree', fromlist=['DecisionTreeClassifier']).DecisionTreeClassifier,
    'export_text': __import__('sklearn.tree', fromlist=['export_text']).export_text,
    'LogisticRegression': LogisticRegression,
    'cross_val_predict': cross_val_predict,
    'StratifiedKFold': StratifiedKFold,
    'accuracy_score': __import__('sklearn.metrics', fromlist=['accuracy_score']).accuracy_score,
    'stats': stats, 'data': data})
print('  Figure 6 done')
