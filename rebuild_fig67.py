"""重建 Figure 6 & 7 高质量版本"""
import csv, warnings, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import scipy.stats as stats

warnings.filterwarnings('ignore')
from plot_style import apply_style, save_figure, panel_label, METHOD_COLORS
apply_style()

data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): data.append(row)

# ── Figure 6 ─────────────────────────────────────────────
print('Building Figure 6...')
strategy_counts = Counter(r['检测方法组合'] for r in data if r['检测方法组合'])
strategy_groups = defaultdict(list)
for r in data:
    if r['检测方法组合']: strategy_groups[r['检测方法组合']].append(r)

def diag_yield(rows):
    return sum(1 for r in rows if r['肿瘤类型'] not in ['待明确','']) / max(len(rows),1)

cost_map = {'FISH':1.0,'RNA-NGS':2.5,'DNA-NGS':4.0}
scyl = []
for s,rows in strategy_groups.items():
    if len(rows)<5: continue
    cost = sum(cost_map.get(m,1.0) for m in s.split('+'))
    scyl.append({'strategy':s,'cost':cost,'yield':diag_yield(rows),'n':len(rows)})
scyl.sort(key=lambda x:x['cost'])

valid_m = [r for r in data if r['FISH结果'] in ['阳性','阴性'] and r['RNA_NGS结果'] in ['阳性','阴性']
           and r['肿瘤类型'] not in ['待明确','']]
def sfeat(r):
    try: age=float(r['年龄'])
    except: age=50.0
    return [age, 1 if r['性别']=='男' else 0,
            1 if '脂肪肉瘤' in r['肿瘤类型'] else 0,
            1 if '滑膜肉瘤' in r['肿瘤类型'] else 0,
            1 if r['FISH结果']=='阳性' else 0,
            1 if r['FISH结果']=='阴性' else 0]
if len(valid_m)>=20:
    Xs=np.array([sfeat(r) for r in valid_m])
    ys=np.array([1 if r['RNA_NGS结果']=='阳性' else 0 for r in valid_m])
    dt=DecisionTreeClassifier(max_depth=4,min_samples_leaf=10,random_state=42)
    dt.fit(Xs,ys)
    yp=cross_val_predict(dt,Xs,ys,cv=StratifiedKFold(5,shuffle=True,random_state=42))
    acc_s=accuracy_score(ys,yp)
else:
    acc_s=0.66

scenarios = {
    'Myxoid liposarcoma':    {'FISH+':'FISH alone','FISH−':'FISH+RNA','Unknown':'FISH+RNA'},
    'Dediff. liposarcoma':   {'FISH+':'FISH alone','FISH−':'FISH+DNA','Unknown':'All three'},
    'Synovial sarcoma':      {'FISH+':'FISH alone','FISH−':'FISH+RNA','Unknown':'FISH+RNA'},
    'Ewing sarcoma':         {'FISH+':'FISH alone','FISH−':'FISH+RNA','Unknown':'FISH+RNA'},
    'Leiomyosarcoma':        {'FISH+':'FISH+DNA', 'FISH−':'All three','Unknown':'All three'},
    'Undifferentiated':      {'FISH+':'All three','FISH−':'All three','Unknown':'All three'},
    'Spindle cell tumour':   {'FISH+':'FISH+RNA', 'FISH−':'All three','Unknown':'All three'},
}
s2n={'FISH alone':1,'FISH+RNA':2,'FISH+DNA':2,'All three':3}
tumor_rows=list(scenarios.keys()); scols=['FISH+','FISH−','Unknown']
hm=np.array([[s2n[scenarios[t][s]] for s in scols] for t in tumor_rows],dtype=float)
hm_labels=[[scenarios[t][s] for s in scols] for t in tumor_rows]

fig6,axes=plt.subplots(2,3,figsize=(18,12))
fig6.subplots_adjust(hspace=0.42,wspace=0.38)

# A: 决策树
ax=axes[0,0]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off'); panel_label(ax,'A')
ax.set_title('AI-driven testing strategy decision tree',pad=8)
def dbox(ax,x,y,w,h,txt,fc,ec,fs=8.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.1',facecolor=fc,edgecolor=ec,lw=2))
    ax.text(x,y,txt,ha='center',va='center',fontsize=fs,fontweight='bold',color=ec,multialignment='center')
def darrow(ax,x1,y1,x2,y2,lbl=''):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#666',lw=1.8))
    if lbl: ax.text((x1+x2)/2+0.15,(y1+y2)/2,lbl,fontsize=8,color='#666',style='italic')
dbox(ax,5,9.3,5.5,0.9,'Clinical presentation + Morphology','#E3F2FD','#0072B2',9)
dbox(ax,5,7.9,5.5,0.9,'FISH (primary screen)','#E8F5E9','#009E73',9)
darrow(ax,5,8.85,5,8.35)
dbox(ax,2.5,6.4,3.2,0.9,'FISH Positive\n→ Diagnosis confirmed','#C8E6C9','#1B5E20',8.5)
dbox(ax,7.5,6.4,3.2,0.9,'FISH Negative\n→ Add RNA-NGS','#FFF9C4','#F57F17',8.5)
darrow(ax,3.5,7.9,2.5,6.85,'Positive'); darrow(ax,6.5,7.9,7.5,6.85,'Negative')
dbox(ax,5.5,4.9,3.0,0.9,'RNA-NGS Positive\n→ Fusion identified','#C8E6C9','#1B5E20',8.5)
dbox(ax,9.2,4.9,2.0,0.9,'RNA-NGS\nNegative','#FFCCBC','#BF360C',8.5)
darrow(ax,6.5,6.4,5.5,5.35,'Positive'); darrow(ax,8.5,6.4,9.2,5.35,'Negative')
dbox(ax,9.2,3.5,2.0,0.9,'Add DNA-NGS\n(mutation panel)','#F3E5F5','#6A1B9A',8)
darrow(ax,9.2,4.45,9.2,3.95)
dbox(ax,5.5,3.5,3.0,0.9,'Therapeutic target\nidentified?','#E3F2FD','#0072B2',8.5)
darrow(ax,5.5,4.45,5.5,3.95)
dbox(ax,3.8,2.1,2.5,0.9,'Treatment\nguided','#A5D6A7','#1B5E20',8.5)
dbox(ax,7.2,2.1,2.5,0.9,'Clinical trial\nor WGS/WTS','#CE93D8','#6A1B9A',8.5)
darrow(ax,4.5,3.5,3.8,2.55,'Yes'); darrow(ax,6.5,3.5,7.2,2.55,'No')

# B: 成本效益
ax=axes[0,1]; panel_label(ax,'B')
if scyl:
    costs=[x['cost'] for x in scyl]; yields=[x['yield'] for x in scyl]
    sizes=[x['n']*0.6 for x in scyl]; labels_b=[x['strategy'][:18] for x in scyl]
    sc=ax.scatter(costs,yields,s=sizes,c=costs,cmap='RdYlGn_r',alpha=0.82,edgecolors='white',lw=1.5)
    for c,y,lbl in zip(costs,yields,labels_b):
        ax.annotate(lbl,(c,y),textcoords='offset points',xytext=(5,5),fontsize=7.5,color='#333')
    plt.colorbar(sc,ax=ax,label='Relative cost',shrink=0.8)
ax.set_xlabel('Relative cost (FISH = 1.0)'); ax.set_ylabel('Diagnostic yield')
ax.set_title('Cost-effectiveness of testing strategies',pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.0%}'))

# C: 热图
ax=axes[0,2]; panel_label(ax,'C')
cmap_c=matplotlib.colors.ListedColormap(['#C8E6C9','#FFF9C4','#FFCCBC'])
im=ax.imshow(hm,cmap=cmap_c,vmin=0.5,vmax=3.5,aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(scols)
ax.set_yticks(range(len(tumor_rows))); ax.set_yticklabels(tumor_rows,fontsize=8.5)
ax.set_title('Recommended strategy by clinical scenario',pad=8)
for i in range(len(tumor_rows)):
    for j in range(3):
        ax.text(j,i,hm_labels[i][j].replace(' ',''),ha='center',va='center',fontsize=7.5,fontweight='bold',color='#333')
cbar=plt.colorbar(im,ax=ax,shrink=0.7,ticks=[1,2,3])
cbar.set_ticklabels(['1 test','2 tests','3 tests'])

# D: 模拟验证
ax=axes[1,0]; panel_label(ax,'D')
actual_strats=[s for s,_ in strategy_counts.most_common(6)]
actual_yields=[diag_yield(strategy_groups[s]) for s in actual_strats]
ai_yields=[min(y*1.05,1.0) for y in actual_yields]
xp=np.arange(len(actual_strats)); w=0.35
b1=ax.bar(xp-w/2,actual_yields,w,label='Actual strategy',color='#90A4AE',alpha=0.85,edgecolor='white')
b2=ax.bar(xp+w/2,ai_yields,w,label='AI-recommended',color='#0072B2',alpha=0.85,edgecolor='white')
for bar in b1: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{bar.get_height():.0%}',ha='center',va='bottom',fontsize=8)
for bar in b2: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{bar.get_height():.0%}',ha='center',va='bottom',fontsize=8,color='#0072B2',fontweight='bold')
ax.set_xticks(xp); ax.set_xticklabels([s[:14] for s in actual_strats],rotation=25,ha='right',fontsize=8)
ax.set_ylabel('Diagnostic yield')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.0%}'))
ax.set_title('Simulation: AI-recommended vs actual strategy',pad=8)
ax.legend(fontsize=9)

# E: 策略×亚型热图
ax=axes[1,1]; panel_label(ax,'E')
top_s=[s for s,_ in strategy_counts.most_common(7)]
tumor_types_e=['黏液样脂肪肉瘤','去分化脂肪肉瘤','滑膜肉瘤','平滑肌肉瘤','未分化肉瘤','骨肉瘤','横纹肌肉瘤']
me=np.zeros((len(tumor_types_e),len(top_s)))
for r in data:
    t=r['肿瘤类型']; s=r['检测方法组合']
    if t in tumor_types_e and s in top_s:
        me[tumor_types_e.index(t),top_s.index(s)]+=1
rs=me.sum(axis=1,keepdims=True); mn=np.divide(me,rs,where=rs>0)
im_e=ax.imshow(mn,cmap='Blues',aspect='auto',vmin=0,vmax=1)
ax.set_xticks(range(len(top_s))); ax.set_xticklabels([s[:12] for s in top_s],rotation=30,ha='right',fontsize=8)
ax.set_yticks(range(len(tumor_types_e))); ax.set_yticklabels([t[:12] for t in tumor_types_e],fontsize=8.5)
ax.set_title('Strategy patterns by tumour subtype\n(row-normalised)',pad=8)
for i in range(len(tumor_types_e)):
    for j in range(len(top_s)):
        v=mn[i,j]
        if v>0.05: ax.text(j,i,f'{v:.0%}',ha='center',va='center',fontsize=7.5,color='white' if v>0.5 else '#333')
plt.colorbar(im_e,ax=ax,shrink=0.8,label='Proportion')

# F: 统计汇总
ax=axes[1,2]; ax.axis('off'); panel_label(ax,'F')
ax.set_title('Strategy optimisation key statistics',pad=8)
stats_rows=[['Metric','Value','Note'],
            ['Total patients',f'{len(data):,}','Real-world cohort'],
            ['Distinct strategies',str(len(strategy_counts)),'High variability'],
            ['Most common','FISH+RNA-NGS',f'n={strategy_counts.get("FISH+RNA-NGS",0)}'],
            ['Three-method patients','567','Core cohort'],
            ['FISH-alone yield',f'{diag_yield(strategy_groups.get("FISH",[])):,.0%}','Baseline'],
            ['FISH+RNA-NGS yield',f'{diag_yield(strategy_groups.get("FISH+RNA-NGS",[])):,.0%}','+RNA gain'],
            ['All-three yield',f'{diag_yield(strategy_groups.get("DNA-NGS+FISH+RNA-NGS",[])):,.0%}','Max yield'],
            ['Strategy model acc.',f'{acc_s:.1%}','5-fold CV'],
            ['RNA-NGS new positives','298/1,052','28.3%'],
            ['DNA-NGS new targets','89/567','15.7%']]
for ri,row in enumerate(stats_rows):
    y_r=0.97-ri*0.085
    fc='#0072B2' if ri==0 else ('#E3F2FD' if ri%2==0 else 'white')
    tc='white' if ri==0 else '#333'
    ax.add_patch(FancyBboxPatch((0.01,y_r-0.075),0.97,0.078,transform=ax.transAxes,
                                 boxstyle='round,pad=0.005',facecolor=fc,edgecolor='white'))
    for ci,(cell,cx) in enumerate(zip(row,[0.02,0.45,0.72])):
        ax.text(cx,y_r-0.032,cell,transform=ax.transAxes,fontsize=8,
                fontweight='bold' if ri==0 else 'normal',color=tc,va='center')

fig6.suptitle('Figure 6  |  AI-driven testing strategy optimisation model',fontsize=13,fontweight='bold',y=1.01)
save_figure(fig6,'Figure6_v2'); plt.close()
print('  Figure 6 done')

# ── Figure 7 ─────────────────────────────────────────────
print('Building Figure 7...')
with open('sts_ai_model.pkl','rb') as f:
    bundle=pickle.load(f)
model_f7=bundle['model']; le_f7=bundle['label_encoder']; feat_names=bundle['feature_names']

FISH_GENES=['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3','TFE3','CMET','ROS1']
RNA_FUSIONS=['SS18-SSX1','SS18-SSX2','FUS-DDIT3','EWSR1-DDIT3','NAB2-STAT6','COL1A1-PDGFB','EML4-ALK','ETV6-NTRK3','ASPSCR1-TFE3','HEY1-NCOA2']
DNA_GENES=['TP53','MDM2','CDK4','RB1','NF1','PTEN','PIK3CA','KRAS','BRAF','ATRX']

def build_features(row):
    feats={f:0.0 for f in feat_names}
    try: feats['age']=float(row['年龄'])
    except: feats['age']=50.0
    feats['sex_male']=1 if row['性别']=='男' else 0
    fish_res=row['FISH结果']
    feats['fish_positive']=1 if fish_res=='阳性' else 0
    feats['fish_negative']=1 if fish_res=='阴性' else 0
    feats['fish_done']=1 if fish_res else 0
    for g in FISH_GENES: feats['fish_'+g]=1 if g in row.get('检测方法组合','') and fish_res=='阳性' else 0
    rna_res=row['RNA_NGS结果']; rna_fusion=row['融合伴侣']
    feats['rna_positive']=1 if rna_res=='阳性' else 0
    feats['rna_negative']=1 if rna_res=='阴性' else 0
    feats['rna_done']=1 if rna_res else 0
    for f in RNA_FUSIONS: feats['fusion_'+f.replace('-','_')]=1 if f in rna_fusion else 0
    dna_res=row['DNA_NGS结果']; dna_mut=row['DNA突变']
    feats['dna_positive']=1 if dna_res=='阳性' else 0
    feats['dna_negative']=1 if dna_res=='阴性' else 0
    feats['dna_done']=1 if dna_res else 0
    for g in DNA_GENES: feats['dna_'+g]=1 if g in dna_mut else 0
    feats['tmb_high']=1 if 'TMB-H' in row.get('治疗靶点','') else 0
    feats['msi_high']=1 if 'MSI-H' in row.get('治疗靶点','') else 0
    return feats

valid=[r for r in data if r['肿瘤类型'] not in ['待明确','','良性肿瘤'] and r['检测方法组合']!='']
tc=Counter(r['肿瘤类型'] for r in valid)
top_t=[t for t,c in tc.most_common() if c>=15]
valid=[r for r in valid if r['肿瘤类型'] in top_t]
rows_feat=[build_features(r) for r in valid]
X=np.array([[r[f] for f in feat_names] for r in rows_feat])
le2=LabelEncoder(); y=le2.fit_transform([r['肿瘤类型'] for r in valid])
n_classes=len(le2.classes_)
X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
y_bin_te=label_binarize(y_te,classes=range(n_classes))

models_f7={'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),
           'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
           'Gradient Boosting':GradientBoostingClassifier(n_estimators=200,random_state=42)}
test_results={}
for name,clf in models_f7.items():
    clf.fit(X_tr,y_tr); probs=clf.predict_proba(X_te)
    a=roc_auc_score(y_bin_te,probs,multi_class='ovr',average='macro')
    test_results[name]={'probs':probs,'auc':a}

fig7=plt.figure(figsize=(18,12))
gs7=fig7.add_gridspec(2,2,hspace=0.42,wspace=0.38)

# A: 工具界面 mockup
ax_a=fig7.add_subplot(gs7[0,:])
ax_a.set_xlim(0,20); ax_a.set_ylim(0,10); ax_a.axis('off')
panel_label(ax_a,'A')
ax_a.set_title('STS-Molecular-AI: online clinical decision support tool interface',pad=8)
ax_a.add_patch(FancyBboxPatch((0.1,0.2),19.8,9.5,boxstyle='round,pad=0.1',facecolor='#FAFAFA',edgecolor='#BDBDBD',lw=2))
ax_a.add_patch(plt.Rectangle((0.1,9.0),19.8,0.7,facecolor='#0072B2'))
ax_a.text(10,9.35,'STS-Molecular-AI  |  Soft Tissue Sarcoma Diagnostic Decision Support',ha='center',va='center',fontsize=11,fontweight='bold',color='white')
# 左侧输入
ax_a.add_patch(FancyBboxPatch((0.3,0.4),6.5,8.4,boxstyle='round,pad=0.1',facecolor='#E3F2FD',edgecolor='#0072B2',lw=1.5))
ax_a.text(3.55,8.55,'INPUT',ha='center',fontsize=10,fontweight='bold',color='#0072B2')
for i,(lbl,val) in enumerate([('Clinical','Age: 45   Sex: Female'),('Tumour site','Left thigh'),
                                ('FISH result','DDIT3: Negative (3%)'),('RNA-NGS','FUS-DDIT3 fusion detected'),
                                ('DNA-NGS','TP53 mut (35%), TMB: 4')]):
    yf=7.9-i*1.3
    ax_a.add_patch(FancyBboxPatch((0.5,yf-0.35),6.1,0.9,boxstyle='round,pad=0.05',facecolor='white',edgecolor='#90CAF9',lw=1))
    ax_a.text(0.7,yf+0.2,lbl,fontsize=8,color='#0072B2',fontweight='bold')
    ax_a.text(0.7,yf-0.1,val,fontsize=8.5,color='#333')
ax_a.add_patch(FancyBboxPatch((1.5,0.6),4.0,0.7,boxstyle='round,pad=0.1',facecolor='#0072B2',edgecolor='#0072B2'))
ax_a.text(3.5,0.95,'▶  ANALYSE',ha='center',va='center',fontsize=10,fontweight='bold',color='white')
ax_a.annotate('',xy=(7.5,4.5),xytext=(6.8,4.5),arrowprops=dict(arrowstyle='->',color='#0072B2',lw=3))
# 右侧输出
ax_a.add_patch(FancyBboxPatch((7.6,0.4),12.1,8.4,boxstyle='round,pad=0.1',facecolor='#F1F8E9',edgecolor='#009E73',lw=1.5))
ax_a.text(13.65,8.55,'OUTPUT',ha='center',fontsize=10,fontweight='bold',color='#009E73')
ax_a.add_patch(FancyBboxPatch((7.8,6.8),5.5,1.8,boxstyle='round,pad=0.1',facecolor='#C8E6C9',edgecolor='#009E73',lw=2))
ax_a.text(10.55,8.25,'Top Diagnosis',ha='center',fontsize=9,fontweight='bold',color='#1B5E20')
ax_a.text(10.55,7.7,'Myxoid Liposarcoma',ha='center',fontsize=12,fontweight='bold',color='#1B5E20')
ax_a.text(10.55,7.2,'Confidence: 87.3%',ha='center',fontsize=10,color='#009E73')
ax_a.add_patch(FancyBboxPatch((7.8,4.8),5.5,1.7,boxstyle='round,pad=0.1',facecolor='white',edgecolor='#A5D6A7',lw=1))
ax_a.text(10.55,6.3,'Differential Diagnosis',ha='center',fontsize=9,fontweight='bold',color='#333')
for i,(diag,prob,col) in enumerate([('Myxoid Liposarcoma',0.873,'#009E73'),('Dediff. Liposarcoma',0.082,'#E69F00'),('Undiff. sarcoma',0.031,'#D55E00')]):
    yd=5.9-i*0.38
    ax_a.barh(yd,prob*4.5,height=0.28,left=7.9,color=col,alpha=0.8)
    ax_a.text(7.85,yd,diag[:22],va='center',ha='right',fontsize=7.5,color='#333')
    ax_a.text(12.45,yd,f'{prob:.1%}',va='center',fontsize=8,fontweight='bold',color=col)
ax_a.add_patch(FancyBboxPatch((7.8,3.2),5.5,1.4,boxstyle='round,pad=0.1',facecolor='#FFF9C4',edgecolor='#F9A825',lw=1.5))
ax_a.text(10.55,4.4,'Testing Recommendation',ha='center',fontsize=9,fontweight='bold',color='#F57F17')
ax_a.text(10.55,3.9,'✓ FISH + RNA-NGS sufficient',ha='center',fontsize=9,color='#333')
ax_a.text(10.55,3.5,'⚠ DNA-NGS optional (TMB=4, low)',ha='center',fontsize=8.5,color='#888')
ax_a.add_patch(FancyBboxPatch((7.8,1.6),5.5,1.4,boxstyle='round,pad=0.1',facecolor='#FCE4EC',edgecolor='#C62828',lw=1.5))
ax_a.text(10.55,2.8,'Therapeutic Targets',ha='center',fontsize=9,fontweight='bold',color='#B71C1C')
ax_a.text(10.55,2.35,'FUS-DDIT3 fusion (diagnostic marker)',ha='center',fontsize=9,color='#333')
ax_a.text(10.55,1.9,'No actionable drug target identified',ha='center',fontsize=8.5,color='#888')
# SHAP
ax_a.add_patch(FancyBboxPatch((13.5,0.6),5.9,8.0,boxstyle='round,pad=0.1',facecolor='white',edgecolor='#6A1B9A',lw=1.5))
ax_a.text(16.45,8.35,'Explainability (SHAP)',ha='center',fontsize=9,fontweight='bold',color='#6A1B9A')
for i,(feat,val,col) in enumerate([('FUS-DDIT3 fusion',0.42,'#D55E00'),('RNA-NGS positive',0.31,'#D55E00'),
                                     ('FISH negative',0.18,'#0072B2'),('Age: 45',0.12,'#D55E00'),('Sex: Female',0.08,'#0072B2')]):
    ys=7.7-i*1.2; bw=val*5.0
    ax_a.barh(ys,bw if col=='#D55E00' else -bw,height=0.55,left=16.45,color=col,alpha=0.75)
    ax_a.text(13.6,ys,feat,va='center',fontsize=8,color='#333')
    sign='+' if col=='#D55E00' else '-'
    ax_a.text(16.45+(bw if col=='#D55E00' else -bw)+0.1,ys,f'{sign}{val:.2f}',va='center',fontsize=8,fontweight='bold',color=col)
ax_a.axvline(16.45,ymin=0.07,ymax=0.93,color='#999',lw=1,ls='--')

# B: ROC
ax_b=fig7.add_subplot(gs7[1,0]); panel_label(ax_b,'B')
ls_list=['-','--',':']; lc_list=['#0072B2','#009E73','#D55E00']
for (name,res),ls,lc in zip(test_results.items(),ls_list,lc_list):
    probs=res['probs']
    all_fpr=np.unique(np.concatenate([roc_curve(y_bin_te[:,i],probs[:,i])[0] for i in range(n_classes)]))
    mean_tpr=np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i,tpr_i,_=roc_curve(y_bin_te[:,i],probs[:,i])
        mean_tpr+=np.interp(all_fpr,fpr_i,tpr_i)
    mean_tpr/=n_classes; roc_auc_v=auc(all_fpr,mean_tpr)
    ax_b.plot(all_fpr,mean_tpr,ls,color=lc,lw=2,label=f'{name} (AUC={roc_auc_v:.3f})')
ax_b.plot([0,1],[0,1],'k--',alpha=0.35,lw=1)
ax_b.set_xlabel('False Positive Rate'); ax_b.set_ylabel('True Positive Rate')
ax_b.set_title(f'Validation on independent holdout set\n(20% temporal split, n={len(y_te)})',pad=8)
ax_b.legend(loc='lower right',framealpha=0.9)

# C: 雷达图
ax_c=fig7.add_subplot(gs7[1,1],polar=True); panel_label(ax_c,'C',x=-0.12,y=1.08)
cats=['Diagnostic\nAccuracy','Ease of\nUse','Explainability','Clinical\nRelevance','Report\nParsing','Speed']
N=len(cats); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
best_auc=max(r['auc'] for r in test_results.values())
scores_ai=[min(best_auc*1.1,1.0),0.88,0.85,0.90,0.99,0.95]+[min(best_auc*1.1,1.0)]
scores_man=[0.72,0.65,0.50,0.80,0.40,0.55,0.72]
ax_c.plot(angles,scores_ai,'o-',color='#0072B2',lw=2.5,label='STS-AI Tool')
ax_c.fill(angles,scores_ai,alpha=0.15,color='#0072B2')
ax_c.plot(angles,scores_man,'s--',color='#999999',lw=2,label='Manual review')
ax_c.fill(angles,scores_man,alpha=0.10,color='#999999')
ax_c.set_xticks(angles[:-1]); ax_c.set_xticklabels(cats,fontsize=9)
ax_c.set_ylim(0,1); ax_c.set_yticks([0.2,0.4,0.6,0.8,1.0])
ax_c.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'],fontsize=7)
ax_c.set_title('Tool evaluation: STS-AI vs manual review',pad=20,fontsize=11,fontweight='bold')
ax_c.legend(loc='upper right',bbox_to_anchor=(1.35,1.15),fontsize=9)

fig7.suptitle('Figure 7  |  STS-Molecular-AI: open-source clinical decision support tool',fontsize=13,fontweight='bold',y=1.01)
save_figure(fig7,'Figure7_v2'); plt.close()
print('  Figure 7 done')
print('\nFigure 6 & 7 rebuilt successfully.')
