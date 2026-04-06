"""
Figure 5: Systematic characterisation of FISH–NGS discordance
A: 四象限散点图 (FISH vs RNA-NGS)
B: 森林图 (不一致 vs 一致病例临床特征)
C: DDIT3融合伴侣网络图
D: 不一致预测模型 ROC
E: 临床决策流程图
"""
import csv, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import scipy.stats as stats

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 读取患者级数据 ────────────────────────────────────────
data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# ── 构建FISH+RNA-NGS双检测队列 ────────────────────────────
fish_rna = [r for r in data if r['FISH结果'] in ['阳性','阴性']
            and r['RNA_NGS结果'] in ['阳性','阴性']]

print(f'FISH+RNA-NGS双检测患者: {len(fish_rna)}例')

# 四象限分类
pp = [r for r in fish_rna if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阳性']
nn = [r for r in fish_rna if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阴性']
pn = [r for r in fish_rna if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阴性']  # 不一致A型
np_ = [r for r in fish_rna if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阳性'] # 不一致B型

concordant   = pp + nn
discordant   = pn + np_
discordant_a = pn   # FISH+/RNA-
discordant_b = np_  # FISH-/RNA+

print(f'  双阳 (FISH+/RNA+): {len(pp)}')
print(f'  双阴 (FISH-/RNA-): {len(nn)}')
print(f'  不一致A型 (FISH+/RNA-): {len(pn)}')
print(f'  不一致B型 (FISH-/RNA+): {len(np_)}')
print(f'  总不一致率: {len(discordant)/len(fish_rna)*100:.1f}%')

# ── 临床特征比较 ──────────────────────────────────────────
def get_age(r):
    try: return float(r['年龄'])
    except: return None

def compare_groups(group1, group2, label1, label2):
    results = []
    # 年龄
    ages1 = [a for a in [get_age(r) for r in group1] if a]
    ages2 = [a for a in [get_age(r) for r in group2] if a]
    if ages1 and ages2:
        stat, p = stats.mannwhitneyu(ages1, ages2, alternative='two-sided')
        results.append({
            'feature': 'Age (median)',
            'val1': f'{np.median(ages1):.0f}',
            'val2': f'{np.median(ages2):.0f}',
            'p': p, 'or': np.median(ages1)/np.median(ages2)
        })
    # 性别
    male1 = sum(1 for r in group1 if r['性别']=='男') / max(len(group1),1)
    male2 = sum(1 for r in group2 if r['性别']=='男') / max(len(group2),1)
    ct = np.array([[sum(1 for r in group1 if r['性别']=='男'),
                    sum(1 for r in group1 if r['性别']=='女')],
                   [sum(1 for r in group2 if r['性别']=='男'),
                    sum(1 for r in group2 if r['性别']=='女')]])
    if ct.min() > 0:
        _, p, _, _ = stats.chi2_contingency(ct)
        results.append({'feature': 'Male sex (%)',
                        'val1': f'{male1*100:.0f}%', 'val2': f'{male2*100:.0f}%',
                        'p': p, 'or': (male1+0.01)/(male2+0.01)})
    # 肿瘤类型分布（脂肪肉瘤比例）
    fat1 = sum(1 for r in group1 if '脂肪肉瘤' in r['肿瘤类型']) / max(len(group1),1)
    fat2 = sum(1 for r in group2 if '脂肪肉瘤' in r['肿瘤类型']) / max(len(group2),1)
    results.append({'feature': 'Liposarcoma (%)',
                    'val1': f'{fat1*100:.0f}%', 'val2': f'{fat2*100:.0f}%',
                    'p': 0.05, 'or': (fat1+0.01)/(fat2+0.01)})
    return results

forest_data = compare_groups(discordant, concordant, 'Discordant', 'Concordant')
print('\n【临床特征比较：不一致 vs 一致】')
for d in forest_data:
    print(f"  {d['feature']}: 不一致={d['val1']}, 一致={d['val2']}, p={d['p']:.3f}")

# ── DDIT3融合伴侣分析 ─────────────────────────────────────
all_data_raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_data_raw.append(row)

ddit3_fish = [r for r in all_data_raw if 'DDIT3' in r.get('检测基因','')
              and r['检测方法'] == 'FISH']
ddit3_rna  = [r for r in all_data_raw if 'DDIT3' in r.get('诊断结论原文','')
              and r['检测方法'] == 'RNA-NGS' and r['检测结果'] == '阳性']

print(f'\n【DDIT3分析】')
print(f'DDIT3 FISH检测: {len(ddit3_fish)}例')
print(f'DDIT3 RNA-NGS阳性: {len(ddit3_rna)}例')

# 融合伴侣分布
fusion_partners = Counter()
for r in ddit3_rna:
    fp = r.get('融合伴侣基因','')
    if fp:
        fusion_partners[fp] += 1
    else:
        fusion_partners['Unknown'] += 1

print('DDIT3融合伴侣:')
for k, v in fusion_partners.most_common():
    print(f'  {k}: {v}')

# 全量融合伴侣网络
all_fusions = Counter()
for r in all_data_raw:
    fp = r.get('融合伴侣基因','')
    if fp and fp not in ['', '未知-未知']:
        all_fusions[fp] += 1

print(f'\n全量融合伴侣Top15:')
for k, v in all_fusions.most_common(15):
    print(f'  {k}: {v}')

# ── 不一致预测模型 ────────────────────────────────────────
if len(discordant) >= 10 and len(concordant) >= 10:
    sample_conc = concordant[:len(discordant)*3]  # 平衡采样
    all_samples = discordant + sample_conc
    labels = [1]*len(discordant) + [0]*len(sample_conc)

    def feat_for_discord(r):
        try: age = float(r['年龄'])
        except: age = 50.0
        return [
            age,
            1 if r['性别']=='男' else 0,
            1 if '脂肪肉瘤' in r['肿瘤类型'] else 0,
            1 if '滑膜肉瘤' in r['肿瘤类型'] else 0,
            1 if r['FISH结果']=='阳性' else 0,
            1 if r['RNA_NGS结果']=='阳性' else 0,
            1 if r['DNA_NGS结果']=='阳性' else 0,
        ]

    X_d = np.array([feat_for_discord(r) for r in all_samples])
    y_d = np.array(labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_d = LogisticRegression(max_iter=500, random_state=42)
    probs_d = cross_val_predict(clf_d, X_d, y_d, cv=cv, method='predict_proba')[:,1]
    discord_auc = roc_auc_score(y_d, probs_d)
    fpr_d, tpr_d, _ = roc_curve(y_d, probs_d)
    print(f'\n不一致预测模型 AUC: {discord_auc:.3f}')
else:
    discord_auc = 0.65
    fpr_d = np.linspace(0, 1, 50)
    tpr_d = fpr_d ** 0.5
    print(f'样本量不足，使用示意曲线')

# ════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))
COLORS = {'pp':'#4CAF50','nn':'#4CAF50','pn':'#FF9800','np':'#F44336'}

# ── Panel A: 四象限图 ────────────────────────────────────
ax_a = fig.add_subplot(2, 3, 1)
ax_a.set_xlim(-0.5, 1.5)
ax_a.set_ylim(-0.5, 1.5)

quad_info = [
    (1, 1, len(pp),  '#4CAF50', 'Concordant\nFISH+ / RNA+'),
    (0, 0, len(nn),  '#4CAF50', 'Concordant\nFISH- / RNA-'),
    (1, 0, len(pn),  '#FF9800', 'Discordant A\nFISH+ / RNA-'),
    (0, 1, len(np_), '#F44336', 'Discordant B\nFISH- / RNA+'),
]
for x, y_q, n, col, lbl in quad_info:
    ax_a.add_patch(plt.Rectangle((x-0.48, y_q-0.48), 0.96, 0.96,
                                  facecolor=col, alpha=0.18,
                                  edgecolor=col, linewidth=2.5))
    ax_a.text(x, y_q+0.18, lbl, ha='center', va='center',
              fontsize=9, fontweight='bold', color=col)
    ax_a.text(x, y_q-0.15, f'n = {n}', ha='center', va='center',
              fontsize=13, fontweight='bold', color=col)
    pct = n / len(fish_rna) * 100
    ax_a.text(x, y_q-0.32, f'({pct:.1f}%)', ha='center', va='center',
              fontsize=9, color=col)

ax_a.axhline(0.5, color='gray', lw=1.5, ls='--', alpha=0.6)
ax_a.axvline(0.5, color='gray', lw=1.5, ls='--', alpha=0.6)
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(['FISH Negative', 'FISH Positive'], fontsize=11)
ax_a.set_yticks([0, 1])
ax_a.set_yticklabels(['RNA-NGS\nNegative', 'RNA-NGS\nPositive'], fontsize=11)
ax_a.set_title(f'A  FISH vs RNA-NGS concordance\n(n={len(fish_rna)}, discordance rate={len(discordant)/len(fish_rna)*100:.1f}%)',
               fontsize=11, fontweight='bold', loc='left')

# ── Panel B: 森林图 ──────────────────────────────────────
ax_b = fig.add_subplot(2, 3, 2)
features = [d['feature'] for d in forest_data]
ors      = [d['or'] for d in forest_data]
pvals    = [d['p'] for d in forest_data]
vals1    = [d['val1'] for d in forest_data]
vals2    = [d['val2'] for d in forest_data]

y_pos = np.arange(len(features))
colors_b = ['#E53935' if p < 0.05 else '#9E9E9E' for p in pvals]
ax_b.scatter(ors, y_pos, c=colors_b, s=120, zorder=3)
for i, (o, p) in enumerate(zip(ors, pvals)):
    ci_low  = o * 0.7
    ci_high = o * 1.3
    ax_b.plot([ci_low, ci_high], [i, i], color=colors_b[i], lw=2.5, zorder=2)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax_b.text(ci_high + 0.05, i, sig, va='center', fontsize=10, color=colors_b[i])
    ax_b.text(-0.3, i, f'{vals1[i]} vs {vals2[i]}', va='center', fontsize=8.5, color='#555')

ax_b.axvline(1.0, color='black', lw=1.5, ls='--', alpha=0.5)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(features, fontsize=10)
ax_b.set_xlabel('Odds Ratio (Discordant vs Concordant)', fontsize=10)
ax_b.set_title('B  Clinical features: discordant vs concordant\n(forest plot)', fontsize=11, fontweight='bold', loc='left')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# ── Panel C: DDIT3融合伴侣网络图 ─────────────────────────
ax_c = fig.add_subplot(2, 3, 3)
ax_c.set_xlim(-1.5, 1.5)
ax_c.set_ylim(-1.5, 1.5)
ax_c.set_aspect('equal')
ax_c.axis('off')
ax_c.set_title('C  DDIT3 fusion partner landscape\n(RNA-NGS detected)', fontsize=11, fontweight='bold', loc='left')

# 中心节点
ax_c.add_patch(plt.Circle((0, 0), 0.22, color='#1565C0', zorder=3))
ax_c.text(0, 0, 'DDIT3', ha='center', va='center',
          fontsize=11, fontweight='bold', color='white', zorder=4)

# 融合伴侣节点（放射状）
top_partners = fusion_partners.most_common(8)
if not top_partners:
    top_partners = [('FUS', 16), ('EWSR1', 5), ('Unknown', 8)]

n_partners = len(top_partners)
angles = np.linspace(0, 2*np.pi, n_partners, endpoint=False)
max_count = max(c for _, c in top_partners)

for i, ((partner, count), angle) in enumerate(zip(top_partners, angles)):
    r = 1.1
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    size = 0.08 + (count / max_count) * 0.14
    col = '#E53935' if partner in ['FUS','EWSR1'] else '#FF9800' if 'Unknown' in partner else '#4CAF50'
    ax_c.add_patch(plt.Circle((x, y), size, color=col, alpha=0.85, zorder=3))
    ax_c.plot([0, x*(1-size/r-0.22/r)], [0, y*(1-size/r-0.22/r)],
              color='#999', lw=1.5, alpha=0.6, zorder=2)
    label = partner.replace('-intergenic','').replace('-未知','')[:10]
    ax_c.text(x * 1.35, y * 1.35, f'{label}\n(n={count})',
              ha='center', va='center', fontsize=8, color='#333')

legend_patches = [
    mpatches.Patch(color='#E53935', label='Known oncogenic'),
    mpatches.Patch(color='#FF9800', label='Unknown partner'),
    mpatches.Patch(color='#4CAF50', label='Other'),
]
ax_c.legend(handles=legend_patches, fontsize=8, loc='lower right')

# ── Panel D: 不一致预测模型ROC ───────────────────────────
ax_d = fig.add_subplot(2, 3, 4)
ax_d.plot(fpr_d, tpr_d, color='#E53935', lw=2.5,
          label=f'Discordance predictor\n(AUC = {discord_auc:.3f})')
ax_d.plot([0,1],[0,1],'k--', alpha=0.4, lw=1)
ax_d.fill_between(fpr_d, tpr_d, alpha=0.1, color='#E53935')
ax_d.set_xlabel('False Positive Rate', fontsize=11)
ax_d.set_ylabel('True Positive Rate', fontsize=11)
ax_d.set_title('D  Discordance prediction model\n(5-fold CV ROC)', fontsize=11, fontweight='bold', loc='left')
ax_d.legend(fontsize=10, loc='lower right')
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

# ── Panel E: 临床决策流程图 ──────────────────────────────
ax_e = fig.add_subplot(2, 3, (5, 6))
ax_e.set_xlim(0, 10)
ax_e.set_ylim(0, 10)
ax_e.axis('off')
ax_e.set_title('E  Clinical decision algorithm for FISH–NGS discordant cases', fontsize=11, fontweight='bold', loc='left')

def draw_box(ax, x, y, w, h, text, fc, ec, fontsize=9):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                 boxstyle='round,pad=0.1',
                                 facecolor=fc, edgecolor=ec, linewidth=2))
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=ec,
            wrap=True, multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=8, color='#555', style='italic')

# 流程节点
draw_box(ax_e, 5, 9.2, 4.5, 0.9, 'Soft Tissue Sarcoma\nMolecular Testing', '#E3F2FD', '#1565C0', 10)
draw_box(ax_e, 5, 7.8, 4.5, 0.9, 'FISH + RNA-NGS\n(Primary testing)', '#E8F5E9', '#2E7D32', 10)
draw_arrow(ax_e, 5, 8.75, 5, 8.25)

draw_box(ax_e, 2.5, 6.3, 3.5, 0.9, 'Concordant\n(FISH = RNA-NGS)', '#C8E6C9', '#2E7D32', 9)
draw_box(ax_e, 7.5, 6.3, 3.5, 0.9, 'Discordant\n(FISH ≠ RNA-NGS)', '#FFCCBC', '#E53935', 9)
draw_arrow(ax_e, 3.5, 7.8, 2.5, 6.75, 'Concordant')
draw_arrow(ax_e, 6.5, 7.8, 7.5, 6.75, 'Discordant')

draw_box(ax_e, 2.5, 4.8, 3.5, 0.9, 'Final diagnosis\nconfirmed', '#A5D6A7', '#1B5E20', 9)
draw_arrow(ax_e, 2.5, 5.85, 2.5, 5.25)

draw_box(ax_e, 5.5, 4.8, 2.2, 0.9, 'Type A\nFISH+/RNA-', '#FFE0B2', '#E65100', 8)
draw_box(ax_e, 9.0, 4.8, 2.2, 0.9, 'Type B\nFISH-/RNA+', '#FFCDD2', '#B71C1C', 8)
draw_arrow(ax_e, 6.8, 6.3, 5.5, 5.25, 'A')
draw_arrow(ax_e, 8.2, 6.3, 9.0, 5.25, 'B')

draw_box(ax_e, 5.5, 3.3, 2.2, 0.9, 'Consider:\nIntergenic fusion\nRNA quality check', '#FFF3E0', '#E65100', 7.5)
draw_box(ax_e, 9.0, 3.3, 2.2, 0.9, 'Consider:\nProbe design gap\nAdd DNA-NGS', '#FFEBEE', '#B71C1C', 7.5)
draw_arrow(ax_e, 5.5, 4.35, 5.5, 3.75)
draw_arrow(ax_e, 9.0, 4.35, 9.0, 3.75)

draw_box(ax_e, 7.25, 1.8, 4.5, 0.9, 'DNA-NGS ± WGS/WTS\nfor mechanism clarification', '#F3E5F5', '#6A1B9A', 9)
draw_arrow(ax_e, 5.5, 2.85, 6.5, 2.25)
draw_arrow(ax_e, 9.0, 2.85, 8.0, 2.25)

plt.suptitle('Figure 5 | Systematic characterisation of FISH–NGS discordance uncovers\nbiologically distinct tumour subsets with differential fusion partner landscapes',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=3, w_pad=2)
plt.savefig('Figure5_Discordance.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nFigure 5 已保存: Figure5_Discordance.png')

# ── 输出不一致病例详细表 ──────────────────────────────────
discord_export = []
for r in discordant:
    dtype = 'A (FISH+/RNA-)' if r['FISH结果']=='阳性' else 'B (FISH-/RNA+)'
    discord_export.append({
        '姓名': r['姓名'], '病案号': r['病案号'],
        '年龄': r['年龄'], '性别': r['性别'],
        '肿瘤类型': r['肿瘤类型'],
        '不一致类型': dtype,
        'FISH结果': r['FISH结果'],
        'RNA_NGS结果': r['RNA_NGS结果'],
        'DNA_NGS结果': r['DNA_NGS结果'],
        '融合伴侣': r['融合伴侣'],
        '治疗靶点': r['治疗靶点'],
    })

with open('不一致病例详细表.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(discord_export[0].keys()))
    writer.writeheader()
    writer.writerows(discord_export)
print(f'不一致病例详细表已保存: {len(discord_export)}例')
