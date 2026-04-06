"""
Figure 1: Landscape of multi-modal molecular testing in a real-world STS cohort
A: 队列纳入流程图数据
B: 2018-2025年各方法使用趋势（堆叠面积图）
C: 肿瘤亚型 × 检测方法气泡矩阵
D: 三方法重叠Venn图
"""
import csv
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

# 字体设置（支持中文）
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 读取结构化数据
data = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

df = pd.DataFrame(data)

# ── 患者级数据 ──────────────────────────────────────────────
patient_methods = defaultdict(set)
patient_info = {}
for row in data:
    name, pid = row['姓名'], row['病案号']
    if not name:
        continue
    key = (name, pid)
    m = row['检测方法']
    if m != '会诊':
        patient_methods[key].add(m)
    if key not in patient_info:
        patient_info[key] = {'age': row['年龄'], 'sex': row['性别'], 'time': row['登记时间']}

total_patients = len(patient_methods)
fish_set  = {k for k, v in patient_methods.items() if 'FISH' in v}
rna_set   = {k for k, v in patient_methods.items() if 'RNA-NGS' in v}
dna_set   = {k for k, v in patient_methods.items() if 'DNA-NGS' in v}
all_three = fish_set & rna_set & dna_set
fish_rna  = (fish_set & rna_set) - dna_set
fish_dna  = (fish_set & dna_set) - rna_set
rna_dna   = (rna_set & dna_set) - fish_set
fish_only = fish_set - rna_set - dna_set
rna_only  = rna_set - fish_set - dna_set
dna_only  = dna_set - fish_set - rna_set

print('=== Figure 1 数据汇总 ===')
print(f'\n【队列基本信息】')
print(f'总记录数: {len(data)}')
print(f'总患者数（有分子检测）: {total_patients}')
print(f'FISH患者: {len(fish_set)}')
print(f'RNA-NGS患者: {len(rna_set)}')
print(f'DNA-NGS患者: {len(dna_set)}')
print(f'三方法全做: {len(all_three)}')
print(f'FISH+RNA-NGS: {len(fish_rna)}')
print(f'FISH+DNA-NGS: {len(fish_dna)}')
print(f'RNA+DNA-NGS: {len(rna_dna)}')
print(f'仅FISH: {len(fish_only)}')
print(f'仅RNA-NGS: {len(rna_only)}')
print(f'仅DNA-NGS: {len(dna_only)}')

# ── 年龄性别统计 ──────────────────────────────────────────
ages, sexes = [], []
for v in patient_info.values():
    try:
        ages.append(int(v['age']))
    except:
        pass
    if v['sex'] in ['男', '女']:
        sexes.append(v['sex'])

sex_count = Counter(sexes)
print(f'\n【人口学特征】')
print(f'年龄中位数: {np.median(ages):.0f}岁 (IQR {np.percentile(ages,25):.0f}-{np.percentile(ages,75):.0f})')
print(f'性别: 男{sex_count["男"]}例, 女{sex_count["女"]}例')

# ── 时间趋势 ──────────────────────────────────────────────
year_method = defaultdict(lambda: defaultdict(int))
for row in data:
    t = row['登记时间']
    m = row['检测方法']
    if not t or m == '会诊':
        continue
    try:
        year = int(str(t)[:4])
        if 2018 <= year <= 2025:
            year_method[year][m] += 1
    except:
        pass

years = sorted(year_method.keys())
print(f'\n【年度检测量】')
for y in years:
    total = sum(year_method[y].values())
    print(f'  {y}: {total}例 (FISH:{year_method[y]["FISH"]}, RNA:{year_method[y]["RNA-NGS"]}, DNA:{year_method[y]["DNA-NGS"]})')

# ── 肿瘤亚型 × 检测方法 ──────────────────────────────────
top_tumors = [k for k, v in Counter(
    r['肿瘤类型'] for r in data if r['肿瘤类型'] not in ['待明确', '']
).most_common(10)]

tumor_method_matrix = defaultdict(lambda: defaultdict(int))
for row in data:
    t = row['肿瘤类型']
    m = row['检测方法']
    if t in top_tumors and m != '会诊':
        tumor_method_matrix[t][m] += 1

print(f'\n【肿瘤亚型×检测方法矩阵（Top10）】')
methods_list = ['FISH', 'RNA-NGS', 'DNA-NGS']
header = f"{'肿瘤类型':<20}" + ''.join(f'{m:>12}' for m in methods_list)
print(header)
for tumor in top_tumors:
    row_str = f'{tumor:<20}' + ''.join(f'{tumor_method_matrix[tumor][m]:>12}' for m in methods_list)
    print(row_str)

# ════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
colors = {'FISH': '#2196F3', 'RNA-NGS': '#4CAF50', 'DNA-NGS': '#FF9800'}

# ── Panel B: 年度趋势堆叠面积图 ──────────────────────────
ax_b = fig.add_subplot(2, 2, 1)
fish_vals  = [year_method[y]['FISH']    for y in years]
rna_vals   = [year_method[y]['RNA-NGS'] for y in years]
dna_vals   = [year_method[y]['DNA-NGS'] for y in years]

ax_b.stackplot(years, fish_vals, rna_vals, dna_vals,
               labels=['FISH', 'RNA-NGS', 'DNA-NGS'],
               colors=[colors['FISH'], colors['RNA-NGS'], colors['DNA-NGS']],
               alpha=0.85)
ax_b.set_xlabel('Year', fontsize=12)
ax_b.set_ylabel('Number of tests', fontsize=12)
ax_b.set_title('B  Annual testing volume by method (2018–2025)', fontsize=13, fontweight='bold', loc='left')
ax_b.legend(loc='upper left', fontsize=10)
ax_b.set_xticks(years)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# ── Panel C: 气泡矩阵 ────────────────────────────────────
ax_c = fig.add_subplot(2, 2, 2)
tumor_labels = [t[:12] for t in top_tumors]
method_labels = ['FISH', 'RNA-NGS', 'DNA-NGS']
matrix = np.array([[tumor_method_matrix[t][m] for m in method_labels] for t in top_tumors])

for i, tumor in enumerate(top_tumors):
    for j, method in enumerate(method_labels):
        val = matrix[i, j]
        if val > 0:
            size = np.sqrt(val) * 15
            ax_c.scatter(j, i, s=size, c=list(colors.values())[j], alpha=0.7, zorder=3)
            ax_c.text(j, i, str(val), ha='center', va='center', fontsize=8, fontweight='bold', color='white')

ax_c.set_xticks(range(3))
ax_c.set_xticklabels(method_labels, fontsize=11)
ax_c.set_yticks(range(len(top_tumors)))
ax_c.set_yticklabels(tumor_labels, fontsize=9)
ax_c.set_title('C  Testing method distribution by tumour subtype', fontsize=13, fontweight='bold', loc='left')
ax_c.grid(True, alpha=0.3)
ax_c.set_xlim(-0.5, 2.5)
ax_c.set_ylim(-0.5, len(top_tumors) - 0.5)

# ── Panel D: Venn图（手动绘制） ──────────────────────────
ax_d = fig.add_subplot(2, 2, 3)
ax_d.set_xlim(0, 10)
ax_d.set_ylim(0, 10)
ax_d.set_aspect('equal')
ax_d.axis('off')

# 三个圆
circles = [
    plt.Circle((3.5, 6.0), 2.8, color=colors['FISH'],    alpha=0.35, zorder=2),
    plt.Circle((6.5, 6.0), 2.8, color=colors['RNA-NGS'], alpha=0.35, zorder=2),
    plt.Circle((5.0, 3.5), 2.8, color=colors['DNA-NGS'], alpha=0.35, zorder=2),
]
for c in circles:
    ax_d.add_patch(c)

# 标签
ax_d.text(1.8, 7.8, f'FISH\n(n={len(fish_set)})',      ha='center', fontsize=10, fontweight='bold', color='#1565C0')
ax_d.text(8.2, 7.8, f'RNA-NGS\n(n={len(rna_set)})',    ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
ax_d.text(5.0, 1.0, f'DNA-NGS\n(n={len(dna_set)})',    ha='center', fontsize=10, fontweight='bold', color='#E65100')

# 交集数字
ax_d.text(5.0, 6.8, str(len(fish_rna)),  ha='center', fontsize=11, fontweight='bold', color='#333')
ax_d.text(3.2, 4.5, str(len(fish_dna)),  ha='center', fontsize=11, fontweight='bold', color='#333')
ax_d.text(6.8, 4.5, str(len(rna_dna)),   ha='center', fontsize=11, fontweight='bold', color='#333')
ax_d.text(5.0, 5.5, str(len(all_three)), ha='center', fontsize=14, fontweight='bold', color='#B71C1C',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B71C1C', linewidth=1.5))

ax_d.text(2.0, 8.8, str(len(fish_only)), ha='center', fontsize=10, color='#555')
ax_d.text(8.0, 8.8, str(len(rna_only)),  ha='center', fontsize=10, color='#555')
ax_d.text(5.0, 0.2, str(len(dna_only)),  ha='center', fontsize=10, color='#555')

ax_d.set_title('D  Patient overlap across three testing modalities', fontsize=13, fontweight='bold', loc='left')

# ── Panel A: 队列流程图（文字版） ────────────────────────
ax_a = fig.add_subplot(2, 2, 4)
ax_a.axis('off')

flow_text = (
    f"COHORT OVERVIEW\n"
    f"{'─'*38}\n"
    f"Total records:          {len(data):>6,}\n"
    f"Total patients:         {total_patients:>6,}\n\n"
    f"TESTING BREAKDOWN\n"
    f"{'─'*38}\n"
    f"FISH tests:             {len(fish_set):>6,}\n"
    f"RNA-NGS tests:          {len(rna_set):>6,}\n"
    f"DNA-NGS tests:          {len(dna_set):>6,}\n\n"
    f"MULTI-MODAL PATIENTS\n"
    f"{'─'*38}\n"
    f"FISH + RNA-NGS only:    {len(fish_rna):>6,}\n"
    f"FISH + DNA-NGS only:    {len(fish_dna):>6,}\n"
    f"RNA + DNA-NGS only:     {len(rna_dna):>6,}\n"
    f"All three methods:      {len(all_three):>6,}\n\n"
    f"DEMOGRAPHICS\n"
    f"{'─'*38}\n"
    f"Median age:             {np.median(ages):.0f} yrs\n"
    f"Male / Female:          {sex_count['男']} / {sex_count['女']}\n"
    f"Study period:           2018–2025\n"
)
ax_a.text(0.05, 0.95, flow_text, transform=ax_a.transAxes,
          fontsize=10, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
ax_a.set_title('A  Cohort summary', fontsize=13, fontweight='bold', loc='left')

plt.suptitle('Figure 1 | Landscape of multi-modal molecular testing in a real-world\nsoft tissue sarcoma cohort (n=1,489 patients)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('Figure1_Cohort_Landscape.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nFigure 1 已保存: Figure1_Cohort_Landscape.png')
