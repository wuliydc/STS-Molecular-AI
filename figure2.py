"""
Figure 2: Stepwise diagnostic gain of sequential molecular testing
+ 修复肿瘤亚型×检测方法矩阵（患者级关联）
"""
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict, Counter

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 读取数据 ──────────────────────────────────────────────
data = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# ── 患者级数据构建 ────────────────────────────────────────
# 同一患者的所有记录合并，会诊记录提供肿瘤类型，分子检测提供结果
patient_records = defaultdict(lambda: {
    'tumor_type': '待明确',
    'methods': set(),
    'fish_result': None,
    'rna_result': None,
    'dna_result': None,
    'fish_gene': '',
    'rna_fusion': '',
    'dna_mutation': '',
    'targets': set(),
    'age': '',
    'sex': '',
})

for row in data:
    name, pid = row['姓名'], row['病案号']
    if not name:
        continue
    key = (name, pid)
    p = patient_records[key]

    # 人口学
    if not p['age'] and row['年龄']:
        p['age'] = row['年龄']
    if not p['sex'] and row['性别']:
        p['sex'] = row['性别']

    m = row['检测方法']
    result = row['检测结果']
    tumor = row['肿瘤类型']

    # 肿瘤类型：优先取会诊的明确诊断
    if tumor not in ['待明确', ''] and p['tumor_type'] == '待明确':
        p['tumor_type'] = tumor

    if m == 'FISH':
        p['methods'].add('FISH')
        if result == '阳性':
            p['fish_result'] = '阳性'
            p['fish_gene'] = row['检测基因']
        elif result == '阴性' and p['fish_result'] is None:
            p['fish_result'] = '阴性'

    elif m == 'RNA-NGS':
        p['methods'].add('RNA-NGS')
        if result == '阳性':
            p['rna_result'] = '阳性'
            p['rna_fusion'] = row['融合伴侣基因']
        elif result == '阴性' and p['rna_result'] is None:
            p['rna_result'] = '阴性'

    elif m == 'DNA-NGS':
        p['methods'].add('DNA-NGS')
        if result == '阳性':
            p['dna_result'] = '阳性'
            p['dna_mutation'] = row['突变类型']
        elif result == '阴性' and p['dna_result'] is None:
            p['dna_result'] = '阴性'

    # 治疗靶点
    t = row['治疗靶点']
    if t and t != '无':
        for tgt in t.split('/'):
            p['targets'].add(tgt)

# ── 统计分析 ──────────────────────────────────────────────
all_patients = list(patient_records.values())
three_method = [p for p in all_patients if {'FISH','RNA-NGS','DNA-NGS'}.issubset(p['methods'])]
fish_rna     = [p for p in all_patients if 'FISH' in p['methods'] and 'RNA-NGS' in p['methods']]

print('=== Figure 2 数据分析 ===')
print(f'\n三方法全做患者: {len(three_method)}例')
print(f'FISH+RNA-NGS患者: {len(fish_rna)}例')

# ── 诊断增量分析 ──────────────────────────────────────────
# Step1: FISH单独阳性率
fish_pos = sum(1 for p in all_patients if p['fish_result'] == '阳性')
fish_total = sum(1 for p in all_patients if p['fish_result'] is not None)

# Step2: FISH阴性但RNA-NGS阳性（RNA-NGS新增诊断）
fish_neg_rna_pos = sum(1 for p in fish_rna
                       if p['fish_result'] == '阴性' and p['rna_result'] == '阳性')
fish_pos_rna_pos = sum(1 for p in fish_rna
                       if p['fish_result'] == '阳性' and p['rna_result'] == '阳性')
fish_pos_rna_neg = sum(1 for p in fish_rna
                       if p['fish_result'] == '阳性' and p['rna_result'] == '阴性')
fish_neg_rna_neg = sum(1 for p in fish_rna
                       if p['fish_result'] == '阴性' and p['rna_result'] == '阴性')

# Step3: 在FISH+RNA基础上，DNA-NGS新增治疗靶点
dna_new_targets = sum(1 for p in three_method
                      if p['dna_result'] == '阳性' and len(p['targets']) > 0)

print(f'\n【诊断增量分析】')
print(f'FISH阳性率: {fish_pos}/{fish_total} = {fish_pos/fish_total*100:.1f}%')
print(f'\nFISH+RNA-NGS一致性（{len(fish_rna)}例）:')
print(f'  双阳 (FISH+/RNA+): {fish_pos_rna_pos}例')
print(f'  双阴 (FISH-/RNA-): {fish_neg_rna_neg}例')
print(f'  FISH-/RNA+ (RNA新增): {fish_neg_rna_pos}例 ({fish_neg_rna_pos/len(fish_rna)*100:.1f}%)')
print(f'  FISH+/RNA- (不一致): {fish_pos_rna_neg}例 ({fish_pos_rna_neg/len(fish_rna)*100:.1f}%)')

print(f'\nDNA-NGS新增治疗靶点（三方法{len(three_method)}例中）: {dna_new_targets}例')

# ── 肿瘤亚型×检测方法矩阵（患者级） ─────────────────────
top_tumors_raw = Counter(p['tumor_type'] for p in all_patients
                         if p['tumor_type'] not in ['待明确', '']).most_common(10)
top_tumors = [t for t, _ in top_tumors_raw]

print(f'\n【肿瘤亚型分布（患者级）Top10】')
for t, c in top_tumors_raw:
    print(f'  {t}: {c}例')

tumor_method_matrix = defaultdict(lambda: defaultdict(int))
for p in all_patients:
    t = p['tumor_type']
    if t in top_tumors:
        for m in p['methods']:
            tumor_method_matrix[t][m] += 1

print(f'\n【肿瘤亚型×检测方法矩阵（患者级）】')
methods_list = ['FISH', 'RNA-NGS', 'DNA-NGS']
print(f"{'肿瘤类型':<22}" + ''.join(f'{m:>12}' for m in methods_list))
for tumor in top_tumors:
    row_str = f'{tumor:<22}' + ''.join(f'{tumor_method_matrix[tumor][m]:>12}' for m in methods_list)
    print(row_str)

# ── 治疗靶点统计 ──────────────────────────────────────────
all_targets = []
for p in all_patients:
    all_targets.extend(p['targets'])
target_counts = Counter(all_targets)
print(f'\n【治疗靶点分布（患者级）】')
for t, c in target_counts.most_common():
    print(f'  {t}: {c}例')

# ════════════════════════════════════════════════════════════
# 绘图 Figure 2
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
colors = {'FISH': '#2196F3', 'RNA-NGS': '#4CAF50', 'DNA-NGS': '#FF9800'}

# ── Panel A: 诊断增量瀑布图 ──────────────────────────────
ax_a = axes[0, 0]
steps = ['FISH alone', '+RNA-NGS\n(new positives)', '+DNA-NGS\n(new targets)']
values = [fish_pos, fish_neg_rna_pos, dna_new_targets]
bar_colors = [colors['FISH'], colors['RNA-NGS'], colors['DNA-NGS']]
bars = ax_a.bar(steps, values, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
              f'n={val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_a.set_ylabel('Number of patients', fontsize=12)
ax_a.set_title('A  Stepwise diagnostic gain by adding each modality', fontsize=12, fontweight='bold', loc='left')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.set_ylim(0, max(values) * 1.2)

# ── Panel B: FISH vs RNA-NGS 一致性四象限 ────────────────
ax_b = axes[0, 1]
quadrant_data = {
    'FISH+/RNA+': fish_pos_rna_pos,
    'FISH-/RNA-': fish_neg_rna_neg,
    'FISH-/RNA+': fish_neg_rna_pos,
    'FISH+/RNA-': fish_pos_rna_neg,
}
quad_colors = ['#4CAF50', '#4CAF50', '#FF5722', '#FF9800']
quad_labels = list(quadrant_data.keys())
quad_values = list(quadrant_data.values())

positions = [(1, 1), (0, 0), (1, 0), (0, 1)]
ax_b.set_xlim(-0.5, 2.5)
ax_b.set_ylim(-0.5, 2.5)
for (x, y), label, val, col in zip(positions, quad_labels, quad_values, quad_colors):
    ax_b.add_patch(plt.Rectangle((x-0.45, y-0.45), 0.9, 0.9,
                                  facecolor=col, alpha=0.3, edgecolor=col, linewidth=2))
    ax_b.text(x, y+0.15, label, ha='center', va='center', fontsize=10, fontweight='bold')
    ax_b.text(x, y-0.15, f'n={val}', ha='center', va='center', fontsize=12, color=col)

ax_b.set_xticks([0, 1])
ax_b.set_xticklabels(['FISH Negative', 'FISH Positive'], fontsize=11)
ax_b.set_yticks([0, 1])
ax_b.set_yticklabels(['RNA-NGS\nNegative', 'RNA-NGS\nPositive'], fontsize=11)
ax_b.set_title(f'B  FISH vs RNA-NGS concordance (n={len(fish_rna)})', fontsize=12, fontweight='bold', loc='left')
ax_b.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax_b.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

# ── Panel C: 肿瘤亚型×检测方法气泡矩阵 ──────────────────
ax_c = axes[1, 0]
matrix = np.array([[tumor_method_matrix[t][m] for m in methods_list] for t in top_tumors])
tumor_labels_short = [t[:14] for t in top_tumors]

for i, tumor in enumerate(top_tumors):
    for j, method in enumerate(methods_list):
        val = matrix[i, j]
        if val > 0:
            size = np.sqrt(val) * 20
            ax_c.scatter(j, i, s=size, c=list(colors.values())[j], alpha=0.75, zorder=3)
            ax_c.text(j, i, str(val), ha='center', va='center', fontsize=8,
                      fontweight='bold', color='white')

ax_c.set_xticks(range(3))
ax_c.set_xticklabels(methods_list, fontsize=11)
ax_c.set_yticks(range(len(top_tumors)))
ax_c.set_yticklabels(tumor_labels_short, fontsize=9)
ax_c.set_title('C  Testing method by tumour subtype (patient-level)', fontsize=12, fontweight='bold', loc='left')
ax_c.grid(True, alpha=0.25)
ax_c.set_xlim(-0.5, 2.5)
ax_c.set_ylim(-0.5, len(top_tumors) - 0.5)

# ── Panel D: 治疗靶点堆叠柱状图 ──────────────────────────
ax_d = axes[1, 1]
target_items = target_counts.most_common(8)
t_labels = [t for t, _ in target_items]
t_values = [v for _, v in target_items]
t_colors = plt.cm.Set2(np.linspace(0, 1, len(t_labels)))
bars_d = ax_d.barh(t_labels, t_values, color=t_colors, edgecolor='white', linewidth=1)
for bar, val in zip(bars_d, t_values):
    ax_d.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
              f'n={val}', va='center', fontsize=10)
ax_d.set_xlabel('Number of patients', fontsize=12)
ax_d.set_title('D  Actionable therapeutic targets identified', fontsize=12, fontweight='bold', loc='left')
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.set_xlim(0, max(t_values) * 1.25)

plt.suptitle('Figure 2 | Stepwise diagnostic gain of sequential molecular testing\nin soft tissue sarcoma',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('Figure2_Diagnostic_Gain.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nFigure 2 已保存: Figure2_Diagnostic_Gain.png')

# 保存患者级结构化数据
patient_export = []
for (name, pid), p in patient_records.items():
    patient_export.append({
        '姓名': name, '病案号': pid,
        '年龄': p['age'], '性别': p['sex'],
        '肿瘤类型': p['tumor_type'],
        '检测方法组合': '+'.join(sorted(p['methods'])),
        'FISH结果': p['fish_result'] or '',
        'RNA_NGS结果': p['rna_result'] or '',
        'DNA_NGS结果': p['dna_result'] or '',
        '融合伴侣': p['rna_fusion'],
        'DNA突变': p['dna_mutation'],
        '治疗靶点': '/'.join(p['targets']) if p['targets'] else '无',
    })

with open('患者级结构化数据集.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(patient_export[0].keys()))
    writer.writeheader()
    writer.writerows(patient_export)

print(f'患者级数据集已保存: 患者级结构化数据集.csv ({len(patient_export)}例)')
