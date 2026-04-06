"""Figure 1 v2 — 300 dpi, RGB, 统一样式"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
from plot_style import apply_style, save_figure, panel_label, METHOD_COLORS

apply_style()

# ── 数据 ──────────────────────────────────────────────────
data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        data.append(row)

raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        raw.append(row)

patient_methods = defaultdict(set)
patient_info    = {}
for row in data:
    name, pid = row['姓名'], row['病案号']
    if not name: continue
    key = (name, pid)
    m = row['检测方法组合']
    for method in ['FISH','RNA-NGS','DNA-NGS']:
        if method in m:
            patient_methods[key].add(method)
    if key not in patient_info:
        patient_info[key] = {'age': row['年龄'], 'sex': row['性别']}

fish_set  = {k for k,v in patient_methods.items() if 'FISH'    in v}
rna_set   = {k for k,v in patient_methods.items() if 'RNA-NGS' in v}
dna_set   = {k for k,v in patient_methods.items() if 'DNA-NGS' in v}
all_three = fish_set & rna_set & dna_set
fish_rna  = (fish_set & rna_set) - dna_set
fish_dna  = (fish_set & dna_set) - rna_set
rna_dna   = (rna_set  & dna_set) - fish_set
fish_only = fish_set - rna_set - dna_set
rna_only  = rna_set  - fish_set - dna_set
dna_only  = dna_set  - fish_set - rna_set

ages = [float(v['age']) for v in patient_info.values() if v['age'].replace('.','').isdigit()]
sexes = Counter(v['sex'] for v in patient_info.values() if v['sex'] in ['男','女'])

year_method = defaultdict(lambda: defaultdict(int))
for row in raw:
    t = row.get('登记时间',''); m = row.get('检测方法','')
    if t and m != '会诊':
        try:
            yr = int(str(t)[:4])
            if 2018 <= yr <= 2025: year_method[yr][m] += 1
        except: pass
years = sorted(year_method.keys())

tumor_counts = Counter(r['肿瘤类型'] for r in data if r['肿瘤类型'] not in ['待明确',''])
top_tumors   = [t for t,_ in tumor_counts.most_common(10)]
tumor_method_matrix = defaultdict(lambda: defaultdict(int))
for r in data:
    t = r['肿瘤类型']
    if t in top_tumors:
        for m in ['FISH','RNA-NGS','DNA-NGS']:
            if m in r['检测方法组合']:
                tumor_method_matrix[t][m] += 1

# ── 绘图 ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38)

# A: 队列汇总文字框
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axis('off')
panel_label(ax_a, 'A')
summary = (
    f"Cohort overview\n"
    f"{'─'*32}\n"
    f"Total records:       {len(raw):>6,}\n"
    f"Total patients:      {len(patient_methods):>6,}\n\n"
    f"Testing modalities\n"
    f"{'─'*32}\n"
    f"FISH:                {len(fish_set):>6,}\n"
    f"RNA-NGS:             {len(rna_set):>6,}\n"
    f"DNA-NGS:             {len(dna_set):>6,}\n\n"
    f"Multi-modal patients\n"
    f"{'─'*32}\n"
    f"FISH + RNA-NGS:      {len(fish_rna):>6,}\n"
    f"FISH + DNA-NGS:      {len(fish_dna):>6,}\n"
    f"RNA + DNA-NGS:       {len(rna_dna):>6,}\n"
    f"All three methods:   {len(all_three):>6,}\n\n"
    f"Demographics\n"
    f"{'─'*32}\n"
    f"Median age:          {np.median(ages):.0f} yrs\n"
    f"Male / Female:       {sexes['男']} / {sexes['女']}\n"
    f"Study period:        2018–2025\n"
)
ax_a.text(0.05, 0.97, summary, transform=ax_a.transAxes,
          fontsize=9, va='top', fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA',
                    edgecolor='#DEE2E6', linewidth=1.2))

# B: 年度趋势堆叠面积图
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')
fish_v = [year_method[y]['FISH']    for y in years]
rna_v  = [year_method[y]['RNA-NGS'] for y in years]
dna_v  = [year_method[y]['DNA-NGS'] for y in years]
ax_b.stackplot(years, fish_v, rna_v, dna_v,
               labels=['FISH','RNA-NGS','DNA-NGS'],
               colors=[METHOD_COLORS['FISH'], METHOD_COLORS['RNA-NGS'], METHOD_COLORS['DNA-NGS']],
               alpha=0.88)
ax_b.set_xlabel('Year')
ax_b.set_ylabel('Number of tests')
ax_b.set_title('Annual testing volume by modality (2018–2025)', pad=8)
ax_b.legend(loc='upper left', framealpha=0.9)
ax_b.set_xticks(years)
ax_b.set_xticklabels(years, rotation=30)

# C: 气泡矩阵
ax_c = fig.add_subplot(gs[0, 2])
panel_label(ax_c, 'C')
methods_list = ['FISH','RNA-NGS','DNA-NGS']
for i, tumor in enumerate(top_tumors):
    for j, method in enumerate(methods_list):
        val = tumor_method_matrix[tumor][method]
        if val > 0:
            size = np.sqrt(val) * 18
            ax_c.scatter(j, i, s=size, c=METHOD_COLORS[method], alpha=0.75, zorder=3)
            ax_c.text(j, i, str(val), ha='center', va='center',
                      fontsize=7.5, fontweight='bold', color='white', zorder=4)
ax_c.set_xticks(range(3)); ax_c.set_xticklabels(methods_list)
ax_c.set_yticks(range(len(top_tumors)))
ax_c.set_yticklabels([t[:14] for t in top_tumors], fontsize=8)
ax_c.set_title('Testing method by tumour subtype', pad=8)
ax_c.grid(True, alpha=0.2, zorder=0)
ax_c.set_xlim(-0.6, 2.6); ax_c.set_ylim(-0.6, len(top_tumors)-0.4)

# D: Venn图
ax_d = fig.add_subplot(gs[1, 0])
ax_d.set_xlim(0, 10); ax_d.set_ylim(0, 10)
ax_d.set_aspect('equal'); ax_d.axis('off')
panel_label(ax_d, 'D', x=-0.05)
circles = [
    plt.Circle((3.5, 6.2), 2.7, color=METHOD_COLORS['FISH'],    alpha=0.30, zorder=2),
    plt.Circle((6.5, 6.2), 2.7, color=METHOD_COLORS['RNA-NGS'], alpha=0.30, zorder=2),
    plt.Circle((5.0, 3.8), 2.7, color=METHOD_COLORS['DNA-NGS'], alpha=0.30, zorder=2),
]
for c in circles: ax_d.add_patch(c)
# 边框圆
for (cx,cy), col in zip([(3.5,6.2),(6.5,6.2),(5.0,3.8)],
                         [METHOD_COLORS['FISH'],METHOD_COLORS['RNA-NGS'],METHOD_COLORS['DNA-NGS']]):
    ax_d.add_patch(plt.Circle((cx,cy), 2.7, fill=False, edgecolor=col, lw=1.8, zorder=3))

ax_d.text(1.5, 8.5, f'FISH\n(n={len(fish_set)})',   ha='center', fontsize=10, fontweight='bold', color=METHOD_COLORS['FISH'])
ax_d.text(8.5, 8.5, f'RNA-NGS\n(n={len(rna_set)})', ha='center', fontsize=10, fontweight='bold', color=METHOD_COLORS['RNA-NGS'])
ax_d.text(5.0, 0.8, f'DNA-NGS\n(n={len(dna_set)})', ha='center', fontsize=10, fontweight='bold', color=METHOD_COLORS['DNA-NGS'])
ax_d.text(5.0, 7.0, str(len(fish_rna)),  ha='center', fontsize=11, fontweight='bold', color='#333')
ax_d.text(3.2, 4.7, str(len(fish_dna)),  ha='center', fontsize=11, fontweight='bold', color='#333')
ax_d.text(6.8, 4.7, str(len(rna_dna)),   ha='center', fontsize=11, fontweight='bold', color='#333')
ax_d.text(5.0, 5.7, str(len(all_three)), ha='center', fontsize=14, fontweight='bold', color='#B71C1C',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#B71C1C', lw=1.5))
ax_d.text(1.8, 9.3, str(len(fish_only)), ha='center', fontsize=9, color='#555')
ax_d.text(8.2, 9.3, str(len(rna_only)),  ha='center', fontsize=9, color='#555')
ax_d.text(5.0, 0.1, str(len(dna_only)),  ha='center', fontsize=9, color='#555')
ax_d.set_title('Patient overlap across three modalities', pad=8)

# E: 年龄分布直方图
ax_e = fig.add_subplot(gs[1, 1])
panel_label(ax_e, 'E')
ax_e.hist(ages, bins=20, color=METHOD_COLORS['FISH'], alpha=0.82, edgecolor='white', linewidth=0.5)
ax_e.axvline(np.median(ages), color=METHOD_COLORS['red'] if 'red' in METHOD_COLORS else '#D55E00',
             lw=2, ls='--', label=f'Median = {np.median(ages):.0f} yrs')
ax_e.set_xlabel('Age (years)'); ax_e.set_ylabel('Number of patients')
ax_e.set_title('Age distribution', pad=8)
ax_e.legend()

# F: 肿瘤亚型分布
ax_f = fig.add_subplot(gs[1, 2])
panel_label(ax_f, 'F')
top12 = tumor_counts.most_common(12)
colors_f = plt.cm.tab20(np.linspace(0, 1, len(top12)))
ax_f.barh([t[:16] for t,_ in top12[::-1]], [v for _,v in top12[::-1]],
          color=colors_f, edgecolor='white', linewidth=0.5)
ax_f.set_xlabel('Number of patients')
ax_f.set_title('Tumour subtype distribution (Top 12)', pad=8)

fig.suptitle('Figure 1  |  Landscape of multi-modal molecular testing in a real-world\nsoft tissue sarcoma cohort',
             fontsize=13, fontweight='bold', y=1.01)
save_figure(fig, 'Figure1_v2')
plt.close()
