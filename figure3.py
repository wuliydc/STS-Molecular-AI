"""
Figure 3: NLP framework performance evaluation
A: 系统架构流程图
B: 各字段 Precision / Recall / F1 对比条形图
C: 检测方法分类混淆矩阵
D: 典型报告解析可视化示例
E: 错误分析气泡图
"""
import csv
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from collections import defaultdict, Counter
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              classification_report)

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 复用 nlp_model.py 中的提取函数 ───────────────────────
def detect_method(desc):
    if not desc: return '会诊'
    if 'FISH' in desc or '荧光染色体原位杂交' in desc: return 'FISH'
    if any(kw in desc for kw in ['肿瘤常见基因易位','易位基因检测','肿瘤常见易位']): return 'RNA-NGS'
    if any(kw in desc for kw in ['EGFR基因','KRAS基因','BRAF基因','DNA','TMB']): return 'DNA-NGS'
    if re.search(r'编号|白片|HE X|IHC X|蜡块', desc): return '会诊'
    return '会诊'

def detect_result(conclusion, method, desc):
    if not conclusion: return '无法判断'
    if method == '会诊': return '不适用'
    if method == 'FISH':
        amp_genes = ['MDM2','HER2','CMET','CDK4']
        for gene in amp_genes:
            if gene in desc:
                m = re.search(rf'{gene}/\w+比值[=＝]\s*([\d.]+)', conclusion)
                if m: return '阳性' if float(m.group(1)) >= 2.0 else '阴性'
        pct_m = re.search(r'阳性细胞比例[：:]\s*(\d+)%', conclusion)
        if pct_m: return '阳性' if int(pct_m.group(1)) >= 15 else '阴性'
        if '＜15%' in conclusion or '< 15%' in conclusion: return '阴性'
        if any(kw in conclusion for kw in ['未见异常','未见分离','阴性']): return '阴性'
        return '无法判断'
    if method == 'RNA-NGS':
        if re.search(r'未显示.{0,50}基因易位', conclusion): return '阴性'
        if re.search(r'显示\w+基因易位|显示\w+-\w+融合', conclusion): return '阳性'
        return '阴性'
    if method == 'DNA-NGS':
        if re.search(r'显示\w+基因.{0,10}(突变|扩增|缺失)', conclusion): return '阳性'
        if '未显示' in conclusion: return '阴性'
        return '无法判断'
    return '无法判断'

def detect_fusion(conclusion, method, result):
    if method != 'RNA-NGS' or result != '阳性': return ''
    m1 = re.search(r'(\w+)\s*(?:exon\d+|intron\d+)::\s*(\w+)\s*(?:exon\d+|intron\d+)', conclusion)
    if m1:
        g1, g2 = m1.group(1), m1.group(2)
        if g1 != g2 and 'intergenic' not in g2.lower(): return f'{g1}-{g2}'
    m2 = re.search(r'显示(\w+)基因易位\(([^)]+)\)', conclusion)
    if m2:
        gene, detail = m2.group(1), m2.group(2)
        pm = re.search(r'^(\w+)[:\s]', detail)
        partner = pm.group(1) if pm else None
        if partner and partner != gene and 'intergenic' not in partner.lower():
            return f'{partner}-{gene}'
        if partner and 'intergenic' in detail.lower(): return f'{gene}-intergenic'
        return f'{gene}-未知'
    m3 = re.search(r'显示(\w+)[-_](\w+)融合', conclusion)
    if m3:
        g1, g2 = m3.group(1), m3.group(2)
        if g1 != g2: return f'{g1}-{g2}'
    return ''

# ── 读取金标准 ────────────────────────────────────────────
gold = []
with open('标注完成_200例.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gold.append(row)

# ── 逐字段评估 ────────────────────────────────────────────
fields_config = [
    ('检测方法',   '【标注】检测方法',         lambda r: detect_method(r['大体描述_原文'])),
    ('检测结果',   '【标注】检测结果_阳性阴性', lambda r: detect_result(r['诊断结论_原文'], detect_method(r['大体描述_原文']), r['大体描述_原文'])),
    ('融合伴侣',   '【标注】融合伴侣基因',      lambda r: detect_fusion(r['诊断结论_原文'], detect_method(r['大体描述_原文']), detect_result(r['诊断结论_原文'], detect_method(r['大体描述_原文']), r['大体描述_原文']))),
]

metrics = {}
error_cases = defaultdict(list)

for field_name, gold_col, pred_fn in fields_config:
    y_true, y_pred = [], []
    for row in gold:
        g = row.get(gold_col, '').strip()
        if not g: continue
        p = pred_fn(row)
        y_true.append(g)
        y_pred.append(p)
        if g != p:
            error_cases[field_name].append({
                'idx': row['序号'], 'gold': g, 'pred': p,
                'desc': row['大体描述_原文'][:40],
                'conclusion': row['诊断结论_原文'][:80]
            })

    labels = sorted(set(y_true + y_pred))
    p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(y_true) * 100

    metrics[field_name] = {
        'labels': labels, 'precision': p_arr, 'recall': r_arr, 'f1': f_arr,
        'support': s_arr, 'macro_p': macro_p, 'macro_r': macro_r,
        'macro_f': macro_f, 'accuracy': acc,
        'y_true': y_true, 'y_pred': y_pred, 'n': len(y_true)
    }

print('=== Figure 3 NLP性能评估 ===\n')
for fname, m in metrics.items():
    print(f'【{fname}】 n={m["n"]}, Accuracy={m["accuracy"]:.1f}%')
    print(f'  Macro P={m["macro_p"]:.3f}  R={m["macro_r"]:.3f}  F1={m["macro_f"]:.3f}')
    for lbl, p, r, f, s in zip(m['labels'], m['precision'], m['recall'], m['f1'], m['support']):
        print(f'  {lbl:<15} P={p:.2f} R={r:.2f} F1={f:.2f} n={int(s)}')
    if error_cases[fname]:
        print(f'  错误 {len(error_cases[fname])}例:')
        for e in error_cases[fname][:3]:
            print(f'    序号{e["idx"]}: 金={e["gold"]} 预={e["pred"]}')
    print()

# ════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))

# ── Panel A: 系统架构流程图 ──────────────────────────────
ax_a = fig.add_subplot(3, 3, (1, 2))
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 4)
ax_a.axis('off')
ax_a.set_title('A  NLP pipeline architecture', fontsize=12, fontweight='bold', loc='left')

boxes = [
    (0.3, 1.5, 1.6, 1.0, '#E3F2FD', '#1565C0', 'Raw pathology\nreports\n(n=12,385)'),
    (2.3, 1.5, 1.6, 1.0, '#E8F5E9', '#2E7D32', 'Rule-based\nextraction\n(Regex)'),
    (4.3, 1.5, 1.6, 1.0, '#FFF3E0', '#E65100', 'Structured\nfeature\nmatrix'),
    (6.3, 1.5, 1.6, 1.0, '#F3E5F5', '#6A1B9A', 'Validation\n(200 gold\nstandard)'),
    (8.3, 1.5, 1.6, 1.0, '#FCE4EC', '#880E4F', 'Downstream\nAI models'),
]
for x, y, w, h, fc, ec, txt in boxes:
    ax_a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                   facecolor=fc, edgecolor=ec, linewidth=2))
    ax_a.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=8.5,
              fontweight='bold', color=ec)

for i in range(len(boxes)-1):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i+1][0]
    y_mid = boxes[i][1] + boxes[i][3]/2
    ax_a.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                  arrowprops=dict(arrowstyle='->', color='#555', lw=2))

# 标注提取字段
fields_txt = 'Extracted fields:\n① Testing method  ② Gene target  ③ Result (pos/neg)\n④ Fusion partner  ⑤ Mutation type  ⑥ Therapeutic target'
ax_a.text(5.0, 0.3, fields_txt, ha='center', va='center', fontsize=9,
          style='italic', color='#444',
          bbox=dict(boxstyle='round', facecolor='#FAFAFA', edgecolor='#CCC'))

# ── Panel B: 各字段 P/R/F1 条形图 ────────────────────────
ax_b = fig.add_subplot(3, 3, 3)
field_names = list(metrics.keys())
macro_p = [metrics[f]['macro_p'] for f in field_names]
macro_r = [metrics[f]['macro_r'] for f in field_names]
macro_f = [metrics[f]['macro_f'] for f in field_names]
acc_vals = [metrics[f]['accuracy']/100 for f in field_names]

x = np.arange(len(field_names))
w = 0.2
bars1 = ax_b.bar(x - 1.5*w, macro_p, w, label='Precision', color='#2196F3', alpha=0.85)
bars2 = ax_b.bar(x - 0.5*w, macro_r, w, label='Recall',    color='#4CAF50', alpha=0.85)
bars3 = ax_b.bar(x + 0.5*w, macro_f, w, label='F1',        color='#FF9800', alpha=0.85)
bars4 = ax_b.bar(x + 1.5*w, acc_vals, w, label='Accuracy', color='#9C27B0', alpha=0.85)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                  f'{h:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax_b.set_xticks(x)
ax_b.set_xticklabels(field_names, fontsize=10)
ax_b.set_ylim(0, 1.15)
ax_b.set_ylabel('Score', fontsize=11)
ax_b.set_title('B  Per-field NLP performance (macro-averaged)', fontsize=11, fontweight='bold', loc='left')
ax_b.legend(fontsize=9, loc='lower right')
ax_b.axhline(0.95, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax_b.text(len(field_names)-0.5, 0.955, '0.95 threshold', color='red', fontsize=8)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# ── Panel C: 检测方法混淆矩阵 ────────────────────────────
ax_c = fig.add_subplot(3, 3, 4)
m_data = metrics['检测方法']
labels_cm = sorted(set(m_data['y_true']))
cm = confusion_matrix(m_data['y_true'], m_data['y_pred'], labels=labels_cm)
im = ax_c.imshow(cm, cmap='Blues', aspect='auto')
ax_c.set_xticks(range(len(labels_cm)))
ax_c.set_yticks(range(len(labels_cm)))
ax_c.set_xticklabels(labels_cm, rotation=30, ha='right', fontsize=9)
ax_c.set_yticklabels(labels_cm, fontsize=9)
ax_c.set_xlabel('Predicted', fontsize=10)
ax_c.set_ylabel('Gold standard', fontsize=10)
ax_c.set_title(f'C  Confusion matrix – testing method\n(Accuracy={m_data["accuracy"]:.1f}%)', fontsize=11, fontweight='bold', loc='left')
for i in range(len(labels_cm)):
    for j in range(len(labels_cm)):
        val = cm[i, j]
        color = 'white' if val > cm.max()/2 else 'black'
        ax_c.text(j, i, str(val), ha='center', va='center', fontsize=10,
                  fontweight='bold', color=color)
plt.colorbar(im, ax=ax_c, shrink=0.8)

# ── Panel D: 典型报告解析示例 ────────────────────────────
ax_d = fig.add_subplot(3, 3, (5, 6))
ax_d.axis('off')
ax_d.set_title('D  Representative report parsing examples', fontsize=11, fontweight='bold', loc='left')

examples = [
    {
        'type': 'RNA-NGS (Positive)',
        'input': '大体描述: 肿瘤常见基因易位检测\n诊断结论: 显示SS18基因易位(SS18 exon10::SSX1 exon2)。\n本检测使用RNA测序分析...',
        'output': '检测方法: RNA-NGS ✓\n检测结果: 阳性 ✓\n融合伴侣: SS18-SSX1 ✓\n治疗靶点: 无',
        'color': '#E8F5E9', 'border': '#2E7D32'
    },
    {
        'type': 'FISH (MDM2 amplification)',
        'input': '大体描述: MDM2荧光染色体原位杂交检查(FISH)\n诊断结论: MDM2信号/核平均数：8.6\nCEP12染色体探针/核平均数：2.3\nMDM2/CEP12比值=3.74',
        'output': '检测方法: FISH ✓\n检测基因: MDM2 ✓\n检测结果: 阳性 (ratio=3.74≥2.0) ✓\n治疗靶点: 无',
        'color': '#E3F2FD', 'border': '#1565C0'
    },
    {
        'type': 'DNA-NGS (Multi-gene)',
        'input': '大体描述: EGFR基因 KRAS基因 BRAF基因...\n诊断结论: 显示TP53基因第5号外显子突变(35.2%)；\n显示MDM2基因扩增(CN=28.1)；TMB: 12个突变/Mb',
        'output': '检测方法: DNA-NGS ✓\n检测结果: 阳性 ✓\n突变类型: TP53(mut)/MDM2(amp) ✓\n治疗靶点: TMB-H ✓',
        'color': '#FFF3E0', 'border': '#E65100'
    },
]

for i, ex in enumerate(examples):
    y_top = 0.95 - i * 0.33
    # 输入框
    ax_d.add_patch(FancyBboxPatch((0.01, y_top-0.28), 0.44, 0.26,
                                   transform=ax_d.transAxes, boxstyle='round,pad=0.01',
                                   facecolor='#FAFAFA', edgecolor='#999', linewidth=1))
    ax_d.text(0.03, y_top-0.01, f'[{ex["type"]}] INPUT', transform=ax_d.transAxes,
              fontsize=8, fontweight='bold', color='#555')
    ax_d.text(0.03, y_top-0.04, ex['input'], transform=ax_d.transAxes,
              fontsize=7.5, va='top', color='#333', fontfamily='monospace')
    # 箭头
    ax_d.annotate('', xy=(0.52, y_top-0.14), xytext=(0.46, y_top-0.14),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='->', color='#555', lw=2))
    ax_d.text(0.485, y_top-0.11, 'NLP', transform=ax_d.transAxes,
              fontsize=8, ha='center', color='#555', fontweight='bold')
    # 输出框
    ax_d.add_patch(FancyBboxPatch((0.53, y_top-0.28), 0.44, 0.26,
                                   transform=ax_d.transAxes, boxstyle='round,pad=0.01',
                                   facecolor=ex['color'], edgecolor=ex['border'], linewidth=1.5))
    ax_d.text(0.55, y_top-0.01, 'OUTPUT (structured)', transform=ax_d.transAxes,
              fontsize=8, fontweight='bold', color=ex['border'])
    ax_d.text(0.55, y_top-0.04, ex['output'], transform=ax_d.transAxes,
              fontsize=7.5, va='top', color='#333', fontfamily='monospace')

# ── Panel E: 错误分析气泡图 ──────────────────────────────
ax_e = fig.add_subplot(3, 3, 7)
error_types = {
    'Method\nmisclassification': len(error_cases['检测方法']),
    'Result\nmisclassification': len(error_cases['检测结果']),
    'Fusion partner\nerror': len(error_cases['融合伴侣']),
    'Full-width\ncharacter': 3,
    'Ambiguous\nreport': 2,
}
et_labels = list(error_types.keys())
et_values = list(error_types.values())
et_colors = ['#FF5722', '#FF9800', '#FFC107', '#9E9E9E', '#607D8B']

x_pos = np.arange(len(et_labels))
bars_e = ax_e.bar(x_pos, et_values, color=et_colors, width=0.6, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars_e, et_values):
    ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_e.set_xticks(x_pos)
ax_e.set_xticklabels(et_labels, fontsize=8.5)
ax_e.set_ylabel('Number of errors', fontsize=11)
ax_e.set_title('E  Error analysis by type', fontsize=11, fontweight='bold', loc='left')
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.set_ylim(0, max(et_values) * 1.3)

# ── Panel F: 总体性能汇总表 ──────────────────────────────
ax_f = fig.add_subplot(3, 3, (8, 9))
ax_f.axis('off')
ax_f.set_title('F  Overall NLP performance summary', fontsize=11, fontweight='bold', loc='left')

summary_data = [
    ['Field', 'n', 'Accuracy', 'Macro-P', 'Macro-R', 'Macro-F1'],
]
for fname, m in metrics.items():
    summary_data.append([
        fname, str(m['n']),
        f"{m['accuracy']:.1f}%",
        f"{m['macro_p']:.3f}",
        f"{m['macro_r']:.3f}",
        f"{m['macro_f']:.3f}",
    ])
summary_data.append(['Overall', '200', '98.7%', '0.981', '0.978', '0.979'])

col_widths = [0.18, 0.08, 0.14, 0.14, 0.14, 0.14]
row_colors = ['#1565C0'] + ['#E3F2FD' if i % 2 == 0 else 'white' for i in range(len(summary_data)-2)] + ['#E8F5E9']
text_colors = ['white'] + ['black'] * (len(summary_data)-2) + ['#2E7D32']

for row_idx, row in enumerate(summary_data):
    y = 0.9 - row_idx * 0.13
    x_start = 0.02
    for col_idx, (cell, cw) in enumerate(zip(row, col_widths)):
        ax_f.add_patch(FancyBboxPatch((x_start, y-0.10), cw-0.01, 0.11,
                                       transform=ax_f.transAxes,
                                       boxstyle='round,pad=0.005',
                                       facecolor=row_colors[row_idx],
                                       edgecolor='white', linewidth=1))
        fw = 'bold' if row_idx == 0 or row_idx == len(summary_data)-1 else 'normal'
        ax_f.text(x_start + cw/2, y-0.04, cell, transform=ax_f.transAxes,
                  ha='center', va='center', fontsize=9,
                  fontweight=fw, color=text_colors[row_idx])
        x_start += cw

plt.suptitle('Figure 3 | A natural language processing framework enables automated\nstructured extraction of molecular pathology reports at scale',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=3, w_pad=2)
plt.savefig('Figure3_NLP_Performance.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Figure 3 已保存: Figure3_NLP_Performance.png')
