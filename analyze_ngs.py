import csv
import re
from collections import defaultdict, Counter

data = []
with open('合并汇总.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print('【NGS类型细分分析】\n')

rna_ngs = []
dna_ngs = []
fish_rows = []

for row in data:
    desc = row.get('大体描述', '')
    conclusion = row.get('诊断结论', '')
    report_type = row.get('报告类型', '')

    # RNA NGS: 检测融合/易位基因
    if any(kw in desc for kw in ['肿瘤常见基因易位', 'RNA', '易位检测', '肿瘤常见易位']):
        rna_ngs.append(row)
    # DNA NGS: 检测突变/扩增
    elif any(kw in desc for kw in ['EGFR基因', 'KRAS基因', 'BRAF基因', '基因突变', 'DNA', 'TMB', 'NGS']):
        dna_ngs.append(row)
    # FISH
    elif 'FISH' in desc or '荧光染色体原位杂交' in desc:
        fish_rows.append(row)

print(f'FISH检测记录数: {len(fish_rows)}')
print(f'RNA NGS记录数: {len(rna_ngs)}')
print(f'DNA NGS记录数: {len(dna_ngs)}')

# RNA NGS 融合基因检出分析
print('\n【RNA NGS - 融合基因检出情况】')
rna_genes = ['DDIT3', 'EWSR1', 'SS18', 'ALK', 'NTRK', 'RET', 'ROS1', 'FGFR', 'NRG1', 'BCOR']
rna_positive = Counter()
rna_negative = Counter()

for row in rna_ngs:
    conclusion = row.get('诊断结论', '')
    for gene in rna_genes:
        if gene in conclusion:
            if any(kw in conclusion for kw in ['显示' + gene, gene + '基因易位', gene + '融合']):
                rna_positive[gene] += 1
            elif '未显示' + gene in conclusion:
                rna_negative[gene] += 1

print('阳性检出:')
for gene, count in rna_positive.most_common():
    neg = rna_negative.get(gene, 0)
    total = count + neg
    rate = count / total * 100 if total > 0 else 0
    print(f'  {gene}: 阳性{count} / 阴性{neg} / 阳性率{rate:.1f}%')

# DNA NGS 突变基因分析
print('\n【DNA NGS - 突变基因检出情况】')
dna_genes = ['TP53', 'MDM2', 'CDK4', 'RB1', 'ATRX', 'NF1', 'PTEN', 'PIK3CA', 'KRAS', 'BRAF', 'ALK', 'RET', 'NTRK', 'TMB']
dna_positive = Counter()

for row in dna_ngs:
    conclusion = row.get('诊断结论', '')
    for gene in dna_genes:
        if gene in conclusion:
            if '未显示' not in conclusion[:conclusion.find(gene)] or gene not in conclusion:
                if any(kw in conclusion for kw in [gene + '基因', '显示' + gene]):
                    dna_positive[gene] += 1

print('突变/扩增检出:')
for gene, count in dna_positive.most_common(10):
    print(f'  {gene}: {count}例')

# 患者级三方法分析
print('\n【患者级三方法联合分析】')
patient_methods = defaultdict(set)
for row in data:
    name = row.get('姓名', '')
    pid = row.get('病案号', '')
    if not name:
        continue
    key = (name, pid)
    desc = row.get('大体描述', '')
    if 'FISH' in desc or '荧光染色体原位杂交' in desc:
        patient_methods[key].add('FISH')
    elif any(kw in desc for kw in ['肿瘤常见基因易位', '易位检测', '肿瘤常见易位']):
        patient_methods[key].add('RNA_NGS')
    elif any(kw in desc for kw in ['EGFR基因', 'KRAS基因', 'BRAF基因', '基因突变']):
        patient_methods[key].add('DNA_NGS')

fish_only = sum(1 for m in patient_methods.values() if m == {'FISH'})
rna_only = sum(1 for m in patient_methods.values() if m == {'RNA_NGS'})
dna_only = sum(1 for m in patient_methods.values() if m == {'DNA_NGS'})
fish_rna = sum(1 for m in patient_methods.values() if 'FISH' in m and 'RNA_NGS' in m and 'DNA_NGS' not in m)
fish_dna = sum(1 for m in patient_methods.values() if 'FISH' in m and 'DNA_NGS' in m and 'RNA_NGS' not in m)
rna_dna = sum(1 for m in patient_methods.values() if 'RNA_NGS' in m and 'DNA_NGS' in m and 'FISH' not in m)
all_three = sum(1 for m in patient_methods.values() if 'FISH' in m and 'RNA_NGS' in m and 'DNA_NGS' in m)

print(f'仅FISH: {fish_only}例')
print(f'仅RNA NGS: {rna_only}例')
print(f'仅DNA NGS: {dna_only}例')
print(f'FISH + RNA NGS: {fish_rna}例')
print(f'FISH + DNA NGS: {fish_dna}例')
print(f'RNA NGS + DNA NGS: {rna_dna}例')
print(f'三方法全做: {all_three}例')
print(f'总患者数: {len(patient_methods)}例')
