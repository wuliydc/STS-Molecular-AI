import csv
import random
from collections import defaultdict

random.seed(42)

data = []
with open('合并汇总.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# 按检测方法分层抽样，保证各类型都有代表
fish_rows = []
rna_rows = []
dna_rows = []
consult_rows = []

for row in data:
    desc = row.get('大体描述', '')
    report_type = row.get('报告类型', '')
    conclusion = row.get('诊断结论', '').strip()
    if not conclusion:
        continue
    if 'FISH' in desc or '荧光染色体原位杂交' in desc:
        fish_rows.append(row)
    elif any(kw in desc for kw in ['肿瘤常见基因易位', '易位检测', '肿瘤常见易位']):
        rna_rows.append(row)
    elif any(kw in desc for kw in ['EGFR基因', 'KRAS基因', 'BRAF基因', '基因突变']):
        dna_rows.append(row)
    elif report_type in ['会诊', '特需会诊']:
        consult_rows.append(row)

# 分层抽样：FISH 70例，RNA NGS 50例，DNA NGS 40例，会诊 40例
sample_fish = random.sample(fish_rows, min(70, len(fish_rows)))
sample_rna = random.sample(rna_rows, min(50, len(rna_rows)))
sample_dna = random.sample(dna_rows, min(40, len(dna_rows)))
sample_consult = random.sample(consult_rows, min(40, len(consult_rows)))

all_samples = sample_fish + sample_rna + sample_dna + sample_consult
random.shuffle(all_samples)

print(f'FISH样本: {len(sample_fish)}例')
print(f'RNA NGS样本: {len(sample_rna)}例')
print(f'DNA NGS样本: {len(sample_dna)}例')
print(f'会诊样本: {len(sample_consult)}例')
print(f'总计: {len(all_samples)}例')

# 输出标注工作表
fieldnames = [
    '序号', '病理号', '报告类型', '大体描述_原文', '诊断结论_原文',
    # 以下为标注字段（留空）
    '【标注】肿瘤类型',
    '【标注】肿瘤良恶性',
    '【标注】检测方法',
    '【标注】检测基因',
    '【标注】检测结果_阳性阴性',
    '【标注】融合伴侣基因',
    '【标注】突变类型',
    '【标注】治疗靶点',
    '【标注】诊断是否明确',
    '【标注】备注_疑难点'
]

with open('标注工作表_200例.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, row in enumerate(all_samples, 1):
        writer.writerow({
            '序号': i,
            '病理号': row.get('病理号', ''),
            '报告类型': row.get('报告类型', ''),
            '大体描述_原文': row.get('大体描述', ''),
            '诊断结论_原文': row.get('诊断结论', ''),
            '【标注】肿瘤类型': '',
            '【标注】肿瘤良恶性': '',
            '【标注】检测方法': '',
            '【标注】检测基因': '',
            '【标注】检测结果_阳性阴性': '',
            '【标注】融合伴侣基因': '',
            '【标注】突变类型': '',
            '【标注】治疗靶点': '',
            '【标注】诊断是否明确': '',
            '【标注】备注_疑难点': ''
        })

print('\n已生成: 标注工作表_200例.csv')
