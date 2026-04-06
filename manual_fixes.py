import csv

# 人工修正字典：序号 -> {字段: 值}
# 基于规则无法提取、需要病理专业判断的病例
manual_fixes = {
    '2':  {'【标注】肿瘤类型': '低度恶性肌纤维母细胞肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断'},
    '3':  {'【标注】肿瘤类型': '低度恶性纤维黏液样肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断'},
    '9':  {'【标注】肿瘤类型': '黏液样脂肪肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断'},
    '10': {'【标注】肿瘤类型': '梭形细胞软组织肿瘤', '【标注】肿瘤良恶性': '中间型', '【标注】诊断是否明确': '待明确'},
    '16': {'【标注】肿瘤类型': '低度恶性纤维黏液样肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断'},
    '21': {'【标注】肿瘤类型': 'IgG4相关硬化性疾病', '【标注】肿瘤良恶性': '良性', '【标注】诊断是否明确': '倾向性诊断'},
    '22': {'【标注】肿瘤类型': '待明确', '【标注】肿瘤良恶性': '待明确', '【标注】诊断是否明确': '待明确'},
    '27': {'【标注】肿瘤类型': '平滑肌肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '明确'},
    '37': {'【标注】肿瘤类型': '上皮样血管内皮瘤', '【标注】肿瘤良恶性': '中间型', '【标注】诊断是否明确': '明确'},
    '41': {'【标注】肿瘤类型': '黏液样脂肪肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断'},
    '42': {'【标注】肿瘤类型': '恶性血管球瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '明确'},
    '43': {'【标注】肿瘤类型': '高分化脂肪肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '明确'},
    '48': {'【标注】肿瘤类型': '未分化肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断'},
    '50': {'【标注】肿瘤类型': '去分化脂肪肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断'},
    '52': {'【标注】检测结果_阳性阴性': '阴性'},  # "＜15%"无法被正则捕获
    '69': {'【标注】肿瘤类型': '待明确', '【标注】肿瘤良恶性': '待明确', '【标注】诊断是否明确': '待明确'},
    '70': {'【标注】肿瘤类型': '肌上皮癌', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '明确',
           '【标注】备注_疑难点': '涎腺来源，非软组织肉瘤，可考虑排除'},
    '80': {'【标注】肿瘤类型': '脂肪肉瘤', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '待明确'},
    '86': {'【标注】检测方法': 'RNA-NGS', '【标注】检测结果_阳性阴性': '阳性',
           '【标注】融合伴侣基因': 'ETV6-NTRK3', '【标注】治疗靶点': 'NTRK融合',
           '【标注】备注_疑难点': '大体描述为DNA-NGS panel但实际报告为RNA-NGS结果'},
    '90': {'【标注】肿瘤类型': '肾细胞样腺癌', '【标注】肿瘤良恶性': '恶性', '【标注】诊断是否明确': '倾向性诊断',
           '【标注】备注_疑难点': '非软组织肉瘤，涎腺/鼻腔来源，可考虑排除'},
}

# 读取已自动标注的文件
data = []
with open('标注完成_200例.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        data.append(row)

# 应用人工修正
fixed_count = 0
for row in data:
    idx = row['序号']
    if idx in manual_fixes:
        for field, value in manual_fixes[idx].items():
            row[field] = value
        fixed_count += 1

# 保存
with open('标注完成_200例.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f'人工修正完成: {fixed_count}例')

# 统计最终标注结果
from collections import Counter
methods = Counter(row['【标注】检测方法'] for row in data)
results = Counter(row['【标注】检测结果_阳性阴性'] for row in data)
tumors = Counter(row['【标注】肿瘤类型'] for row in data if row['【标注】肿瘤类型'] not in ['待明确', ''])
clarity = Counter(row['【标注】诊断是否明确'] for row in data)

print('\n=== 标注结果统计 ===')
print('\n检测方法分布:')
for k, v in methods.most_common():
    print(f'  {k}: {v}例')

print('\n检测结果分布:')
for k, v in results.most_common():
    print(f'  {k}: {v}例')

print('\n肿瘤类型分布（Top 15）:')
for k, v in tumors.most_common(15):
    print(f'  {k}: {v}例')

print('\n诊断明确性:')
for k, v in clarity.most_common():
    print(f'  {k}: {v}例')

# 统计治疗靶点
targets = []
for row in data:
    t = row.get('【标注】治疗靶点', '')
    if t and t != '无':
        targets.extend(t.split('/'))
target_counts = Counter(targets)
print('\n治疗靶点分布:')
for k, v in target_counts.most_common():
    print(f'  {k}: {v}例')
