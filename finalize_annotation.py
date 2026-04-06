import csv
import re

data = []
with open('标注完成_200例.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        data.append(row)

fixed = 0
for row in data:
    idx = row['序号']
    method = row['【标注】检测方法']
    conclusion = row['诊断结论_原文']
    result = row['【标注】检测结果_阳性阴性']

    # 1. 会诊类：检测结果改为"不适用"
    if method == '会诊' and result == '无法判断':
        row['【标注】检测结果_阳性阴性'] = '不适用'
        fixed += 1

    # 2. CMET FISH：比值 ≥2.0 为阳性
    if method == 'FISH' and 'CMET' in row['大体描述_原文'] and result == '无法判断':
        m = re.search(r'CMET/CEP7比值[=＝]\s*([\d.]+)', conclusion)
        if m:
            ratio = float(m.group(1))
            row['【标注】检测结果_阳性阴性'] = '阳性' if ratio >= 2.0 else '阴性'
            fixed += 1

    # 3. 全角"＜15%"导致正则未识别的FISH阴性
    if method == 'FISH' and result == '无法判断':
        if '＜15%' in conclusion or '< 15%' in conclusion:
            row['【标注】检测结果_阳性阴性'] = '阴性'
            fixed += 1

with open('标注完成_200例.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

# 最终统计
from collections import Counter
results = Counter(row['【标注】检测结果_阳性阴性'] for row in data)
methods = Counter(row['【标注】检测方法'] for row in data)
clarity = Counter(row['【标注】诊断是否明确'] for row in data)
tumors = Counter(r['【标注】肿瘤类型'] for r in data if r['【标注】肿瘤类型'] not in ['待明确', ''])

print(f'本次修正: {fixed}例')
print(f'\n=== 最终标注完成统计（200例）===')
print('\n检测方法:')
for k, v in methods.most_common():
    print(f'  {k}: {v}例')
print('\n检测结果:')
for k, v in results.most_common():
    print(f'  {k}: {v}例')
print('\n诊断明确性:')
for k, v in clarity.most_common():
    print(f'  {k}: {v}例')
print('\n肿瘤类型（已明确诊断）:')
for k, v in tumors.most_common():
    print(f'  {k}: {v}例')

# 检查是否还有无法判断
remaining = [r for r in data if r['【标注】检测结果_阳性阴性'] == '无法判断']
print(f'\n仍需人工确认: {len(remaining)}例')
