import csv
import re
from collections import defaultdict
import json

data = []
with open('合并汇总.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print('【FISH vs NGS 一致性分析】\n')

# 构建患者级数据
patient_tests = defaultdict(lambda: {'FISH': {}, 'NGS': {}})

genes = ['DDIT3', 'EWSR1', 'SS18', 'MDM2', 'ALK', 'NTRK']

for row in data:
    patient_id = (row['姓名'], row['病案号'])
    if not patient_id[0]:
        continue
    
    desc = row.get('大体描述', '')
    conclusion = row.get('诊断结论', '')
    
    # 判断检测方法和基因
    for gene in genes:
        if gene in desc:
            if 'FISH' in desc:
                # 判断阳性/阴性
                if any(kw in conclusion for kw in ['阳性', '融合', '易位', '扩增', '分离信号']):
                    patient_tests[patient_id]['FISH'][gene] = 'Positive'
                elif any(kw in conclusion for kw in ['阴性', '未显示', '未见']):
                    patient_tests[patient_id]['FISH'][gene] = 'Negative'
            elif 'NGS' in desc or '基因易位' in desc:
                if any(kw in conclusion for kw in ['显示' + gene, gene + '基因', '融合', '易位']):
                    patient_tests[patient_id]['NGS'][gene] = 'Positive'
                elif '未显示' + gene in conclusion or '未见' + gene in conclusion:
                    patient_tests[patient_id]['NGS'][gene] = 'Negative'

# 统计一致性
concordance = defaultdict(lambda: {'concordant': 0, 'discordant': 0, 'details': []})

for patient_id, tests in patient_tests.items():
    for gene in genes:
        fish_result = tests['FISH'].get(gene)
        ngs_result = tests['NGS'].get(gene)
        
        if fish_result and ngs_result:
            if fish_result == ngs_result:
                concordance[gene]['concordant'] += 1
            else:
                concordance[gene]['discordant'] += 1
                concordance[gene]['details'].append({
                    'patient': patient_id[0],
                    'FISH': fish_result,
                    'NGS': ngs_result
                })

print('各基因FISH vs NGS一致性:')
for gene in genes:
    total = concordance[gene]['concordant'] + concordance[gene]['discordant']
    if total > 0:
        conc = concordance[gene]['concordant']
        disc = concordance[gene]['discordant']
        rate = conc / total * 100
        print(f'\n{gene}:')
        print(f'  一致: {conc}, 不一致: {disc}, 一致率: {rate:.1f}%')
        if disc > 0 and disc <= 5:
            print(f'  不一致病例:')
            for detail in concordance[gene]['details'][:5]:
                print(f'    - {detail["patient"]}: FISH={detail["FISH"]}, NGS={detail["NGS"]}')

# 总体统计
print(f'\n【总体数据质量评估】')
print(f'可用于建模的患者数: {len([p for p in patient_tests.values() if p["FISH"] or p["NGS"]])}')
print(f'FISH+NGS双检测患者数: {len([p for p in patient_tests.values() if p["FISH"] and p["NGS"]])}')

# 保存结构化数据
output_data = []
for patient_id, tests in patient_tests.items():
    if tests['FISH'] or tests['NGS']:
        output_data.append({
            'patient_name': patient_id[0],
            'patient_id': patient_id[1],
            'FISH_results': tests['FISH'],
            'NGS_results': tests['NGS']
        })

with open('structured_patient_data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f'\n已保存结构化数据到: structured_patient_data.json ({len(output_data)}例患者)')
