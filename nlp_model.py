"""
NLP模型：基于规则+机器学习的病理报告自动标注
Step 1: 在200例金标准上评估规则模型性能
Step 2: 对全量12385条报告做自动提取
Step 3: 输出结构化数据集
"""
import csv
import re
import json
from collections import Counter, defaultdict

# ============================================================
# 核心提取函数（与auto_annotate.py一致，已融入人工修正规则）
# ============================================================

def detect_method(desc):
    if not desc:
        return '会诊'
    if 'FISH' in desc or '荧光染色体原位杂交' in desc:
        return 'FISH'
    if any(kw in desc for kw in ['肿瘤常见基因易位', '易位基因检测', '肿瘤常见易位']):
        return 'RNA-NGS'
    if any(kw in desc for kw in ['EGFR基因', 'KRAS基因', 'BRAF基因', 'DNA', 'TMB']):
        return 'DNA-NGS'
    if re.search(r'编号|白片|HE X|IHC X|蜡块', desc):
        return '会诊'
    return '会诊'

def detect_gene(desc, method):
    if method == '会诊':
        return ''
    if method in ['RNA-NGS', 'DNA-NGS']:
        return 'Panel（多基因）'
    # FISH: 提取具体基因
    genes = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3',
             'TFE3','CMET','ROS1','CDK4','BCOR','CIC','FUS','NRG1','HER2']
    found = [g for g in genes if g in desc]
    return '/'.join(found) if found else 'Panel（多基因）'

def detect_result(conclusion, method, desc):
    if not conclusion:
        return '无法判断'
    if method == '会诊':
        return '不适用'

    if method == 'FISH':
        # 扩增基因（比值判断，阈值≥2.0）
        amp_genes = ['MDM2', 'HER2', 'CMET', 'CDK4']
        for gene in amp_genes:
            if gene in desc:
                m = re.search(rf'{gene}/\w+比值[=＝]\s*([\d.]+)', conclusion)
                if m:
                    return '阳性' if float(m.group(1)) >= 2.0 else '阴性'
        # 易位基因（阳性细胞比例，阈值≥15%）
        pct_m = re.search(r'阳性细胞比例[：:]\s*(\d+)%', conclusion)
        if pct_m:
            return '阳性' if int(pct_m.group(1)) >= 15 else '阴性'
        # 全角<15%
        if '＜15%' in conclusion or '< 15%' in conclusion:
            return '阴性'
        if any(kw in conclusion for kw in ['未见异常', '未见分离', '阴性']):
            return '阴性'
        return '无法判断'

    if method == 'RNA-NGS':
        if re.search(r'未显示.{0,50}基因易位', conclusion):
            return '阴性'
        if re.search(r'显示\w+基因易位|显示\w+-\w+融合', conclusion):
            return '阳性'
        return '阴性'

    if method == 'DNA-NGS':
        if re.search(r'显示\w+基因.{0,10}(突变|扩增|缺失)', conclusion):
            return '阳性'
        if '未显示' in conclusion:
            return '阴性'
        return '无法判断'

    return '无法判断'

def detect_fusion(conclusion, method, result):
    if method != 'RNA-NGS' or result != '阳性':
        return ''

    # 格式1: GENE1:exonN::GENE2:exonN  (标准融合格式)
    m1 = re.search(r'(\w+)\s*(?:exon\d+|intron\d+)::\s*(\w+)\s*(?:exon\d+|intron\d+)', conclusion)
    if m1:
        g1, g2 = m1.group(1), m1.group(2)
        if g1 != g2 and 'intergenic' not in g2.lower():
            return f'{g1}-{g2}'

    # 格式2: 显示XX基因易位(PARTNER:intronN-GENE:exonN)
    m2 = re.search(r'显示(\w+)基因易位\(([^)]+)\)', conclusion)
    if m2:
        gene = m2.group(1)
        detail = m2.group(2)
        # 提取伴侣基因（冒号前的单词）
        partner_m = re.search(r'^(\w+)[:\s]', detail)
        partner = partner_m.group(1) if partner_m else None
        # 排除自身配对和intergenic
        if partner and partner != gene and 'intergenic' not in partner.lower():
            return f'{partner}-{gene}'
        # intergenic融合：标记为基因间融合
        if partner and 'intergenic' in detail.lower():
            return f'{gene}-intergenic'
        return f'{gene}-未知'

    # 格式3: 显示GENE1-GENE2融合
    m3 = re.search(r'显示(\w+)[-_](\w+)融合', conclusion)
    if m3:
        g1, g2 = m3.group(1), m3.group(2)
        if g1 != g2:
            return f'{g1}-{g2}'

    return ''

def detect_mutations(conclusion, method):
    if method != 'DNA-NGS':
        return ''
    muts = []
    for m in re.finditer(r'显示(\w+)基因.{0,5}(突变|缺失)', conclusion):
        muts.append(f'{m.group(1)}(mut)')
    for m in re.finditer(r'显示(\w+).{0,5}扩增', conclusion):
        muts.append(f'{m.group(1)}(amp)')
    if not muts and '未显示' in conclusion:
        return '阴性'
    return '/'.join(muts) if muts else ''

def detect_targets(conclusion, method, result, fusion):
    targets = []
    targetable = ['ALK', 'NTRK', 'RET', 'ROS1', 'FGFR', 'NRG1', 'MET']

    # 只从融合伴侣字段提取靶点（已确认阳性）
    if fusion and result == '阳性':
        for t in targetable:
            if re.search(rf'\b{t}\b', fusion):
                targets.append(f'{t}融合')

    # RNA-NGS阳性：从结论中提取，但必须排除"未显示"前缀
    if method == 'RNA-NGS' and result == '阳性':
        for t in targetable:
            # 匹配"显示XX基因易位"但排除"未显示"
            if re.search(rf'(?<!未)显示{t}', conclusion):
                label = f'{t}融合'
                if label not in targets:
                    targets.append(label)

    # DNA-NGS：只提取明确阳性的突变靶点
    if method == 'DNA-NGS' and result == '阳性':
        if re.search(r'(?<!未)显示BRAF基因.*?V600E', conclusion):
            targets.append('BRAF V600E')

    # TMB-H（≥10 mut/Mb）
    m = re.search(r'TMB[^：:\d]*[：:]\s*([\d.]+)', conclusion)
    if m:
        try:
            if float(m.group(1)) >= 10:
                targets.append('TMB-H')
        except ValueError:
            pass

    # MSI-H
    if 'MSI-H' in conclusion or '微卫星高度不稳定' in conclusion:
        targets.append('MSI-H')

    return '/'.join(targets) if targets else '无'

def detect_tumor_type(conclusion, method):
    if method in ['FISH', 'RNA-NGS', 'DNA-NGS']:
        return '待明确'
    tumor_map = [
        ('黏液样脂肪肉瘤',           ['黏液样脂肪肉瘤', '粘液样脂肪肉瘤']),
        ('高分化脂肪肉瘤',           ['高分化脂肪肉瘤', '非典型脂肪瘤样肿瘤']),
        ('去分化脂肪肉瘤',           ['去分化脂肪肉瘤']),
        ('多形性脂肪肉瘤',           ['多形性脂肪肉瘤']),
        ('滑膜肉瘤',                 ['滑膜肉瘤']),
        ('低度恶性纤维黏液样肉瘤',   ['低度恶性纤维粘液性肉瘤', '低度恶性纤维黏液样肉瘤']),
        ('孤立性纤维性肿瘤',         ['孤立性纤维性肿瘤', '孤立性纤维瘤']),
        ('尤文肉瘤',                 ['尤文肉瘤', 'Ewing']),
        ('胚胎型横纹肌肉瘤',         ['胚胎型横纹肌肉瘤']),
        ('腺泡型横纹肌肉瘤',         ['腺泡型横纹肌肉瘤']),
        ('多形性横纹肌肉瘤',         ['多形性横纹肌肉瘤']),
        ('横纹肌肉瘤',               ['横纹肌肉瘤']),
        ('平滑肌肉瘤',               ['平滑肌肉瘤']),
        ('上皮样血管内皮瘤',         ['上皮样血管内皮细胞瘤', '上皮样血管内皮瘤']),
        ('血管肉瘤',                 ['血管肉瘤']),
        ('恶性血管球瘤',             ['恶性血管球瘤']),
        ('未分化多形性肉瘤',         ['未分化多形性肉瘤']),
        ('未分化圆细胞肉瘤',         ['未分化圆细胞肉瘤']),
        ('未分化肉瘤',               ['未分化肉瘤']),
        ('低度恶性肌纤维母细胞肉瘤', ['低度恶性肌纤维母细胞肉瘤', '肌纤维母细胞肉瘤']),
        ('骨肉瘤',                   ['骨肉瘤']),
        ('软骨肉瘤',                 ['软骨肉瘤']),
        ('腺泡状软组织肉瘤',         ['腺泡状软组织肉瘤']),
        ('上皮样肉瘤',               ['上皮样肉瘤']),
        ('肉瘤样癌',                 ['肉瘤样癌']),
        ('恶性间叶源性肿瘤',         ['恶性间叶源性肿瘤']),
        ('梭形细胞软组织肿瘤',       ['梭形细胞']),
    ]
    for std_name, keywords in tumor_map:
        for kw in keywords:
            if kw in conclusion:
                return std_name
    if any(kw in conclusion for kw in ['良性', '脂肪瘤', '血管瘤', '神经鞘瘤']):
        return '良性肿瘤'
    return '待明确'

def detect_malignancy(conclusion, tumor_type):
    benign = ['良性肿瘤']
    intermediate = ['孤立性纤维性肿瘤', '上皮样血管内皮瘤', '隆突性皮肤纤维肉瘤']
    if tumor_type in benign:
        return '良性'
    if tumor_type in intermediate:
        return '中间型'
    if tumor_type not in ['待明确']:
        return '恶性'
    if '恶性' in conclusion:
        return '恶性'
    if '良性' in conclusion:
        return '良性'
    if any(kw in conclusion for kw in ['不除外低度恶性', '中间型']):
        return '中间型'
    return '待明确'

def detect_clarity(conclusion, method):
    if method in ['FISH', 'RNA-NGS', 'DNA-NGS']:
        return '待明确'
    if any(kw in conclusion for kw in ['符合', '考虑', '倾向', '形态符合', '不除外']):
        return '倾向性诊断'
    if any(kw in conclusion for kw in ['建议', '待进一步', '需结合', '必要时']):
        return '待明确'
    if conclusion.strip():
        return '明确'
    return '待明确'

# ============================================================
# Step 1: 在200例金标准上评估性能
# ============================================================
print('='*60)
print('Step 1: 在200例金标准上评估NLP模型性能')
print('='*60)

gold = []
with open('标注完成_200例.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gold.append(row)

fields_to_eval = {
    '【标注】检测方法':         '检测方法',
    '【标注】检测结果_阳性阴性': '检测结果',
}

for field, label in fields_to_eval.items():
    correct = 0
    total = 0
    errors = []
    for row in gold:
        desc = row['大体描述_原文']
        conclusion = row['诊断结论_原文']
        gold_val = row[field]
        if not gold_val:
            continue

        if field == '【标注】检测方法':
            pred = detect_method(desc)
        elif field == '【标注】检测结果_阳性阴性':
            method = detect_method(desc)
            pred = detect_result(conclusion, method, desc)

        total += 1
        if pred == gold_val:
            correct += 1
        else:
            errors.append((row['序号'], gold_val, pred))

    acc = correct / total * 100 if total > 0 else 0
    print(f'\n{label}: {correct}/{total} = {acc:.1f}%')
    if errors[:5]:
        print(f'  错误示例（前5）:')
        for idx, g, p in errors[:5]:
            print(f'    序号{idx}: 金标准={g}, 预测={p}')

# ============================================================
# Step 2: 对全量12385条报告做自动提取
# ============================================================
print('\n' + '='*60)
print('Step 2: 全量数据提取')
print('='*60)

all_data = []
with open('合并汇总.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_data.append(row)

structured = []
for row in all_data:
    desc = row.get('大体描述', '')
    conclusion = row.get('诊断结论', '')

    method = detect_method(desc)
    gene = detect_gene(desc, method)
    result = detect_result(conclusion, method, desc)
    fusion = detect_fusion(conclusion, method, result)
    mutation = detect_mutations(conclusion, method)
    target = detect_targets(conclusion, method, result, fusion)
    tumor_type = detect_tumor_type(conclusion, method)
    malignancy = detect_malignancy(conclusion, tumor_type)
    clarity = detect_clarity(conclusion, method)

    structured.append({
        '病理号':       row.get('病理号', ''),
        '既往编号':     row.get('既往编号', ''),
        '就诊卡号':     row.get('就诊卡号', ''),
        '病案号':       row.get('病案号', ''),
        '姓名':         row.get('姓名', ''),
        '性别':         row.get('性别', ''),
        '年龄':         row.get('年龄', ''),
        '报告类型':     row.get('报告类型', ''),
        '登记时间':     row.get('登记时间', ''),
        '检测方法':     method,
        '检测基因':     gene,
        '检测结果':     result,
        '融合伴侣基因': fusion,
        '突变类型':     mutation,
        '治疗靶点':     target,
        '肿瘤类型':     tumor_type,
        '肿瘤良恶性':   malignancy,
        '诊断明确性':   clarity,
        '诊断结论原文': conclusion,
    })

# 保存结构化数据集
out_fields = list(structured[0].keys())
with open('结构化数据集_全量.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=out_fields)
    writer.writeheader()
    writer.writerows(structured)

print(f'全量提取完成: {len(structured)}条记录')

# ============================================================
# Step 3: 统计分析
# ============================================================
print('\n' + '='*60)
print('Step 3: 结构化数据统计')
print('='*60)

methods = Counter(r['检测方法'] for r in structured)
results = Counter(r['检测结果'] for r in structured)
tumors = Counter(r['肿瘤类型'] for r in structured if r['肿瘤类型'] not in ['待明确', ''])
fusions = Counter(r['融合伴侣基因'] for r in structured if r['融合伴侣基因'])
targets = []
for r in structured:
    t = r['治疗靶点']
    if t and t != '无':
        targets.extend(t.split('/'))
target_counts = Counter(targets)

print('\n检测方法分布:')
for k, v in methods.most_common():
    print(f'  {k}: {v}')

print('\n检测结果分布:')
for k, v in results.most_common():
    print(f'  {k}: {v}')

print('\n肿瘤类型分布（Top 15）:')
for k, v in tumors.most_common(15):
    print(f'  {k}: {v}')

print('\n融合伴侣基因（Top 15）:')
for k, v in fusions.most_common(15):
    print(f'  {k}: {v}')

print('\n治疗靶点分布:')
for k, v in target_counts.most_common():
    print(f'  {k}: {v}')

# 患者级统计
patient_data = defaultdict(list)
for r in structured:
    key = (r['姓名'], r['病案号'])
    if key[0]:
        patient_data[key].append(r)

print(f'\n患者级统计:')
print(f'  总患者数: {len(patient_data)}')
multi = {k: v for k, v in patient_data.items() if len(set(x["检测方法"] for x in v)) >= 2}
print(f'  多方法检测患者: {len(multi)}')
all_three = {k: v for k, v in patient_data.items()
             if {'FISH', 'RNA-NGS', 'DNA-NGS'}.issubset(set(x["检测方法"] for x in v))}
print(f'  三方法全做患者: {len(all_three)}')

from collections import defaultdict
