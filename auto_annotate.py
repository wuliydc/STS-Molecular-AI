import csv
import re

def detect_method(desc):
    if not desc:
        return '会诊'
    if 'FISH' in desc or '荧光染色体原位杂交' in desc:
        return 'FISH'
    if any(kw in desc for kw in ['肿瘤常见基因易位', '易位基因检测', '肿瘤常见易位']):
        return 'RNA-NGS'
    if any(kw in desc for kw in ['EGFR基因', 'KRAS基因', 'BRAF基因', 'DNA', 'TMB', 'NGS']):
        return 'DNA-NGS'
    if re.search(r'编号|白片|HE X|IHC X|蜡块', desc):
        return '会诊'
    return '会诊'

def detect_gene_fish(desc):
    genes = ['DDIT3','EWSR1','MDM2','SS18','ALK','NTRK1','NTRK2','NTRK3',
             'TFE3','CMET','ROS1','CDK4','BCOR','CIC','FUS','NRG1']
    found = [g for g in genes if g in desc]
    if found:
        return '/'.join(found)
    return 'Panel（多基因）'

def detect_result(conclusion, method):
    if not conclusion:
        return '无法判断', ''
    
    # FISH阳性判断
    if method == 'FISH':
        # MDM2/HER2等扩增基因：用比值判断，阳性标准 ≥2.0
        ratio_m = re.search(r'(MDM2|HER2|CDK4)[^。\n]*?比值[=＝]\s*([\d.]+)', conclusion)
        if ratio_m:
            ratio = float(ratio_m.group(2))
            return ('阳性', '') if ratio >= 2.0 else ('阴性', '')
        # 易位基因：用阳性细胞比例判断，阳性标准 ≥15%
        pct_m = re.search(r'阳性细胞比例[：:]\s*(\d+)%', conclusion)
        if pct_m:
            pct = int(pct_m.group(1))
            return ('阳性', '') if pct >= 15 else ('阴性', '')
        if any(kw in conclusion for kw in ['未见异常', '未见分离', '阴性']):
            return '阴性', ''
        return '无法判断', ''

    # RNA-NGS
    if method == 'RNA-NGS':
        if '未显示' in conclusion and '基因易位' in conclusion:
            return '阴性', ''
        m = re.search(r'显示(\w+)基因易位\(([^)]+)\)', conclusion)
        if m:
            gene = m.group(1)
            partner = m.group(2)
            return '阳性', partner
        m2 = re.search(r'显示(\w+[-_]\w+)融合', conclusion)
        if m2:
            return '阳性', m2.group(1)
        if re.search(r'显示\w+基因易位', conclusion):
            return '阳性', ''
        return '阴性', ''

    # DNA-NGS
    if method == 'DNA-NGS':
        if re.search(r'显示\w+基因.*?突变|显示\w+.*?扩增', conclusion):
            return '阳性', ''
        if '未显示' in conclusion:
            return '阴性', ''
        return '无法判断', ''

    return '无法判断', ''

def detect_fusion_partner(conclusion, method, result):
    if method != 'RNA-NGS' or result != '阳性':
        return ''
    m = re.search(r'显示(\w+[-:]\w+[-:]\w+[-:]\w+)', conclusion)
    if m:
        raw = m.group(1)
        parts = re.split(r'[-:]', raw)
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[-1]}"
    m2 = re.search(r'显示(\w+)基因易位\(([^)]+)\)', conclusion)
    if m2:
        gene = m2.group(1)
        detail = m2.group(2)
        partner_m = re.search(r'(\w+):intron|(\w+):exon', detail)
        if partner_m:
            partner = partner_m.group(1) or partner_m.group(2)
            return f"{partner}-{gene}"
        return f"未知-{gene}"
    m3 = re.search(r'显示(\w+[-_]\w+)融合', conclusion)
    if m3:
        return m3.group(1).replace('_', '-')
    return ''

def detect_mutations(conclusion, method):
    if method != 'DNA-NGS':
        return ''
    mutations = []
    for m in re.finditer(r'显示(\w+)基因.*?突变', conclusion):
        mutations.append(f"{m.group(1)}(mut)")
    for m in re.finditer(r'显示(\w+).*?扩增', conclusion):
        mutations.append(f"{m.group(1)}(amp)")
    if not mutations:
        if '未显示' in conclusion:
            return '阴性'
    return '/'.join(mutations) if mutations else ''

def detect_targets(conclusion, method, result, fusion_partner):
    targets = []
    # 融合靶点
    targetable_fusions = ['ALK', 'NTRK', 'RET', 'ROS1', 'FGFR', 'NRG1', 'MET']
    if result == '阳性':
        for t in targetable_fusions:
            if t in conclusion and '未显示' not in conclusion[:conclusion.find(t)] if t in conclusion else False:
                targets.append(f"{t}融合")
        if fusion_partner:
            for t in targetable_fusions:
                if t in fusion_partner:
                    if f"{t}融合" not in targets:
                        targets.append(f"{t}融合")
    # TMB
    m = re.search(r'TMB[）)]*[：:]\s*(\d+)', conclusion)
    if m and int(m.group(1)) >= 10:
        targets.append('TMB-H')
    # MSI
    if 'MSI-H' in conclusion or '微卫星高度不稳定' in conclusion:
        targets.append('MSI-H')
    return '/'.join(targets) if targets else '无'

def detect_tumor_type(conclusion, method):
    if method in ['FISH', 'RNA-NGS', 'DNA-NGS']:
        return '待明确'
    # 优先匹配最具体的诊断（顺序从窄到宽）
    tumor_map = [
        ('黏液样脂肪肉瘤',              ['黏液样脂肪肉瘤', '粘液样脂肪肉瘤']),
        ('高分化脂肪肉瘤',              ['高分化脂肪肉瘤', '非典型脂肪瘤样肿瘤']),
        ('去分化脂肪肉瘤',              ['去分化脂肪肉瘤']),
        ('多形性脂肪肉瘤',              ['多形性脂肪肉瘤']),
        ('滑膜肉瘤',                    ['滑膜肉瘤']),
        ('低度恶性纤维黏液样肉瘤',      ['低度恶性纤维粘液性肉瘤', '低度恶性纤维黏液样肉瘤', 'LGFMS']),
        ('孤立性纤维性肿瘤',            ['孤立性纤维性肿瘤', '孤立性纤维瘤']),
        ('尤文肉瘤',                    ['尤文肉瘤', 'Ewing']),
        ('胚胎型横纹肌肉瘤',            ['胚胎型横纹肌肉瘤']),
        ('腺泡型横纹肌肉瘤',            ['腺泡型横纹肌肉瘤']),
        ('多形性横纹肌肉瘤',            ['多形性横纹肌肉瘤']),
        ('横纹肌肉瘤',                  ['横纹肌肉瘤']),
        ('平滑肌肉瘤',                  ['平滑肌肉瘤']),
        ('上皮样血管内皮瘤',            ['上皮样血管内皮细胞瘤', '上皮样血管内皮瘤']),
        ('血管肉瘤',                    ['血管肉瘤']),
        ('恶性血管球瘤',                ['恶性血管球瘤']),
        ('未分化多形性肉瘤',            ['未分化多形性肉瘤']),
        ('未分化圆细胞肉瘤',            ['未分化圆细胞肉瘤']),
        ('低度恶性肌纤维母细胞肉瘤',    ['低度恶性肌纤维母细胞肉瘤', '肌纤维母细胞肉瘤']),
        ('骨肉瘤',                      ['骨肉瘤']),
        ('软骨肉瘤',                    ['软骨肉瘤']),
        ('腺泡状软组织肉瘤',            ['腺泡状软组织肉瘤']),
        ('上皮样肉瘤',                  ['上皮样肉瘤']),
        ('肉瘤样癌',                    ['肉瘤样癌']),
        ('恶性间叶源性肿瘤',            ['恶性间叶源性肿瘤']),
        ('梭形细胞软组织肿瘤',          ['梭形细胞']),
    ]
    for std_name, keywords in tumor_map:
        for kw in keywords:
            if kw in conclusion:
                return std_name
    if '良性' in conclusion or '脂肪瘤' in conclusion:
        return '良性肿瘤'
    if '待明确' in conclusion or '建议' in conclusion or '不除外' in conclusion:
        return '待明确'
    return '待明确'

def detect_malignancy(conclusion, tumor_type):
    if tumor_type == '待明确':
        if '恶性' in conclusion:
            return '恶性'
        if '良性' in conclusion:
            return '良性'
        if '不除外低度恶性' in conclusion or '中间型' in conclusion:
            return '中间型'
        return '待明确'
    benign = ['良性肿瘤', '脂肪瘤', '血管瘤', '神经鞘瘤']
    intermediate = ['孤立性纤维性肿瘤', '隆突性皮肤纤维肉瘤']
    if tumor_type in benign:
        return '良性'
    if tumor_type in intermediate:
        return '中间型'
    return '恶性'

def detect_diagnosis_clarity(conclusion, method):
    if method in ['FISH', 'RNA-NGS', 'DNA-NGS']:
        return '待明确'
    if any(kw in conclusion for kw in ['符合', '考虑', '倾向', '形态符合', '不除外']):
        return '倾向性诊断'
    if any(kw in conclusion for kw in ['建议', '待进一步', '需结合', '必要时']):
        return '待明确'
    if conclusion.strip():
        return '明确'
    return '待明确'

# ---- 主流程 ----
data = []
with open('标注工作表_200例.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        data.append(row)

uncertain_cases = []
annotated = []

for row in data:
    desc = row.get('大体描述_原文', '')
    conclusion = row.get('诊断结论_原文', '')
    idx = row['序号']

    method = detect_method(desc)
    gene = detect_gene_fish(desc) if method == 'FISH' else ('Panel（多基因）' if method in ['RNA-NGS', 'DNA-NGS'] else '')
    result, _ = detect_result(conclusion, method)
    fusion = detect_fusion_partner(conclusion, method, result)
    mutation = detect_mutations(conclusion, method)
    target = detect_targets(conclusion, method, result, fusion)
    tumor_type = detect_tumor_type(conclusion, method)
    malignancy = detect_malignancy(conclusion, tumor_type)
    clarity = detect_diagnosis_clarity(conclusion, method)

    row['【标注】肿瘤类型'] = tumor_type
    row['【标注】肿瘤良恶性'] = malignancy
    row['【标注】检测方法'] = method
    row['【标注】检测基因'] = gene
    row['【标注】检测结果_阳性阴性'] = result
    row['【标注】融合伴侣基因'] = fusion
    row['【标注】突变类型'] = mutation
    row['【标注】治疗靶点'] = target
    row['【标注】诊断是否明确'] = clarity
    row['【标注】备注_疑难点'] = ''

    # 标记不确定病例
    if result == '无法判断' or (method == 'FISH' and result == '无法判断'):
        uncertain_cases.append({
            'idx': idx,
            'desc': desc[:60],
            'conclusion': conclusion[:120],
            'method': method,
            'result': result
        })

    annotated.append(row)

# 保存标注结果
with open('标注完成_200例.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(annotated)

print(f'标注完成: {len(annotated)}例')
print(f'需人工确认: {len(uncertain_cases)}例')
print(f'\n=== 需要你确认的病例 ===')
for c in uncertain_cases[:20]:
    print(f"\n序号{c['idx']} | 方法:{c['method']} | 当前判断:{c['result']}")
    print(f"  大体描述: {c['desc']}")
    print(f"  诊断结论: {c['conclusion']}")
