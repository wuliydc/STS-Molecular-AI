"""
深入分析不一致的真实来源：
区分"同基因不一致"vs"不同基因导致的表观不一致"
"""
import csv
from collections import Counter, defaultdict

data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): data.append(row)

raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): raw.append(row)

# 构建患者级详细检测记录
patient_tests = defaultdict(lambda: {'FISH_genes': set(), 'RNA_genes': set(),
                                      'FISH_pos': set(), 'RNA_pos': set(),
                                      'FISH_neg': set(), 'RNA_neg': set()})

FISH_GENE_MAP = {
    'DDIT3':'DDIT3','EWSR1':'EWSR1','MDM2':'MDM2','SS18':'SS18',
    'ALK':'ALK','NTRK1':'NTRK','NTRK2':'NTRK','NTRK3':'NTRK',
    'TFE3':'TFE3','CMET':'MET','ROS1':'ROS1','HER2':'HER2'
}
RNA_GENE_MAP = {
    'SS18':'SS18','EWSR1':'EWSR1','DDIT3':'DDIT3','ALK':'ALK',
    'NTRK':'NTRK','RET':'RET','ROS1':'ROS1','FGFR':'FGFR',
    'FUS':'FUS','BCOR':'BCOR','CIC':'CIC','NAB2':'NAB2'
}

for row in raw:
    name = row.get('姓名',''); pid = row.get('病案号','')
    if not name: continue
    key = (name, pid)
    method = row.get('检测方法','')
    gene   = row.get('检测基因','')
    result = row.get('检测结果','')
    fusion = row.get('融合伴侣基因','')

    if method == 'FISH':
        for g, std in FISH_GENE_MAP.items():
            if g in gene:
                patient_tests[key]['FISH_genes'].add(std)
                if result == '阳性': patient_tests[key]['FISH_pos'].add(std)
                elif result == '阴性': patient_tests[key]['FISH_neg'].add(std)

    elif method == 'RNA-NGS':
        if result == '阳性' and fusion:
            # 从融合伴侣提取基因
            for g, std in RNA_GENE_MAP.items():
                if g in fusion:
                    patient_tests[key]['RNA_genes'].add(std)
                    patient_tests[key]['RNA_pos'].add(std)
        elif result == '阴性':
            # RNA-NGS阴性：记录检测了哪些基因
            patient_tests[key]['RNA_genes'].update(['SS18','EWSR1','DDIT3','ALK','NTRK','RET','ROS1'])
            patient_tests[key]['RNA_neg'].update(['SS18','EWSR1','DDIT3','ALK','NTRK','RET','ROS1'])

# 分析真正的同基因不一致
same_gene_discord = []   # 同一基因，FISH vs RNA-NGS结果不同
diff_gene_discord = []   # 不同基因，表观不一致

fish_rna_patients = [r for r in data
                     if r['FISH结果'] in ['阳性','阴性']
                     and r['RNA_NGS结果'] in ['阳性','阴性']]

for r in fish_rna_patients:
    name, pid = r['姓名'], r['病案号']
    key = (name, pid)
    pt = patient_tests[key]

    fish_result = r['FISH结果']
    rna_result  = r['RNA_NGS结果']

    if fish_result == rna_result:
        continue  # 一致，跳过

    # 不一致病例：检查是否有共同检测基因
    common_genes = pt['FISH_genes'] & pt['RNA_genes']

    if common_genes:
        # 有共同基因：检查该基因的结果是否一致
        for g in common_genes:
            fish_g = 'pos' if g in pt['FISH_pos'] else 'neg' if g in pt['FISH_neg'] else 'unknown'
            rna_g  = 'pos' if g in pt['RNA_pos']  else 'neg' if g in pt['RNA_neg']  else 'unknown'
            if fish_g != rna_g and fish_g != 'unknown' and rna_g != 'unknown':
                same_gene_discord.append({
                    'patient': name, 'gene': g,
                    'FISH': fish_g, 'RNA': rna_g,
                    'tumor': r['肿瘤类型']
                })
    else:
        # 无共同基因：表观不一致（检测了不同基因）
        diff_gene_discord.append({
            'patient': name,
            'FISH_genes': pt['FISH_genes'],
            'RNA_genes': pt['RNA_genes'],
            'FISH_result': fish_result,
            'RNA_result': rna_result,
            'tumor': r['肿瘤类型']
        })

print('=== 不一致来源分析 ===')
print(f'总不一致患者: {len(same_gene_discord)+len(diff_gene_discord)}')
print(f'同基因真正不一致: {len(same_gene_discord)}例')
print(f'不同基因表观不一致: {len(diff_gene_discord)}例')

if same_gene_discord:
    print('\n同基因不一致详情:')
    gene_counts = Counter(d['gene'] for d in same_gene_discord)
    for g,c in gene_counts.most_common():
        print(f'  {g}: {c}例')

print('\n=== SS18融合伴侣详细分析 ===')
ss18_rows = [r for r in raw if 'SS18' in r.get('融合伴侣基因','')
             and r['检测方法']=='RNA-NGS' and r['检测结果']=='阳性']
ss18_partners = Counter(r['融合伴侣基因'] for r in ss18_rows)
print(f'SS18 RNA-NGS阳性总数: {len(ss18_rows)}')
for p,c in ss18_partners.most_common(8):
    print(f'  {p}: {c}')

print('\n=== ALK融合患者肿瘤类型 ===')
alk_patients = [r for r in data if 'ALK融合' in r.get('治疗靶点','')]
alk_tumors = Counter(r['肿瘤类型'] for r in alk_patients)
for t,c in alk_tumors.most_common():
    print(f'  {t}: {c}')
