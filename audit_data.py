import csv
from collections import Counter, defaultdict

data = []
with open('患者级结构化数据集.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): data.append(row)

raw = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f): raw.append(row)

print('=== 肿瘤亚型分布（患者级）===')
tc = Counter(r['肿瘤类型'] for r in data if r['肿瘤类型'] not in ['待明确',''])
for t,c in tc.most_common(15):
    print(f'  {t}: {c}')

print('\n=== FISH各基因阳性率 ===')
fish_rows = [r for r in raw if r['检测方法']=='FISH']
gene_res = defaultdict(Counter)
for r in fish_rows:
    g = r.get('检测基因',''); res = r.get('检测结果','')
    if g and res in ['阳性','阴性']: gene_res[g][res] += 1
for g,c in sorted(gene_res.items(), key=lambda x: x[1].total(), reverse=True):
    total = c.total()
    pos_n = c['阳性']
    pos_rate = pos_n/total*100 if total>0 else 0
    print(f'  {g}: 阳性{pos_n}/{total} = {pos_rate:.1f}%')

print('\n=== 双阴患者（FISH-/RNA-）肿瘤类型 ===')
nn = [r for r in data if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阴性']
nn_t = Counter(r['肿瘤类型'] for r in nn if r['肿瘤类型'] not in ['待明确',''])
for t,c in nn_t.most_common(8):
    print(f'  {t}: {c}')

print('\n=== 平滑肌肉瘤检测模式 ===')
lms = [r for r in data if r['肿瘤类型']=='平滑肌肉瘤']
lms_strat = Counter(r['检测方法组合'] for r in lms)
for s,c in lms_strat.most_common():
    print(f'  {s}: {c}')
fish_pos = sum(1 for r in lms if r['FISH结果']=='阳性')
rna_pos  = sum(1 for r in lms if r['RNA_NGS结果']=='阳性')
dna_pos  = sum(1 for r in lms if r['DNA_NGS结果']=='阳性')
print(f'  FISH阳性: {fish_pos}/{len(lms)}')
print(f'  RNA-NGS阳性: {rna_pos}/{len(lms)}')
print(f'  DNA-NGS阳性: {dna_pos}/{len(lms)}')

print('\n=== 滑膜肉瘤SS18融合伴侣 ===')
ss18_rna = [r for r in raw if 'SS18' in r.get('融合伴侣基因','') and r['检测方法']=='RNA-NGS']
ss18_partners = Counter(r['融合伴侣基因'] for r in ss18_rna)
for p,c in ss18_partners.most_common(5):
    print(f'  {p}: {c}')

print('\n=== 治疗靶点分布（患者级）===')
all_targets = []
for r in data:
    t = r.get('治疗靶点','')
    if t and t != '无':
        all_targets.extend(t.split('/'))
tc2 = Counter(all_targets)
for t,c in tc2.most_common():
    print(f'  {t}: {c}')

print('\n=== 不一致率详细分析 ===')
fish_rna = [r for r in data if r['FISH结果'] in ['阳性','阴性'] and r['RNA_NGS结果'] in ['阳性','阴性']]
pp = sum(1 for r in fish_rna if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阳性')
nn2= sum(1 for r in fish_rna if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阴性')
pn = sum(1 for r in fish_rna if r['FISH结果']=='阳性' and r['RNA_NGS结果']=='阴性')
np_= sum(1 for r in fish_rna if r['FISH结果']=='阴性' and r['RNA_NGS结果']=='阳性')
print(f'  总计: {len(fish_rna)}')
print(f'  双阳: {pp} ({pp/len(fish_rna)*100:.1f}%)')
print(f'  双阴: {nn2} ({nn2/len(fish_rna)*100:.1f}%)')
print(f'  FISH+/RNA-: {pn} ({pn/len(fish_rna)*100:.1f}%)')
print(f'  FISH-/RNA+: {np_} ({np_/len(fish_rna)*100:.1f}%)')
print(f'  不一致率: {(pn+np_)/len(fish_rna)*100:.1f}%')
