import csv

data = []
with open('结构化数据集_全量.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# 问题1：自身配对融合伴侣
print('=== 自身配对融合伴侣样本（前5例）===')
self_pairs = [r for r in data if r['融合伴侣基因'] and
              r['融合伴侣基因'].split('-')[0] == r['融合伴侣基因'].split('-')[-1]]
print(f'总数: {len(self_pairs)}')
for r in self_pairs[:5]:
    print(f'  融合: {r["融合伴侣基因"]}')
    print(f'  结论: {r["诊断结论原文"][:120]}')
    print()

# 问题2：治疗靶点误判 - 看几个ALK融合的原始结论
print('=== 治疗靶点ALK融合样本（前5例）===')
alk = [r for r in data if 'ALK融合' in r['治疗靶点']]
print(f'总数: {len(alk)}')
for r in alk[:5]:
    print(f'  检测方法: {r["检测方法"]}  检测结果: {r["检测结果"]}')
    print(f'  结论: {r["诊断结论原文"][:150]}')
    print()
