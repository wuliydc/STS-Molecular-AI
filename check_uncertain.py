import csv

data = []
with open('标注完成_200例.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

uncertain = [r for r in data if r['【标注】检测结果_阳性阴性'] == '无法判断']
print(f'剩余无法判断: {len(uncertain)}例\n')

# 会诊类：检测结果本来就不适用，应改为"不适用"
consult = [r for r in uncertain if r['【标注】检测方法'] == '会诊']
non_consult = [r for r in uncertain if r['【标注】检测方法'] != '会诊']

print(f'会诊类（检测结果应为"不适用"）: {len(consult)}例')
print(f'分子检测类（真正需要确认）: {len(non_consult)}例\n')

if non_consult:
    print('=== 分子检测类需确认病例 ===')
    for r in non_consult:
        idx = r['序号']
        method = r['【标注】检测方法']
        desc = r['大体描述_原文'][:60]
        conclusion = r['诊断结论_原文'][:150]
        print(f'序号{idx} | {method}')
        print(f'  大体描述: {desc}')
        print(f'  诊断结论: {conclusion}')
        print()
