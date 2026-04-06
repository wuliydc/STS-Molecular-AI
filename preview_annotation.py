import csv

data = []
with open('标注工作表_200例.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# 打印前10例的原始内容，便于查看
for i, row in enumerate(data[:10], 1):
    print(f"\n{'='*60}")
    print(f"序号: {row['序号']}  病理号: {row['病理号']}  报告类型: {row['报告类型']}")
    print(f"大体描述: {row['大体描述_原文'][:80]}")
    print(f"诊断结论: {row['诊断结论_原文'][:150]}")
