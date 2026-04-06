"""直接测试 API 逻辑，无需启动服务器"""
import sys
sys.path.insert(0, 'sts_ai_app')
from app import predict, PatientInput

test_cases = [
    {
        'name': 'Myxoid liposarcoma (FUS-DDIT3)',
        'input': PatientInput(age=45, sex='Female', fish_result='阴性',
                              fish_gene='DDIT3', rna_result='阳性',
                              rna_fusion='FUS-DDIT3')
    },
    {
        'name': 'Synovial sarcoma (SS18-SSX1)',
        'input': PatientInput(age=32, sex='Male', fish_result='阳性',
                              fish_gene='SS18', rna_result='阳性',
                              rna_fusion='SS18-SSX1')
    },
    {
        'name': 'Leiomyosarcoma (DNA-NGS positive)',
        'input': PatientInput(age=58, sex='Female', fish_result='阴性',
                              rna_result='阴性', dna_result='阳性',
                              dna_mutations='TP53(mut)/RB1(mut)', tmb_high=True)
    },
    {
        'name': 'NTRK fusion (targetable)',
        'input': PatientInput(age=28, sex='Male', fish_result='阴性',
                              rna_result='阳性', rna_fusion='ETV6-NTRK3')
    },
]

print('=== STS-Molecular-AI API Test ===\n')
all_pass = True
for tc in test_cases:
    result = predict(tc['input'])
    ok = result['top_diagnosis'] and result['confidence'] > 0
    status = '✓ PASS' if ok else '✗ FAIL'
    if not ok: all_pass = False
    print(f'{status}  [{tc["name"]}]')
    print(f'       Diagnosis : {result["top_diagnosis"]} ({result["confidence"]:.1%})')
    print(f'       Targets   : {[t["gene"] for t in result["therapeutic_targets"]] or "None"}')
    print(f'       Recommend : {result["testing_recommendation"][0][:60]}')
    print()

print('=== Health check ===')
from app import health
h = health()
print(f'Status: {h["status"]}  |  Classes: {h["n_classes"]}  |  Model: {h["model"]}')
print()
print('OVERALL:', '✓ ALL PASS' if all_pass else '✗ SOME FAILED')
print()
print('To start the web server, run in your terminal:')
print('  cd sts_ai_app')
print('  python -m uvicorn app:app --host 0.0.0.0 --port 8000')
print('Then open: http://localhost:8000')
