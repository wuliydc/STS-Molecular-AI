"""
统一绘图样式配置 — Nature Communications 投稿标准
- 分辨率: 300 dpi (主图), 300 dpi (Supp)
- 格式: TIFF (主图) + PNG (预览)
- 背景: 白色 RGB (无透明通道)
- 字体: 统一 Arial/Helvetica 风格, 轴标签 11pt, 标题 12pt, 刻度 9pt
- 颜色: 色盲友好调色板
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 颜色方案（色盲友好，Wong 2011 palette）────────────────
COLORS = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'sky':    '#56B4E9',
    'yellow': '#F0E442',
    'black':  '#000000',
}
METHOD_COLORS = {
    'FISH':    '#0072B2',
    'RNA-NGS': '#009E73',
    'DNA-NGS': '#E69F00',
    '会诊':    '#CC79A7',
}
CONCORDANCE_COLORS = {
    'pp': '#009E73',   # concordant positive
    'nn': '#56B4E9',   # concordant negative
    'pn': '#E69F00',   # discordant A
    'np': '#D55E00',   # discordant B
}

# ── 全局 rcParams ─────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        # 字体
        'font.family':          ['Arial', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'font.size':            10,
        'axes.titlesize':       12,
        'axes.labelsize':       11,
        'xtick.labelsize':      9,
        'ytick.labelsize':      9,
        'legend.fontsize':      9,
        'figure.titlesize':     13,
        # 线条
        'axes.linewidth':       0.8,
        'xtick.major.width':    0.8,
        'ytick.major.width':    0.8,
        'lines.linewidth':      1.8,
        # 背景
        'figure.facecolor':     'white',
        'axes.facecolor':       'white',
        'savefig.facecolor':    'white',
        'savefig.transparent':  False,
        # 其他
        'axes.unicode_minus':   False,
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.grid':            False,
        'figure.dpi':           100,   # screen preview
    })

def save_figure(fig, name, dpi=300):
    """保存为 TIFF (投稿) + PNG (预览)"""
    fig.savefig(f'{name}.tiff', dpi=dpi, bbox_inches='tight',
                facecolor='white', format='tiff')
    fig.savefig(f'{name}.png',  dpi=150, bbox_inches='tight',
                facecolor='white', format='png')
    print(f'  Saved: {name}.tiff ({dpi} dpi) + {name}.png')

def panel_label(ax, label, x=-0.12, y=1.05):
    """添加 A/B/C 面板标签"""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

apply_style()

# ── Tumour type name translations (Chinese → English) ─────
TUMOR_EN = {
    # Lipomatous tumours
    '黏液样脂肪肉瘤':       'Myxoid liposarcoma',
    '黏液样/圆细胞脂肪肉瘤': 'Myxoid/round cell liposarcoma',
    '去分化脂肪肉瘤':       'Dedifferentiated liposarcoma',
    '高分化脂肪肉瘤':       'Well-differentiated liposarcoma',
    '多形性脂肪肉瘤':       'Pleomorphic liposarcoma',
    '脂肪肉瘤':             'Liposarcoma, NOS',
    # Synovial sarcoma
    '滑膜肉瘤':             'Synovial sarcoma',
    # Small round cell tumours
    '尤文肉瘤':             'Ewing sarcoma',
    '圆细胞肉瘤':           'Round cell sarcoma',
    '促纤维增生性小圆细胞肿瘤': 'Desmoplastic small round cell tumour',
    # Rhabdomyosarcoma
    '横纹肌肉瘤':           'Rhabdomyosarcoma',
    '胚胎性横纹肌肉瘤':     'Embryonal rhabdomyosarcoma',
    '腺泡状横纹肌肉瘤':     'Alveolar rhabdomyosarcoma',
    # Leiomyosarcoma
    '平滑肌肉瘤':           'Leiomyosarcoma',
    # Undifferentiated
    '未分化肉瘤':           'Undifferentiated sarcoma, NOS',
    '未分化多形性肉瘤':     'Undifferentiated pleomorphic sarcoma',
    # Bone-forming
    '骨肉瘤':               'Osteosarcoma',
    '软骨肉瘤':             'Chondrosarcoma',
    # Fibroblastic / myofibroblastic
    '孤立性纤维性肿瘤':     'Solitary fibrous tumour',
    '梭形细胞肿瘤':         'Spindle cell tumour, NOS',
    '纤维肉瘤':             'Fibrosarcoma',
    '低级别纤维肉瘤':       'Low-grade fibrosarcoma',
    '黏液纤维肉瘤':         'Myxofibrosarcoma',
    '隆突性皮肤纤维肉瘤':   'Dermatofibrosarcoma protuberans',
    '炎性肌纤维母细胞瘤':   'Inflammatory myofibroblastic tumour',
    '肌纤维母细胞肿瘤':     'Myofibroblastic tumour',
    # Vascular
    '血管肉瘤':             'Angiosarcoma',
    # Nerve sheath
    '恶性外周神经鞘瘤':     'Malignant peripheral nerve sheath tumour',
    # Clear cell / epithelioid / alveolar soft parts
    '腺泡状软组织肉瘤':     'Alveolar soft part sarcoma',
    '透明细胞肉瘤':         'Clear cell sarcoma',
    '上皮样肉瘤':           'Epithelioid sarcoma',
    # GIST
    '胃肠道间质瘤':         'GIST',
    # Joint
    '腱鞘巨细胞瘤':         'Tenosynovial giant cell tumour',
    # Misc
    '软组织肉瘤':           'Soft tissue sarcoma, NOS',
    '会诊':                 'Consultation',
    '待明确':               'Pending',
    '良性肿瘤':             'Benign tumour',
    '未知':                 'Unknown',
}

def tumor_en(name, maxlen=None):
    """Translate a Chinese tumour type name to English; fall back to original."""
    result = TUMOR_EN.get(name, name)
    if maxlen:
        result = result[:maxlen]
    return result
