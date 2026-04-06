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
