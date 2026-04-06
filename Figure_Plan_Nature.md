# 完整图谱方案（Nature Communications / Cancer Cell 投稿级别）

## 主图 Figure 1–7 + Supplementary Figures

---

| 图号 | 标题（顶刊风格） | 核心信息（1-3条） | 可视化形式 | 分析方法/模型 |
|------|----------------|-----------------|-----------|--------------|
| **Figure 1** | **Landscape of multi-modal molecular testing in a real-world soft tissue sarcoma cohort** | ①队列规模和构成（1,401例，7年，三方法）；②各检测方法的使用模式和时间趋势；③肿瘤亚型与检测方法的对应关系 | A: 队列纳入流程图（CONSORT）；B: 堆叠面积图（2018-2025年各方法使用趋势）；C: 气泡矩阵图（肿瘤亚型 × 检测方法频率）；D: Venn图（三方法重叠，568例核心） | 描述性统计；时间序列分析；卡方检验 |
| **Figure 2** | **Stepwise diagnostic gain of sequential molecular testing reveals method-specific and tumour-subtype-specific complementarity** | ①每加一种方法，诊断改变的比例（增量价值）；②RNA NGS和DNA NGS各自新增的诊断信息类型不同；③不同肿瘤亚型对三方法的依赖程度不同 | A: Sankey/Alluvial图（诊断从FISH→+RNA NGS→+DNA NGS的流向）；B: 瀑布图（每例患者诊断变化，按亚型着色）；C: 分组柱状图（各亚型中每步新增诊断率）；D: 热图（肿瘤亚型 × 方法贡献度） | 诊断一致性分析；McNemar检验；Cohen's Kappa；逐步回归 |
| **Figure 3** | **A natural language processing framework enables automated structured extraction of molecular pathology reports at scale** | ①NLP模型可高精度从非结构化报告中提取关键分子信息（F1>0.95）；②不同字段（肿瘤类型/基因结果/治疗靶点）的提取难度不同；③NLP提取标签与人工标注高度一致 | A: NLP系统架构流程图；B: 各字段Precision/Recall/F1对比条形图；C: 混淆矩阵（肿瘤类型分类）；D: 典型报告解析可视化（原文高亮→结构化输出）；E: NLP vs 人工标注一致性散点图 | BERT-NER / 规则+ML混合模型；命名实体识别；Cohen's Kappa；Bootstrap置信区间 |
| **Figure 4** | **A multi-modal AI classifier integrating FISH, RNA-NGS, and DNA-NGS achieves high diagnostic accuracy with interpretable molecular feature contributions** | ①三模态整合模型显著优于任意单一方法（AUC对比）；②消融实验量化每种模态的贡献；③SHAP分析揭示不同肿瘤亚型的诊断驱动特征不同 | A: 多分类ROC曲线（各模型对比）；B: 消融实验柱状图（去掉每种模态后AUC下降）；C: SHAP蜂群图（全局特征重要性）；D: SHAP瀑布图（3个典型病例个体化解释）；E: t-SNE/UMAP（568例患者分子特征空间分布，按亚型着色） | XGBoost/LightGBM；5折交叉验证；SHAP值；t-SNE/UMAP降维；消融实验 |
| **Figure 5** | **Systematic characterisation of FISH–NGS discordance uncovers biologically distinct tumour subsets with differential fusion partner landscapes** | ①FISH与RNA NGS不一致病例具有独特的临床和分子特征；②以DDIT3为核心，不同融合伴侣导致检测方法敏感性差异；③不一致类型（A型/B型）代表不同的生物学机制 | A: 四象限散点图（FISH vs RNA NGS结果，不一致病例高亮）；B: 森林图（不一致 vs 一致病例临床特征对比）；C: 环形图/网络图（DDIT3融合伴侣分布，FISH vs RNA NGS检出差异）；D: 不一致类型分类树；E: 不一致预测模型ROC曲线 | 逻辑回归；Fisher精确检验；网络分析；融合伴侣注释；随机森林不一致预测模型 |
| **Figure 6** | **An AI-driven testing strategy optimisation model minimises redundant testing while preserving diagnostic yield** | ①基于AI的检测策略推荐可减少X%的冗余检测；②不同临床场景下最优检测路径不同；③成本-效益分析支持分层检测策略 | A: 决策树可视化（检测策略推荐逻辑）；B: 成本-效益曲线（诊断准确率 vs 检测数量）；C: 热图（不同临床特征组合下推荐检测策略）；D: 模拟验证柱状图（AI推荐策略 vs 实际策略的诊断结果对比） | 决策树；强化学习/规则引擎；成本效益分析；蒙特卡洛模拟验证 |
| **Figure 7** | **STS-Molecular-AI: an open-source clinical decision support tool enabling real-time multi-modal diagnostic assistance** | ①在线工具实现端到端诊断辅助（输入→诊断建议→检测推荐）；②工具在独立测试集上性能验证；③用户评估显示临床可用性高 | A: 工具界面截图（输入/输出全流程）；B: 工具性能验证ROC曲线（独立测试集）；C: 用户评估雷达图（准确性/易用性/可解释性/临床相关性）；D: 工具架构图（技术栈） | FastAPI + React；Docker部署；用户体验评估（SUS量表）；独立测试集验证 |

---

## Supplementary Figures

| 图号 | 标题 | 核心内容 | 可视化形式 |
|------|------|---------|-----------|
| **Supp Fig 1** | Extended cohort characteristics | 年龄/性别/部位分布；时间趋势详细数据 | 小提琴图；地理分布图；时间折线图 |
| **Supp Fig 2** | Detailed FISH result distributions by gene | 各基因FISH阳性率、信号模式分布 | 分组箱线图；信号比值分布直方图 |
| **Supp Fig 3** | RNA NGS fusion partner landscape | 所有融合基因的伴侣基因网络 | 弦图（Chord diagram）；网络图 |
| **Supp Fig 4** | DNA NGS mutation spectrum | 突变基因频率；TMB分布；共突变矩阵 | OncoPrint图；TMB分布图；共突变热图 |
| **Supp Fig 5** | NLP model training details and error analysis | 学习曲线；错误类型分析；边界案例 | 学习曲线图；错误分类散点图 |
| **Supp Fig 6** | Full model comparison and hyperparameter tuning | 所有对比模型详细性能；超参数敏感性 | 多模型ROC对比；热图（超参数） |
| **Supp Fig 7** | SHAP analysis for individual tumour subtypes | 每种肿瘤亚型的特征重要性差异 | 分亚型SHAP蜂群图（6-8个亚型） |
| **Supp Fig 8** | Complete discordance case catalogue | 所有不一致病例的详细分子特征 | 热图；个案报告表格 |
| **Supp Fig 9** | Testing strategy simulation sensitivity analysis | 不同参数假设下策略推荐的稳健性 | 敏感性分析龙卷风图 |
| **Supp Fig 10** | Online tool validation on external-like holdout set | 工具在时间分割验证集上的性能 | ROC曲线；校准曲线（Calibration plot） |

---

## 图表逻辑串联（故事线）

```
Figure 1  →  Figure 2  →  Figure 3  →  Figure 4  →  Figure 5  →  Figure 6  →  Figure 7
  ↓              ↓              ↓              ↓              ↓              ↓              ↓
我们有什么    三方法各有    数据怎么      AI模型有      不一致病例    AI如何优化    工具如何
数据（队列）  贡献（增量）  结构化（NLP） 多准（分类）  说明什么（机制）检测策略     落地（转化）
```

**核心叙事逻辑**：
1. Fig1 建立可信度（我们的数据是真实的、大规模的）
2. Fig2 提出问题（三方法各有价值，但如何整合？）
3. Fig3 解决数据瓶颈（NLP让大规模分析成为可能）
4. Fig4 给出AI答案（整合模型优于任何单一方法）
5. Fig5 深挖机制（不一致不是误差，是生物学信号）
6. Fig6 指导实践（AI优化检测策略，减少冗余）
7. Fig7 转化落地（开源工具，任何医院都能用）

---

## 数据-图表对应关系

| 图号 | 所需数据 | 当前数据状态 | 需要补充 |
|------|---------|------------|---------|
| Figure 1 | 队列基本信息、检测记录 | ✅ 已有（12,385条） | 无 |
| Figure 2 | 568例三方法患者的诊断变化 | ⚠️ 需NLP提取诊断标签 | NLP标注 |
| Figure 3 | 200例人工标注金标准 | ❌ 需人工标注 | 人工标注200例 |
| Figure 4 | 568例结构化特征矩阵 | ⚠️ 需特征工程 | NLP完成后可做 |
| Figure 5 | 484例FISH+RNA NGS对比 | ⚠️ 需精细化提取 | 优化提取规则 |
| Figure 6 | 1,401例检测路径数据 | ✅ 已有 | 成本数据（可估算） |
| Figure 7 | 训练好的模型 | ❌ 待开发 | 模型训练完成后 |

---

## 关键里程碑时间表（9个月）

| 月份 | 里程碑 | 对应图表 |
|------|--------|---------|
| 第1个月 | 人工标注200例金标准 | Fig3基础 |
| 第2个月 | NLP模型开发完成，全量提取 | Fig3完成 |
| 第3个月 | 特征工程 + Fig1/Fig2完成 | Fig1, Fig2 |
| 第4个月 | AI分类模型 + SHAP分析 | Fig4完成 |
| 第5个月 | 不一致病例深度分析 | Fig5完成 |
| 第6个月 | 策略优化模型 | Fig6完成 |
| 第7个月 | 在线工具开发 | Fig7完成 |
| 第8个月 | 论文撰写 + Supp Figures | 全部 |
| 第9个月 | 内部审阅 + 投稿 | 投Nature Communications |

---

*方案生成时间: 2026-04-02*
