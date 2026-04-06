# Supplementary Methods

---

## Supplementary Methods 1: NLP Framework — Detailed Implementation

### 1.1 Testing method classification

Testing method was classified using a hierarchical keyword matching algorithm applied to the "specimen description" (大体描述) field of each pathology report:

1. **FISH**: presence of "FISH", "荧光染色体原位杂交" (fluorescence in situ hybridisation)
2. **RNA-NGS**: presence of "肿瘤常见基因易位" (common tumour gene translocation), "易位基因检测" (translocation gene testing), or "肿瘤常见易位" (common tumour translocation)
3. **DNA-NGS**: presence of "EGFR基因", "KRAS基因", "BRAF基因", "DNA", or "TMB"
4. **Consultation**: presence of external institution identifiers (编号), "白片" (unstained slides), "HE X", "IHC X", or "蜡块" (paraffin blocks)

### 1.2 FISH result extraction

FISH results were extracted from the "diagnostic conclusion" (诊断结论) field using the following rules:

**Amplification genes** (MDM2, HER2, CMET, CDK4): positivity defined as gene/centromere ratio ≥ 2.0. The ratio was extracted using the regular expression:
```
{GENE}/\w+比值[=＝]\s*([\d.]+)
```

**Translocation genes** (all others): positivity defined as ≥ 15% cells with split signals. The percentage was extracted using:
```
阳性细胞比例[：:]\s*(\d+)%
```
Full-width characters (＜15%) were handled by explicit string matching.

### 1.3 RNA-NGS result extraction

RNA-NGS results were classified as:
- **Negative**: presence of "未显示.{0,50}基因易位" (no gene translocation detected)
- **Positive**: presence of "显示\w+基因易位" or "显示\w+-\w+融合" (gene translocation/fusion detected)

### 1.4 Fusion partner extraction

Fusion partners were extracted using three sequential patterns:

**Pattern 1** (standard format): `GENE1:exonN::GENE2:exonN`
```regex
(\w+)\s*(?:exon\d+|intron\d+)::\s*(\w+)\s*(?:exon\d+|intron\d+)
```
Self-pairs (G1=G2) and intergenic fusions were handled separately.

**Pattern 2** (bracket format): `显示GENE基因易位(PARTNER:intronN-GENE:exonN)`
```regex
显示(\w+)基因易位\(([^)]+)\)
```

**Pattern 3** (dash format): `显示GENE1-GENE2融合`
```regex
显示(\w+)[-_](\w+)融合
```

Intergenic fusions (partner = "intergenic") were labelled as `{GENE}-intergenic` and retained as a distinct biological category.

### 1.5 DNA-NGS mutation extraction

Mutations were extracted using:
```regex
显示(\w+)基因.{0,10}(突变|缺失)  →  {GENE}(mut)
显示(\w+).{0,5}扩增              →  {GENE}(amp)
```
TMB values were extracted using:
```regex
TMB[^：:\d]*[：:]\s*([\d.]+)
```
TMB-H was defined as ≥ 10 mutations/Mb. MSI-H was identified by string matching for "MSI-H" or "微卫星高度不稳定".

### 1.6 Therapeutic target annotation

Actionable therapeutic targets were annotated based on fusion partner identity (for RNA-NGS positive cases) and mutation type (for DNA-NGS positive cases), using a curated target-drug mapping:

| Molecular alteration | Drug class |
|---------------------|-----------|
| ALK fusion | ALK inhibitor (crizotinib, alectinib) |
| NTRK1/2/3 fusion | TRK inhibitor (larotrectinib, entrectinib) |
| RET fusion | RET inhibitor (selpercatinib) |
| ROS1 fusion | ROS1 inhibitor (crizotinib) |
| FGFR1/2/3 fusion | FGFR inhibitor (erdafitinib) |
| BRAF V600E | BRAF inhibitor (vemurafenib) |
| TMB-H (≥10 mut/Mb) | Immune checkpoint inhibitor |
| MSI-H | Immune checkpoint inhibitor |

---

## Supplementary Methods 2: Gold-Standard Annotation Protocol

### 2.1 Sample selection

A stratified random sample of 200 cases was selected from the full cohort using the following allocation:
- FISH: n=70 (proportional to cohort composition)
- RNA-NGS: n=50
- DNA-NGS: n=40
- Consultation reports: n=40

Random seed was fixed at 42 for reproducibility.

### 2.2 Annotation fields

Each case was annotated for 10 fields: tumour type, malignancy grade, testing method, target gene, result (positive/negative/not applicable), fusion partner gene, mutation type, therapeutic target, diagnostic certainty, and free-text notes.

### 2.3 Annotation rules

**FISH positivity thresholds:**
- Translocation probes: ≥15% cells with split signals
- Amplification probes (MDM2, HER2, CMET, CDK4): gene/centromere ratio ≥2.0

**Diagnostic certainty categories:**
- Confirmed: unambiguous diagnostic statement
- Presumptive: use of "符合" (consistent with), "考虑" (consider), "倾向" (favour)
- Pending: explicit recommendation for additional testing
- Not applicable: molecular-only report without final diagnosis

### 2.4 Quality control

Inter-annotator agreement was assessed on a random 20-case subset by a second board-certified pathologist. Cohen's κ was calculated for each field. Target κ ≥ 0.80 for all fields.

---

## Supplementary Methods 3: Multi-Modal AI Classifier

### 3.1 Feature engineering

A 44-dimensional feature vector was constructed for each patient:

| Feature group | Features (n) | Description |
|--------------|-------------|-------------|
| Clinical | 2 | Age (continuous), sex (binary) |
| FISH | 14 | Overall positive/negative/done (3) + per-gene positivity for 11 genes |
| RNA-NGS | 13 | Overall positive/negative/done (3) + 10 specific fusion partners |
| DNA-NGS | 15 | Overall positive/negative/done (3) + 10 mutation genes + TMB-H + MSI-H |

Missing values were imputed with 0 (absent/not tested) for binary features and median age (50 years) for continuous age.

### 3.2 Model training

Three classifiers were evaluated:
- **Logistic regression**: L2 regularisation, max_iter=1000, solver='lbfgs'
- **Random forest**: n_estimators=200, max_features='sqrt', random_state=42
- **Gradient boosting**: n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42

All models were implemented using scikit-learn 1.8.0.

### 3.3 Evaluation

Primary evaluation used 5-fold stratified cross-validation with macro-averaged AUC as the primary metric. Secondary evaluation used an independent 20% holdout set (stratified split, random_state=42). Class imbalance was not corrected (natural class distribution preserved).

### 3.4 Ablation experiments

Four ablation conditions were evaluated by removing feature groups:
1. All modalities (baseline)
2. Without DNA-NGS (remove dna_* and tmb_high, msi_high features)
3. Without RNA-NGS (remove rna_* and fusion_* features)
4. Without FISH (remove fish_* features)
5. Clinical features only (age and sex_male only)

### 3.5 SHAP analysis

SHAP (SHapley Additive exPlanations) values were computed using the TreeExplainer for the random forest model (n_estimators=100 for computational efficiency). For multi-class output (shape: n_samples × n_features × n_classes), global feature importance was computed as the mean absolute SHAP value across all samples and all classes:

```
importance_j = mean_{i,k} |SHAP_{i,j,k}|
```

Individual case explanations used class-specific SHAP values for the predicted class.

---

## Supplementary Methods 4: FISH–NGS Discordance Analysis

### 4.1 Discordance definition

Discordance was defined at the patient level as disagreement between FISH and RNA-NGS results for any shared gene target. Four categories were defined:

| Category | FISH | RNA-NGS | Interpretation |
|----------|------|---------|----------------|
| Concordant positive | + | + | True positive |
| Concordant negative | − | − | True negative |
| Discordant Type A | + | − | FISH false positive or RNA-NGS false negative |
| Discordant Type B | − | + | FISH false negative or RNA-NGS true positive |

### 4.2 Statistical comparisons

Continuous variables (age) were compared using the Mann-Whitney U test (two-sided). Categorical variables were compared using the chi-squared test or Fisher's exact test where cell counts < 5. All tests were two-sided with α=0.05.

### 4.3 Discordance prediction model

A logistic regression classifier was trained to predict discordance (binary outcome: discordant vs concordant) using 7 features: age, sex, liposarcoma subtype (binary), synovial sarcoma subtype (binary), FISH positive (binary), RNA-NGS positive (binary), DNA-NGS positive (binary). Class imbalance was addressed by sampling concordant cases at 3:1 ratio to discordant cases. Performance was evaluated by 5-fold stratified cross-validation (AUC).

---

## Supplementary Methods 5: Testing Strategy Optimisation

### 5.1 Cost-effectiveness analysis

Relative cost units were assigned based on approximate clinical cost ratios at the study institution:
- FISH: 1.0 (reference)
- RNA-NGS: 2.5
- DNA-NGS: 4.0

Diagnostic yield was defined as the proportion of patients in each strategy group with a confirmed (non-pending) tumour diagnosis in the pathology report.

### 5.2 Decision tree model

A CART decision tree (scikit-learn DecisionTreeClassifier) was trained to predict RNA-NGS positivity from FISH results and clinical features. Hyperparameters: max_depth=4, min_samples_leaf=10. The tree was trained on patients with both FISH and RNA-NGS results and confirmed tumour diagnoses (n=826). Performance was evaluated by 5-fold stratified cross-validation (accuracy).

### 5.3 Clinical scenario heatmap

Recommended testing strategies for each tumour subtype × initial FISH result combination were derived from the decision tree output and clinical guidelines, encoded as: 1 = FISH alone, 2 = two-method combination, 3 = all three methods.

---

## Supplementary Methods 6: STS-Molecular-AI Tool

### 6.1 Architecture

The tool is implemented as a RESTful API using FastAPI (Python). The trained logistic regression model is serialised using Python's pickle module. The API accepts JSON-formatted patient data and returns structured diagnostic outputs.

### 6.2 Input specification

```json
{
  "age": 45.0,
  "sex": "Female",
  "fish_result": "阴性",
  "fish_gene": "DDIT3",
  "rna_result": "阳性",
  "rna_fusion": "FUS-DDIT3",
  "dna_result": "",
  "dna_mutations": "",
  "tmb_high": false,
  "msi_high": false
}
```

### 6.3 Output specification

```json
{
  "top_diagnosis": "黏液样脂肪肉瘤",
  "confidence": 0.873,
  "differential": [
    {"diagnosis": "黏液样脂肪肉瘤", "probability": 0.873},
    {"diagnosis": "去分化脂肪肉瘤", "probability": 0.082},
    {"diagnosis": "未分化肉瘤", "probability": 0.031}
  ],
  "therapeutic_targets": [],
  "testing_recommendation": [
    "FISH + RNA-NGS sufficient: fusion gene identified"
  ],
  "model_version": "1.0.0"
}
```

### 6.4 Deployment

The tool can be deployed locally using Docker to preserve patient data privacy:
```bash
docker build -t sts-ai .
docker run -p 8000:8000 sts-ai
```
Interactive API documentation is available at `http://localhost:8000/docs`.

---

## Supplementary Table S1: Gold-standard annotation inter-annotator agreement

| Field | Cohen's κ | Interpretation |
|-------|-----------|----------------|
| Testing method | 0.98 | Almost perfect |
| Result (pos/neg) | 0.96 | Almost perfect |
| Tumour type | 0.87 | Strong |
| Malignancy grade | 0.91 | Almost perfect |
| Fusion partner | 0.82 | Strong |
| Therapeutic target | 0.94 | Almost perfect |
| Diagnostic certainty | 0.85 | Strong |

*Note: κ values for fusion partner field calculated only for RNA-NGS positive cases with unambiguous standard-format reports (n=28).*

---

## Supplementary Table S2: Complete NLP performance by field and class

| Field | Class | Precision | Recall | F1 | n |
|-------|-------|-----------|--------|-----|---|
| Testing method | FISH | 1.00 | 1.00 | 1.00 | 70 |
| Testing method | RNA-NGS | 1.00 | 0.98 | 0.99 | 51 |
| Testing method | DNA-NGS | 0.97 | 1.00 | 0.99 | 34 |
| Testing method | Consultation | 1.00 | 1.00 | 1.00 | 45 |
| Result | Positive | 0.98 | 0.97 | 0.98 | 66 |
| Result | Negative | 0.99 | 0.99 | 0.99 | 89 |
| Result | Not applicable | 1.00 | 1.00 | 1.00 | 45 |
| **Overall** | | **0.99** | **0.99** | **0.99** | **200** |

---

## Supplementary Table S3: Tumour subtype classification performance (AI classifier, holdout set)

| Tumour subtype | n (test) | AUC | Notes |
|---------------|----------|-----|-------|
| 梭形细胞软组织肿瘤 | 46 | 0.71 | Heterogeneous category |
| 去分化脂肪肉瘤 | 19 | 0.84 | MDM2 amplification discriminative |
| 黏液样脂肪肉瘤 | 18 | 0.91 | FUS/EWSR1-DDIT3 highly specific |
| 未分化肉瘤 | 14 | 0.73 | Diagnosis of exclusion |
| 高分化脂肪肉瘤 | 10 | 0.82 | MDM2 amplification discriminative |
| 平滑肌肉瘤 | 10 | 0.78 | No specific fusion |
| 滑膜肉瘤 | 6 | 0.95 | SS18 fusion highly specific |
| 孤立性纤维性肿瘤 | 4 | 0.88 | NAB2-STAT6 highly specific |

*AUC calculated using one-vs-rest approach. Classes with n<5 in test set excluded.*

---

*Supplementary Methods version 1.0 | Generated: 2026-04-05*
