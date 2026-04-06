# Submission Checklist — Nature Communications

## Manuscript
- [x] Title updated: "RNA-NGS Substantially Extends the Diagnostic Reach of FISH..."
- [x] Abstract: all key numbers present (n=1,489, 28.3%, AUC=0.780, 95% CI)
- [x] Introduction: research gap clearly stated
- [x] Methods: IRB approval number placeholder added
- [x] Methods: statistical analysis section includes multiple comparison correction
- [x] Methods: Bootstrap CI method described (n=300 resamples)
- [x] Results: all figures referenced in order
- [x] Discussion: limitations section present
- [x] References: 12 references (need to expand to 40-50 for submission)
- [ ] Author contributions: to be completed
- [ ] Competing interests: to be completed
- [ ] Acknowledgements: to be completed

## Figures (latest versions)
- [x] Figure 1 v3: CONSORT flowchart + colour-coded tumour subtypes
- [x] Figure 2 v4: Sankey diagram + subtype-specific FISH rates
- [x] Figure 3 v3: NLP performance (fusion partner field clarified)
- [x] Figure 4 v4: Bootstrap 95% CI + confusion matrix + per-class AUC
- [x] Figure 5 v5: Stacked bar + reclassified discordance
- [x] Figure 6 v3: Corrected leiomyosarcoma recommendation
- [x] Figure 7 v4: Simulated annotation + API example + trabectedin
- [x] All figures: 300 dpi TIFF + PNG preview
- [x] All figures: RGB (no transparency)
- [x] All figures: colour-blind friendly palette (Wong 2011)

## Extended Data Figures
- [x] ExtFig1: Extended cohort characteristics
- [x] ExtFig2: Molecular testing landscape
- [x] ExtFig3: NLP details + per-class ROC
- [x] ExtFig4: AI model details (SHAP + calibration)
- [x] ExtFig5: Discordance extended analysis

## Supplementary Materials
- [x] Supplementary Methods (6 sections)
- [x] Supplementary Table S1: Inter-annotator agreement
- [x] Supplementary Table S2: NLP performance by field
- [x] Supplementary Table S3: Per-class classifier performance
- [x] Figure Legends (complete, panel-by-panel)

## Data & Code
- [x] Structured patient dataset (de-identified CSV)
- [x] NLP extraction code (nlp_model.py)
- [x] AI classifier code (figure4.py, editor_fixes.py)
- [x] Online tool backend (sts_ai_app/app.py)
- [x] Trained model (sts_ai_model.pkl)
- [x] GitHub repository: https://github.com/wuliydc/STS-Molecular-AI
- [ ] DOI for dataset: to be registered (Zenodo)

## Outstanding Items Before Submission
1. [ ] Expand references to 40-50 (add recent STS molecular diagnostics papers)
2. [ ] Add IRB approval number — **[ACTION REQUIRED before submission]**
   - Institution: NCC/NCRCC, Chinese Academy of Medical Sciences (中国医学科学院肿瘤医院)
   - Apply to: 医院伦理委员会 → 回顾性研究豁免/快速审查
   - Manuscript placeholder: `[IRB-PENDING]` (搜索替换即可)
   - 表单参考: http://www.cicams.ac.cn/ethics
3. [ ] Create GitHub repository and update URL in manuscript
4. [ ] Register dataset DOI (Zenodo or Figshare)
5. [ ] External validation cohort (if available from collaborating institution)
6. [ ] Native English editing (manuscript currently in English but may need polish)
7. [ ] Cover letter (see cover_letter_template.md)

## Target Journal
- Primary: Nature Communications (IF ~17)
- Secondary: EBioMedicine / npj Precision Oncology
- Submission type: Article
- Word count target: 4,000-5,000 words (main text)
