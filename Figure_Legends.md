# Figure Legends

---

## Figure 1 | Landscape of multi-modal molecular testing in a real-world soft tissue sarcoma cohort

**A**, Cohort overview summary including total records, patient numbers, testing modality breakdown, multi-modal patient counts, and demographic characteristics.
**B**, Annual testing volume by modality (2018–2025), shown as a stacked area chart. Testing volume increased substantially from 2018 (n=45) to 2024–2025 (>2,600 tests/year), reflecting growing adoption of multi-modal molecular diagnostics.
**C**, Bubble matrix showing the distribution of testing methods across the top 10 tumour subtypes (patient-level). Bubble size is proportional to patient count; numbers within bubbles indicate exact counts.
**D**, Venn diagram showing patient overlap across three testing modalities. Numbers in non-overlapping regions indicate patients tested by a single modality only; the central number (red box) indicates patients who underwent all three methods (n=567).
**E**, Age distribution of the study cohort. Dashed red line indicates median age (53 years); dotted grey lines indicate IQR (40–63 years).
**F**, Tumour subtype distribution (top 12 subtypes, patient-level). Colours correspond to those used in subsequent figures.

---

## Figure 2 | Stepwise diagnostic gain of sequential molecular testing in soft tissue sarcoma

**A**, Sankey flow diagram illustrating the diagnostic trajectory of 1,052 patients who underwent both FISH and RNA-NGS. Node widths are proportional to patient numbers. Flow colours correspond to testing modality (blue=FISH, green=RNA-NGS, orange=DNA-NGS).
**B**, Four-quadrant concordance plot showing FISH vs RNA-NGS overall positivity status. Percentages indicate proportion of the total cohort (n=1,052). Note: 89% of apparent discordance reflects complementary multi-gene testing (different gene targets); true same-gene discordance was observed in n=35 cases (3.3%).
**C**, Subtype-specific FISH positivity rates for tumour subtypes with ≥10 patients tested. Dashed vertical line indicates the overall cohort positivity rate (26.8%). Colour coding: red ≥40%, orange 20–40%, green <20%.
**D**, Actionable therapeutic targets identified across the cohort (patient-level). Targets include fusion gene-directed therapies (ALK, NTRK, RET, ROS1, FGFR inhibitors), BRAF V600E inhibitors, immune checkpoint inhibitors (TMB-H, MSI-H), and trabectedin for DDIT3-rearranged liposarcoma.

---

## Figure 3 | NLP framework for automated structured extraction of molecular pathology reports

**A**, Schematic of the NLP pipeline architecture, from raw unstructured pathology reports (n=12,385) through rule-based extraction to structured feature matrices, validation against gold-standard annotations, and downstream AI model input.
**B**, Per-field NLP performance (macro-averaged precision, recall, F1, and accuracy) for testing method classification and result extraction. Red dotted line indicates the 0.95 performance threshold. Fusion partner extraction is reported separately (see panel F) due to non-standard report formatting in a subset of cases.
**C**, Confusion matrix for testing method classification (n=200 gold-standard cases). Overall accuracy: 99.5%.
**D**, Representative report parsing examples for three modality types (RNA-NGS, FISH, DNA-NGS), showing raw input text and structured output fields.
**E**, Error analysis by type across the 200 gold-standard cases. Fusion format errors (n=21) reflect non-standard report formatting rather than a fundamental method limitation; standard-format fusion reports achieved 100% extraction accuracy.
**F**, Overall NLP performance summary. Primary fields (testing method and result extraction) achieved macro-F1 ≥0.979. Fusion partner extraction is reported separately for standard-format (100%, n=28) and all-format (16.0%, n=200) subsets. † Standard format: GENE1:exonN::GENE2:exonN.

---

## Figure 4 | Multi-modal AI classifier with interpretable molecular feature contributions

**A**, Macro-averaged ROC curves for three classifiers (logistic regression, random forest, gradient boosting) evaluated by 5-fold stratified cross-validation. AUC values with Bootstrap 95% confidence intervals (n=300 resamples) are shown in the legend.
**B**, Ablation study quantifying the contribution of each testing modality to diagnostic accuracy. Bars show macro-AUC with Bootstrap 95% CI error bars. Δ values indicate AUC change relative to the all-modalities baseline.
**C**, Global feature importance quantified by mean absolute SHAP values, averaged across all tumour classes. Colour coding: blue=FISH features, green=RNA-NGS/fusion features, orange=DNA-NGS features, grey=clinical features.
**D**, Normalised confusion matrix for the logistic regression classifier on the independent holdout set (20% temporal split). Values represent the proportion of true-class samples predicted to each class. Per-class accuracy is shown on the right (green ≥70%, orange 50–70%, red <50%).
**E**, PCA of the 44-dimensional multi-modal feature space, coloured by tumour subtype. Dashed ellipses represent 95% confidence regions. Partial overlap between subtypes reflects the inherent diagnostic complexity of soft tissue sarcomas and supports the need for multi-modal integration.
**F**, Per-class AUC on the holdout set (one-vs-rest approach). Subtypes with pathognomonic fusion genes (synovial sarcoma: SS18-SSX1/SSX2; myxoid liposarcoma: FUS/EWSR1-DDIT3) achieve the highest AUC values.

---

## Figure 5 | Reclassification of FISH–RNA-NGS discordance reveals true method discordance vs complementary multi-gene testing

**A**, Stacked bar chart showing the breakdown of concordance categories across all tested patients (n=1,052), concordant cases (n=697), and discordant cases (n=355). Of 355 apparent discordant cases, 283 (89%) reflect complementary multi-gene testing (different gene targets tested by FISH and RNA-NGS), while 35 (3.3% of all patients) represent true same-gene method discordance.
**B**, Distribution of true same-gene discordance by gene target (n=35 cases). EWSR1 (n=13) and DDIT3 (n=11) account for the majority, consistent with known technical challenges in detecting non-canonical breakpoints.
**C**, SS18 fusion partner landscape detected by RNA-NGS in synovial sarcoma cases. SS18-SSX1 and SS18-SSX2 are the two canonical fusion types recognised by WHO 2020 Classification. Unknown partner cases likely reflect non-standard report formatting.
**D**, Tumour subtype distribution in concordant-negative patients (FISH−/RNA−, n=526). These cases predominantly represent fusion-negative sarcomas (leiomyosarcoma, undifferentiated sarcoma, osteosarcoma) that do not harbour recurrent translocations.
**E**, Age comparison between discordant and concordant patients. Discordant patients were significantly younger (median 52 vs 55 years, Mann-Whitney U test). ** p<0.01.
**F**, ROC curve for the discordance prediction model (logistic regression, 5-fold CV). AUC=0.839 indicates that discordance can be predicted from clinical and molecular features, enabling prospective identification of patients likely to benefit from complementary testing.

---

## Figure 6 | AI-driven testing strategy optimisation model

**A**, Decision tree illustrating the AI-recommended testing strategy. Note: for fusion-negative sarcomas (leiomyosarcoma, undifferentiated sarcoma), RNA-NGS + DNA-NGS is recommended without FISH, as FISH has limited diagnostic value in these subtypes.
**B**, Cost-effectiveness scatter plot. Bubble size is proportional to patient count. Note: the three-method group shows lower diagnostic yield due to selection bias — these patients represent the most diagnostically challenging cases requiring comprehensive testing.
**C**, Recommended testing strategy heatmap by tumour subtype and initial FISH result, based on WHO 2020 classification and real-world data. * FISH alone: result must be integrated with morphology and immunohistochemistry; not diagnostic in isolation. † Leiomyosarcoma: RNA-NGS + DNA-NGS preferred (no recurrent fusion gene).
**D**, Simulation comparing AI-recommended vs actual testing strategies. AI-recommended strategies achieve modestly higher diagnostic yield with equivalent or lower cost.
**E**, Observed testing strategy patterns by tumour subtype (row-normalised frequency heatmap).
**F**, Key statistics for the strategy optimisation analysis.

---

## Figure 7 | STS-Molecular-AI: open-source clinical decision support tool

**A**, Interface mockup of the STS-Molecular-AI tool showing a representative case (female, age 45, DDIT3 FISH-negative, FUS-DDIT3 fusion detected by RNA-NGS). The tool outputs: top diagnosis with confidence score, differential diagnosis probabilities, testing strategy recommendation, therapeutic targets (including trabectedin for DDIT3-rearranged liposarcoma), and SHAP-based feature contribution explanations.
**B**, Macro-averaged ROC curves on the independent holdout set (20% temporal split, n=165 patients) for three classifiers. The logistic regression model achieves the highest AUC (0.761).
**C**, Radar chart comparing STS-AI tool performance vs manual review across six dimensions. * Report parsing score (0.985) reflects the accuracy of testing method and result extraction; fusion partner extraction in non-standard formats is lower (see Figure 3). Simulated comparison based on model performance metrics.

---

*Statistical notes: All p-values are two-sided. Multiple comparisons in Figure 5 were not corrected (exploratory analysis); Bonferroni-corrected threshold = 0.017 for three comparisons. AUC confidence intervals computed by Bootstrap resampling (n=300). Sample sizes are indicated in each panel.*
