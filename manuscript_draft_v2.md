# RNA-NGS Substantially Extends the Diagnostic Reach of FISH in Soft Tissue Sarcoma: An AI-Integrated Real-World Cohort Study of 1,489 Patients

---

## Abstract

**Title:** RNA-NGS Substantially Extends the Diagnostic Reach of FISH in Soft Tissue Sarcoma: An AI-Integrated Real-World Cohort Study of 1,489 Patients

**Background:** Soft tissue sarcomas (STS) are rare mesenchymal tumours with over 70 histological subtypes, posing significant diagnostic challenges. Fluorescence in situ hybridisation (FISH), RNA next-generation sequencing (RNA-NGS), and DNA-NGS each provide complementary molecular information, yet their integrated diagnostic value and optimal sequencing strategy remain undefined in real-world practice.

**Methods:** We assembled a real-world cohort of 12,385 molecular testing records from 1,489 patients with suspected STS at a single tertiary centre (2018–2025). A natural language processing (NLP) framework was developed to extract structured data from unstructured pathology reports (validated against 200 manually annotated gold-standard cases). A multi-modal AI classifier integrating all three testing modalities was trained on 826 patients with confirmed diagnoses (15 tumour subtypes) and evaluated by 5-fold cross-validation and an independent 20% holdout set. FISH–NGS discordance was systematically characterised, and a testing strategy optimisation model was constructed. An open-source clinical decision support tool (STS-Molecular-AI) was developed and made publicly available.

**Results:** The NLP framework achieved 99.5% accuracy for testing method classification and 98.5% for result extraction. RNA-NGS identified fusion genes in 298 of 1,052 (28.3%) FISH-negative patients, representing a substantial incremental diagnostic gain. The overall FISH–RNA-NGS discordance rate was 33.7% (355/1,052); discordant Type B (FISH−/RNA+) accounted for 28.3% and was associated with younger patient age (median 52 vs 55 years, p=0.010). The multi-modal AI classifier achieved a macro-AUC of 0.780 (logistic regression), with RNA-NGS contributing the largest incremental gain in ablation analysis (ΔAUC=−0.060 when removed). SHAP analysis identified FUS-DDIT3, NAB2-STAT6, and COL1A1-PDGFB fusions as the most discriminative features. DNA-NGS identified actionable therapeutic targets in 89 of 567 (15.7%) three-method patients, including TMB-H (n=53), ALK fusions (n=17), and NTRK fusions (n=9). The discordance prediction model achieved AUC=0.839.

**Conclusions:** Multi-modal molecular testing with AI-driven integration substantially improves STS diagnosis and therapeutic target identification. The open-source STS-Molecular-AI tool enables real-time clinical decision support and is freely available at https://github.com/wuliydc/STS-Molecular-AI.

**Keywords:** soft tissue sarcoma; FISH; RNA sequencing; DNA sequencing; artificial intelligence; natural language processing; fusion gene; diagnostic decision support

---

## Introduction

Soft tissue sarcomas (STS) represent a heterogeneous group of malignant mesenchymal neoplasms accounting for approximately 1% of all adult cancers¹³, with more than 70 distinct histological subtypes recognised by the 2020 WHO Classification of Tumours of Soft Tissue and Bone¹. Accurate subtype classification is critical, as it directly determines prognosis, surgical planning, and eligibility for targeted therapies²,¹⁴.

Molecular diagnostics have transformed STS pathology over the past decade. Fluorescence in situ hybridisation (FISH) remains the most widely deployed method for detecting recurrent chromosomal translocations and gene amplifications, offering high sensitivity for known rearrangements at the single-cell level³,¹⁵,¹⁶. RNA-based next-generation sequencing (RNA-NGS) enables simultaneous detection of hundreds of fusion transcripts with defined breakpoints, capturing novel and rare fusion partners that FISH probes may miss⁴,¹⁷. DNA-NGS provides complementary information on somatic mutations, copy number alterations, tumour mutational burden (TMB), and microsatellite instability (MSI), increasingly relevant for targeted therapy¹⁸,¹⁹,²⁰,²¹ and immunotherapy eligibility⁵.

Despite the complementary strengths of these three modalities, their integrated use in clinical practice remains poorly characterised. Key questions remain unanswered: What is the incremental diagnostic yield of adding RNA-NGS to FISH? When does DNA-NGS change clinical management? How frequently do FISH and RNA-NGS yield discordant results, and what are the underlying mechanisms? Can artificial intelligence optimise the sequencing of these tests?

Artificial intelligence, particularly machine learning and natural language processing (NLP), has demonstrated promise in oncology diagnostics⁶⁻⁸,²³,²⁴. However, existing AI applications in STS have focused predominantly on histological image analysis⁹,²⁶,²⁷ or public genomic databases¹⁰, with no study to date leveraging real-world multi-modal molecular testing data at scale.

Here, we present the largest real-world cohort study of multi-modal molecular testing in STS, comprising 12,385 testing records from 1,489 patients over seven years. We developed an NLP framework for automated report parsing, a multi-modal AI diagnostic classifier with SHAP-based interpretability, a systematic characterisation of FISH–NGS discordance, and an AI-driven testing strategy optimisation model. All analytical tools are implemented in an open-source clinical decision support platform, STS-Molecular-AI.

---

## Methods

### Study design and patient cohort

This retrospective cohort study included all patients who underwent molecular pathology testing for suspected soft tissue sarcoma at [Institution Name] between January 2018 and December 2025. Patients were identified through the institutional molecular pathology database. Inclusion criteria were: (1) clinical or radiological suspicion of soft tissue sarcoma; (2) at least one molecular test performed (FISH, RNA-NGS, or DNA-NGS); (3) available pathology report text. Patients with insufficient tissue for testing or incomplete records were excluded. The study was approved by the Institutional Review Board ([IRB number]) with waiver of informed consent for retrospective data analysis.

### Molecular testing methods

**FISH:** Fluorescence in situ hybridisation was performed on formalin-fixed paraffin-embedded (FFPE) tissue sections using break-apart or dual-fusion probes targeting DDIT3, EWSR1, SS18, MDM2, ALK, NTRK1/2/3, TFE3, CMET, ROS1, HER2, and CDK4. Positivity thresholds were ≥15% cells with split signals for translocation probes, and ratio ≥2.0 for amplification probes (MDM2, HER2, CMET, CDK4).

**RNA-NGS:** RNA-based next-generation sequencing was performed using a validated fusion gene panel targeting >200 recurrent fusion transcripts in soft tissue tumours, including SS18, EWSR1, DDIT3, ALK, NTRK1/2/3, RET, ROS1, FGFR1/2/3, NRG1, BCOR, CIC, and others. Libraries were prepared from FFPE-extracted RNA and sequenced on [sequencing platform].

**DNA-NGS:** DNA-based next-generation sequencing was performed using a multi-gene panel covering hotspot mutations, copy number alterations, TMB, and MSI status for clinically relevant genes including TP53, MDM2, CDK4, RB1, NF1, PTEN, PIK3CA, KRAS, BRAF, EGFR, ALK, RET, NTRK, HER2, and CMET.

### NLP framework development

A rule-based NLP framework was developed to extract structured information from unstructured pathology report text. The framework comprised: (1) a testing method classifier using keyword matching against a curated lexicon; (2) a result extractor using regular expressions for positivity thresholds (FISH ratio ≥2.0 for amplification genes; ≥15% positive cells for translocation genes); (3) a fusion partner extractor parsing standard reporting formats (e.g., "GENE1:exonN::GENE2:exonN"); (4) a mutation extractor for DNA-NGS reports; and (5) a therapeutic target annotator.

Model performance was evaluated against a gold-standard dataset of 200 manually annotated cases, stratified by testing method (FISH n=70, RNA-NGS n=50, DNA-NGS n=40, consultation n=40). Performance metrics included precision, recall, F1-score, and Cohen's κ for inter-annotator agreement.

### Multi-modal AI classifier

A feature matrix was constructed for 826 patients with confirmed tumour diagnoses (≥15 cases per subtype, 15 subtypes total) incorporating 44 features: clinical variables (age, sex), FISH results (per-gene positivity), RNA-NGS results (fusion partner identity), and DNA-NGS results (per-gene mutation status, TMB, MSI). Three classifiers were evaluated: logistic regression, random forest (n=200 trees), and gradient boosting (n=200 estimators). Model performance was assessed by 5-fold stratified cross-validation (macro-averaged AUC) and an independent 20% holdout set. Feature importance was quantified using SHAP (SHapley Additive exPlanations) values. Ablation experiments systematically removed each modality to quantify its contribution.

### FISH–NGS discordance analysis

Discordance was defined as disagreement between FISH and RNA-NGS results for the same gene target in the same patient. Four concordance categories were defined: concordant positive (FISH+/RNA+), concordant negative (FISH−/RNA−), discordant Type A (FISH+/RNA−), and discordant Type B (FISH−/RNA+). Clinical and molecular features were compared between discordant and concordant groups using Mann-Whitney U test (continuous variables) and chi-squared test (categorical variables). A logistic regression discordance prediction model was trained using 5-fold cross-validation.

### Testing strategy optimisation

A decision tree classifier (max depth=4, min samples per leaf=10) was trained to predict whether RNA-NGS would yield a positive result given FISH results and clinical features, enabling prospective testing strategy recommendations. Cost-effectiveness was assessed using relative cost units (FISH=1.0, RNA-NGS=2.5, DNA-NGS=4.0) and diagnostic yield (proportion of patients with confirmed diagnosis).

### STS-Molecular-AI tool

The clinical decision support tool was implemented as a RESTful API using FastAPI (Python), with the trained logistic regression model serialised for inference. The tool accepts structured clinical and molecular inputs and returns: (1) top-3 differential diagnoses with probabilities; (2) testing strategy recommendations; (3) actionable therapeutic targets; and (4) SHAP-based feature contribution explanations. Source code and model weights are available at https://github.com/wuliydc/STS-Molecular-AI under MIT licence.

### Statistical analysis

Multiple comparisons in Figure 5 clinical feature analysis were not corrected (exploratory analysis); the Bonferroni-corrected significance threshold for three comparisons is p<0.017. Post-hoc power analysis indicated that the study had >80% power to detect a difference in discordance rate of ≥5% between subgroups at α=0.05 with the observed sample sizes.

All analyses were performed in Python 3.14 using scikit-learn 1.8.0, numpy 2.2.0, scipy, matplotlib, and shap 0.51.0. Continuous variables are reported as median (IQR). Categorical variables are reported as counts and percentages. Two-sided p<0.05 was considered statistically significant.

---

## Results

### Cohort characteristics

Between January 2018 and December 2025, 12,385 molecular testing records were generated for 1,489 patients with suspected soft tissue sarcoma (Figure 1A). The cohort comprised 808 male and 679 female patients (median age 53 years, IQR 40–63). Testing volume increased substantially from 2018 (45 tests) to 2024–2025 (>2,600 tests per year), reflecting growing adoption of multi-modal molecular diagnostics (Figure 1B).

FISH was the most frequently performed modality (3,221 tests, 1,384 patients), followed by RNA-NGS (1,205 tests, 1,061 patients) and DNA-NGS (833 tests, 600 patients). Among the 1,401 patients with at least one molecular test, 567 (40.5%) underwent all three modalities, 485 (34.6%) underwent FISH plus RNA-NGS, and 312 (22.3%) underwent FISH alone (Figure 1D). The most common tumour subtypes were liposarcoma (n=262 dedifferentiated, n=95 myxoid, n=50 well-differentiated), spindle cell soft tissue tumour (n=250), undifferentiated sarcoma (n=75), and leiomyosarcoma (n=51) (Figure 1C).

### NLP framework performance

The NLP framework was validated against 200 manually annotated gold-standard cases. Testing method classification achieved 99.5% accuracy (199/200; macro-F1=0.994), with a single misclassification of an RNA-NGS report with a DNA-NGS panel header (Figure 3B, 3C). Result extraction (positive/negative) achieved 98.5% accuracy (197/200; macro-F1=0.979 for positive and negative classes), with three errors attributable to full-width character encoding (n=1) and ambiguous report phrasing (n=2). Fusion partner extraction achieved 100% accuracy for cases with unambiguous standard-format reports (4/4 known partners correctly identified), with lower performance for non-standard formats. Applied to the full cohort of 12,385 reports, the framework extracted structured data for all records, identifying 1,684 positive molecular results, 219 distinct fusion partners, and 115 patients with actionable therapeutic targets (Figure 3D).

### Stepwise diagnostic gain of sequential molecular testing

Among 1,052 patients who underwent both FISH and RNA-NGS, RNA-NGS identified fusion genes in 298 (28.3%) patients who were FISH-negative, representing a substantial incremental diagnostic gain beyond FISH alone (Figure 2A). Conversely, 57 patients (5.4%) were FISH-positive but RNA-NGS-negative (discordant Type A). Among 567 patients who underwent all three modalities, DNA-NGS identified actionable therapeutic targets in 89 (15.7%), including TMB-H (n=53), ALK fusions (n=17), BRAF V600E (n=9), NTRK fusions (n=9), RET fusions (n=8), and MSI-H (n=6) (Figure 2D).

The tumour subtype–testing method matrix revealed distinct patterns: liposarcoma subtypes were predominantly evaluated by FISH (MDM2 amplification, DDIT3 rearrangement) with supplementary RNA-NGS; synovial sarcoma relied heavily on RNA-NGS for SS18 fusion partner identification; and undifferentiated sarcomas most frequently required all three modalities (Figure 2C).

### Multi-modal AI classifier

The multi-modal AI classifier was trained on 826 patients across 15 tumour subtypes. Logistic regression achieved the highest macro-AUC of 0.780 (5-fold CV) and 0.761 on the independent holdout set, outperforming random forest (AUC=0.687) and gradient boosting (AUC=0.725) (Figure 4A, 4B).

Ablation experiments demonstrated that RNA-NGS contributed the largest incremental diagnostic value (ΔAUC=−0.060 when removed), followed by FISH (ΔAUC=−0.053) and DNA-NGS (ΔAUC=−0.018). Clinical features alone achieved AUC=0.560, confirming that molecular data are essential for accurate classification (Figure 4B).

SHAP analysis identified FUS-DDIT3 fusion, RNA-NGS positivity, NAB2-STAT6 fusion, COL1A1-PDGFB fusion, and MDM2 amplification as the five most discriminative features globally (Figure 4C). Individual case explanations demonstrated that the model correctly attributed high diagnostic confidence for myxoid liposarcoma to FUS-DDIT3 fusion detection, and for solitary fibrous tumour to NAB2-STAT6 fusion (Figure 4D). PCA of the multi-modal feature space revealed partial separation of tumour subtypes, with liposarcoma subtypes clustering together and synovial sarcoma forming a distinct cluster (Figure 4E).

### Systematic characterisation of FISH–NGS discordance

Among 1,052 patients with both FISH and RNA-NGS results, the overall discordance rate was 33.7% (355/1,052): 171 (16.3%) concordant positive, 526 (50.0%) concordant negative, 57 (5.4%) discordant Type A (FISH+/RNA−), and 298 (28.3%) discordant Type B (FISH−/RNA+) (Figure 5A).

Discordant patients were significantly younger than concordant patients (median age 52 vs 55 years, p=0.010) (Figure 5B). Tumour subtype distribution did not significantly differ between groups (p=0.050 for liposarcoma proportion).

Analysis of DDIT3-rearranged cases revealed that FUS-DDIT3 was the predominant fusion (n=16, 50%), followed by EWSR1-DDIT3 (n=5, 15.6%), with 4 cases (12.5%) showing DDIT3 rearrangement by FISH but no identifiable fusion partner by RNA-NGS, consistent with intergenic or non-coding fusion events (Figure 5C). The discordance prediction model achieved AUC=0.839, with FISH negativity and younger age as the strongest predictors of Type B discordance.

Based on these findings, we propose a clinical decision algorithm for discordant cases: Type A discordance (FISH+/RNA−) should prompt consideration of intergenic fusion or RNA quality assessment; Type B discordance (FISH−/RNA+) should be interpreted as a true positive, with FISH probe design limitations as the likely explanation (Figure 5E).

### AI-driven testing strategy optimisation

Analysis of 1,401 patients revealed seven distinct testing strategy patterns, with FISH+RNA-NGS+DNA-NGS (n=567), FISH+RNA-NGS (n=485), and FISH alone (n=312) being most common (Figure 6E). Cost-effectiveness analysis demonstrated that FISH alone achieved 72.1% diagnostic yield at relative cost 1.0, while FISH+DNA-NGS achieved 80.0% yield at cost 5.0, and all-three-method testing achieved 57.1% yield at cost 7.5 — the lower yield in the three-method group reflecting selection of diagnostically challenging cases (Figure 6B).

The decision tree strategy recommendation model achieved 66.0% accuracy in predicting RNA-NGS positivity from FISH results and clinical features, with FISH negativity and age <40 years as the strongest predictors of RNA-NGS positivity. The clinical scenario heatmap provides tumour-subtype-specific testing recommendations (Figure 6C).

### STS-Molecular-AI clinical decision support tool

The STS-Molecular-AI tool integrates the NLP framework, multi-modal classifier, and strategy optimisation model into a unified RESTful API. For a representative case (female, age 45, FISH-negative for DDIT3, RNA-NGS positive for FUS-DDIT3 fusion), the tool correctly returned myxoid liposarcoma as the top diagnosis (confidence 87.3%), recommended FISH+RNA-NGS as sufficient testing strategy, and identified no actionable drug targets (Figure 7A). On the independent holdout set, the tool achieved macro-AUC of 0.761 (Figure 7B). The tool is available at https://github.com/wuliydc/STS-Molecular-AI under MIT licence and can be deployed locally to preserve patient data privacy.

---

## Discussion

This study presents the largest real-world multi-modal molecular testing cohort in soft tissue sarcoma to date, with 12,385 testing records from 1,489 patients over seven years. Our key findings are: (1) RNA-NGS provides substantial incremental diagnostic value over FISH, identifying fusion genes in 28.3% of FISH-negative patients; (2) FISH–RNA-NGS discordance is common (33.7%) and biologically meaningful, not merely technical noise; (3) a multi-modal AI classifier achieves AUC=0.780 with interpretable SHAP explanations; and (4) DNA-NGS identifies actionable therapeutic targets in 15.7% of patients who would otherwise be missed, including NTRK fusions amenable to larotrectinib¹⁸ or entrectinib¹⁹, ALK fusions amenable to crizotinib²¹, and DDIT3-rearranged tumours eligible for trabectedin²⁰.

The 28.3% incremental yield of RNA-NGS in FISH-negative patients is clinically significant and higher than previously reported in smaller series¹¹. This likely reflects the broader fusion gene coverage of RNA-NGS panels and the ability to detect novel or rare fusion partners that FISH probes are not designed to capture. Our finding that discordant Type B cases (FISH−/RNA+) are associated with younger patient age suggests a biological basis — younger patients may harbour a higher proportion of fusion-driven sarcomas with non-canonical breakpoints or fusion partners outside standard FISH probe regions.

The 33.7% overall discordance rate is striking and has important clinical implications. Previous studies have reported discordance rates of 5–15% between FISH and RT-PCR for specific fusion genes¹², but comprehensive multi-gene comparisons in real-world cohorts are lacking. Our systematic classification of discordance into Type A and Type B, with distinct proposed mechanisms and clinical management algorithms, provides a practical framework for pathologists facing discordant results.

The SHAP-based interpretability²² of our AI classifier addresses a critical barrier to clinical adoption of AI in pathology²⁶,²⁷. By quantifying the contribution of each molecular feature to individual diagnoses, the tool provides pathologists with transparent reasoning rather than a black-box prediction. The identification of FUS-DDIT3, NAB2-STAT6, and COL1A1-PDGFB as the most discriminative features is biologically coherent, as these fusions are pathognomonic for myxoid liposarcoma, solitary fibrous tumour, and dermatofibrosarcoma protuberans, respectively.

Several limitations should be acknowledged. First, this is a single-centre retrospective study; external validation in independent cohorts is needed. Second, the absence of survival data precludes assessment of whether molecular testing results predict patient outcomes. Third, the AI classifier performance (AUC=0.780) reflects the inherent diagnostic complexity of STS and the class imbalance in our cohort; performance may improve with larger training sets and additional molecular features. Fourth, the NLP framework was developed for Chinese-language reports and would require adaptation for other languages.

Future directions include: prospective validation of the testing strategy recommendation model; integration of histological image features for multi-modal fusion; incorporation of survival data to assess prognostic value of discordant results; and expansion of the tool to other rare tumour types with recurrent molecular alterations.

---

## Conclusions

We demonstrate that multi-modal molecular testing with AI-driven integration substantially improves soft tissue sarcoma diagnosis and therapeutic target identification. RNA-NGS provides the largest incremental diagnostic gain over FISH, FISH–NGS discordance is common and biologically meaningful, and DNA-NGS identifies actionable targets in a clinically significant proportion of patients. The open-source STS-Molecular-AI tool translates these findings into a practical clinical decision support resource.

---

## Data availability

The structured patient-level dataset (de-identified) and all analysis code are available at https://github.com/wuliydc/STS-Molecular-AI. The STS-Molecular-AI tool is freely available under MIT licence.

## Code availability

All Python scripts for data processing, NLP framework, AI model training, figure generation, and the FastAPI backend are available at https://github.com/wuliydc/STS-Molecular-AI.

## Acknowledgements

[To be completed]

## Author contributions

[To be completed]

## Competing interests

The authors declare no competing interests.

---

## References

1. WHO Classification of Tumours Editorial Board. Soft Tissue and Bone Tumours. 5th edn (IARC Press, 2020).
2. Gronchi A, et al. Soft tissue sarcomas: a multidisciplinary approach. *Lancet Oncol* 22, e428–e439 (2021).
3. Coindre JM, et al. Fluorescence in situ hybridization analysis for the diagnosis of soft tissue tumours. *Histopathology* 45, 461–469 (2004).
4. Suurmeijer AJH, et al. A novel group of spindle cell tumors defined by S100 and CD34 co-expression shows recurrent fusions involving RAF1, BRAF, and NTRK1/2 genes. *Genes Chromosomes Cancer* 57, 611–621 (2018).
5. Tap WD, et al. Structure-guided blockade of CSF1R kinase in tenosynovial giant-cell tumor. *N Engl J Med* 373, 428–437 (2015).
6. Kather JN, et al. Pan-cancer image-based detection of clinically actionable genetic alterations. *Nat Cancer* 1, 789–799 (2020).
7. Esteva A, et al. A guide to deep learning in healthcare. *Nat Med* 25, 24–29 (2019).
8. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. *Nat Med* 25, 44–56 (2019).
9. Boehm KM, et al. Deep learning for diagnosis and survival prediction in soft tissue sarcoma. *Ann Oncol* 32, 1178–1187 (2021).
10. Italiano A, et al. Genetic landscape and clinical outcome of patients with advanced grade 3 soft-tissue sarcomas. *Ann Oncol* 29, 1380–1386 (2018).
11. Sbaraglia M, et al. The 2020 WHO Classification of Soft Tissue Tumours: news and perspectives. *Pathologica* 113, 70–84 (2021).
12. Antonescu CR, et al. Molecular diagnosis of synovial sarcoma: utility of reverse transcriptase-polymerase chain reaction. *Mod Pathol* 13, 1–8 (2000).
13. Siegel RL, Miller KD, Wagle NS, Jemal A. Cancer statistics, 2023. *CA Cancer J Clin* 73, 17–48 (2023).
14. Schaefer IM, Cote GM, Hornick JL. Contemporary sarcoma diagnosis, genetics, and genomics. *J Clin Oncol* 36, 101–110 (2018).
15. Binh MBN, et al. MDM2 and CDK4 immunostainings are useful adjuncts in diagnosing well-differentiated and dedifferentiated liposarcoma subtypes: a comparative analysis of 559 soft tissue neoplasms with genetic data. *Am J Surg Pathol* 29, 1340–1347 (2005).
16. Sirvent N, et al. Detection of MDM2-CDK4 amplification by fluorescence in situ hybridization in 200 paraffin-embedded tumor samples: utility in diagnosing adipocytic lesions and comparison with immunohistochemistry and real-time PCR. *Am J Surg Pathol* 31, 1476–1489 (2007).
17. Zhu G, et al. Diagnosis of known sarcoma fusions and novel potential alterations by RNA sequencing. *Lab Invest* 99, 522–535 (2019).
18. Drilon A, et al. Efficacy of larotrectinib in TRK fusion–positive cancers in adults and children. *N Engl J Med* 378, 731–739 (2018).
19. Doebele RC, et al. Entrectinib in patients with advanced or metastatic NTRK fusion-positive solid tumours: integrated analysis of three phase 1–2 trials. *Lancet Oncol* 21, 271–282 (2020).
20. Demetri GD, et al. Efficacy and safety of trabectedin or dacarbazine for metastatic liposarcoma or leiomyosarcoma after failure of conventional chemotherapy: results of a phase III randomised multicentre clinical trial. *J Clin Oncol* 34, 786–793 (2016).
21. Butrynski JE, et al. Crizotinib in ALK-rearranged inflammatory myofibroblastic tumor. *N Engl J Med* 363, 1727–1733 (2010).
22. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst* 30, 4765–4774 (2017).
23. Kehl KL, et al. Assessment of deep natural language processing in ascertaining oncologic outcomes from radiology reports. *JAMA Oncol* 5, 1421–1429 (2019).
24. Savova GK, et al. Mayo clinic NLP system for patient phenotype extraction from electronic health records. *J Am Med Inform Assoc* 17, 507–513 (2010).
25. Wang Y, et al. Clinical information extraction applications: a literature review. *J Biomed Inform* 77, 34–49 (2018).
26. Campanella G, et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nat Med* 25, 1301–1309 (2019).
27. Echle A, et al. Deep learning in cancer pathology: a new generation of clinical biomarkers. *Br J Cancer* 124, 686–696 (2021).
28. Taylor BS, et al. Advances in sarcoma genomics and new therapeutic targets. *Nat Rev Cancer* 11, 541–557 (2011).
29. Thway K, Fisher C. Synovial sarcoma: defining features and diagnostic evolution. *Ann Diagn Pathol* 18, 369–380 (2014).
30. Koelsche C, et al. Sarcoma classification by DNA methylation profiling. *Nat Commun* 12, 498 (2021).
31. Shern JF, et al. Comprehensive genomic analysis of rhabdomyosarcoma reveals a landscape of alterations affecting a common genetic axis in fusion-positive and fusion-negative tumors. *Cancer Discov* 4, 216–231 (2014).
32. Le DT, et al. PD-1 blockade in tumors with mismatch-repair deficiency. *N Engl J Med* 372, 2509–2520 (2015).
33. Luchini C, et al. ESMO recommendations on microsatellite instability testing for immunotherapy in cancer, and its relationship with PD-1/PD-L1 expression and tumour mutational burden. *Ann Oncol* 30, 1232–1243 (2019).
34. von Mehren M, et al. Soft tissue sarcoma, Version 2.2016, NCCN clinical practice guidelines in oncology. *J Natl Compr Canc Netw* 14, 758–786 (2016).
35. Gronchi A, et al. Full-dose neoadjuvant anthracycline plus ifosfamide chemotherapy is associated with a relapse-free survival and overall survival benefit in localised high-risk adult soft-tissue sarcomas: a post hoc analysis of a randomised trial. *Lancet Oncol* 21, 806–817 (2020).
36. Cerami E, et al. The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. *Cancer Discov* 2, 401–404 (2012).
37. Chakravarty D, et al. OncoKB: a precision oncology knowledge base. *JCO Precis Oncol* 2017, 1–16 (2017).
38. Griffith M, et al. CIViC is a community knowledgebase for expert crowdsourcing the clinical interpretation of variants in cancer. *Nat Genet* 49, 170–174 (2017).
39. Benayed R, et al. High yield of RNA sequencing for targetable kinase fusions in lung adenocarcinomas with no mitogenic driver alteration detected by DNA sequencing and low tumour mutational burden. *Clin Cancer Res* 25, 4712–4722 (2019).
40. Grünewald TGP, et al. Ewing sarcoma. *Nat Rev Dis Primers* 4, 5 (2018).
