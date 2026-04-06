---
title: STS-Molecular-AI
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.11.0"
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---

# STS-Molecular-AI

**Soft Tissue Sarcoma Molecular Diagnostic Decision Support**

An AI-powered clinical decision support tool integrating FISH, RNA-NGS, and DNA-NGS for soft tissue sarcoma diagnosis.

## Features
- Multi-modal molecular input (FISH + RNA-NGS + DNA-NGS)
- Differential diagnosis with confidence scores
- Actionable therapeutic target identification (including trabectedin for DDIT3-rearranged liposarcoma)
- Testing strategy recommendations
- FISH–RNA-NGS discordance guidance

## Model
- Algorithm: Logistic Regression
- Training: 826 patients, 15 STS subtypes
- Performance: Macro-AUC = 0.780 (95% CI 0.764–0.800)

## Disclaimer
For research use only. Not validated for clinical diagnosis.
Always integrate molecular results with morphology and immunohistochemistry.

## Citation
> [Paper title]. *Nature Communications* (2026). doi: [pending]
