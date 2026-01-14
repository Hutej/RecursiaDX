# Research Ethics & Safety Disclaimers

## ⚠️ CRITICAL: Research Use Only

This GigaPath AI WSI Breast Cancer Lesion Analysis system is a **RESEARCH TOOL ONLY** and is **NOT approved for clinical diagnosis or patient care**.

---

## 🏥 Clinical Use Prohibition

### NOT a Medical Device

> [!CAUTION]
> **This system is NOT FDA-approved, CE-marked, or certified for clinical use in any jurisdiction.**

- **NOT a substitute** for pathologist review
- **NOT validated** for clinical decision-making
- **NOT intended** for patient diagnosis or treatment planning
- **NOT suitable** for use outside of controlled research environments

### Requires Clinical Oversight

> [!WARNING]
> **All outputs MUST be reviewed by a qualified pathologist before any clinical interpretation.**

This system provides:
- **Predictions**: Computational estimates, NOT clinical diagnoses
- **Confidence scores**: Model uncertainty, NOT medical certainty
- **Attention heatmaps**: Visualization aids, NOT diagnostic markers

---

## 🔬 Intended Use

### Acceptable Research Applications

✅ **Academic research** in computational pathology  
✅ **Algorithm development** and validation studies  
✅ **Educational demonstrations** in machine learning  
✅ **Benchmarking** against other AI methods  
✅ **Internal tool development** for research laboratories

### Prohibited Applications

❌ **Clinical diagnosis** of patients  
❌ **Treatment decisions** based solely on system output  
❌ **Screening programs** without pathologist oversight  
❌ **Regulatory submissions** claiming diagnostic accuracy  
❌ **Commercial deployment** for patient care

---

## 📊 Performance Limitations

### Known Constraints

1. **Training Data Bias**: Model trained on specific datasets may not generalize to all populations, staining protocols, or scanner types.

2. **Edge Cases**: Rare tumor subtypes, borderline cases, and artifacts may produce unreliable predictions.

3. **No Guarantees**: System accuracy varies by slide quality, tissue type, and preprocessing parameters.

4. **False Positives/Negatives**: Misclassifications WILL occur. Do not rely on this system as sole source of truth.

### Validation Requirements

> [!IMPORTANT]
> **Before using this system in ANY context:**
> - Validate on your own dataset
> - Establish performance baselines
> - Define acceptable error rates
> - Implement quality control procedures

---

## 🔒 Data Privacy & HIPAA Compliance

### Protected Health Information (PHI)

> [!CAUTION]
> **This system is NOT designed for HIPAA-compliant PHI handling.**

**User Responsibilities**:
- **De-identify** all WSI data before processing
- **Remove** patient identifiers from filenames and metadata
- **Secure** data storage and transmission
- **Comply** with institutional IRB requirements

**System Does NOT**:
- Encrypt data at rest (user must implement)
- Provide audit trails for PHI access
- Meet HIPAA Technical Safeguards without additional infrastructure

### Data Usage

By using this system, you acknowledge:
- Slides processed are for research purposes only
- No patient-identifiable data should be input
- Outputs may be retained for system improvement
- Compliance with local data protection regulations is YOUR responsibility

---

## 📜 Citation & Attribution

### Academic Use

If using this system in research, please cite:

```bibtex
@software{gigapath_wsi_2026,
  title={GigaPath AI: WSI Breast Cancer Lesion Analysis},
  author={[Your Institution]},
  year={2026},
  note={Research tool - not for clinical use},
  url={https://github.com/YourUsername/GigaPath-AI}
}
```

### Open Source License

This system is provided under the MIT License (see `LICENSE` file).

**NO WARRANTY**: Software is provided "AS IS" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.

---

##  User Acknowledgment

By proceeding to use this system, you explicitly acknowledge and agree that:

1. ☑️ You understand this is a **research tool**, not a medical device
2. ☑️ You will **NOT use** outputs for clinical diagnosis
3. ☑️ You will ensure **pathologist review** of all predictions
4. ☑️ You will **de-identify** patient data before processing
5. ☑️ You will **validate** the system on your own data
6. ☑️ You accept **full responsibility** for any use of this system
7. ☑️ You will **cite** this work if publishing results

---

## 📧 Contact & Reporting

### Reporting Issues

If you encounter:
- **Suspicious predictions** → validate with pathologist, do not rely on system
- **Software bugs** → open GitHub issue with de-identified examples
- **Security concerns** → contact maintainers privately

### Emergency Disclaimer

> [!CAUTION]
> **This system is NOT for emergency use. If you have a medical emergency, contact emergency services immediately.**

---

**Last Updated**: 2026-01-12

**Version**: 1.0

**Maintainers**: See `README.md` for contact information
