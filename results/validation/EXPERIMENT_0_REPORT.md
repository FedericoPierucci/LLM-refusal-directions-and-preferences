# Refusal Directions in LLMs: Experiment 0 - Direction Validation

**Researcher:** Federico Pierucci  
**Date:** 2025-10-25  
**Model:** Qwen/Qwen2.5-1.5B-Instruct  

---

## Executive Summary

We validated the existence of refusal directions in Qwen 2.5-1.5B-Instruct. All 28 layers show statistically significant separation between harmful and harmless prompts, with peak discrimination at layer 18 (Cohen's d = 9.05). Refusal builds progressively through layers rather than occurring at a single decision point.

---

## 1. Research Question

Do refusal directions reliably separate harmful/harmless representations across all layers, and where does discrimination emerge strongest?

---

## 2. Methodology

### Data
- **Harmful:** 256 prompts from `mlabonne/harmful_behaviors`
- **Harmless:** 256 prompts from `mlabonne/harmless_alpaca`

### Procedure
For each layer l ∈ {0, ..., 27}:

1. Extract activations at last token position: h_l ∈ ℝ^1536
2. Compute direction: d_l = mean(harmful) - mean(harmless)
3. Normalize: d̂_l = d_l / ||d_l||
4. Project all activations onto d̂_l
5. Statistical validation:
   - Independent samples t-test (H0: no difference between groups)
   - Effect size: Cohen's d = (μ_harm - μ_harmless) / σ_pooled
   - Valid if: p < 0.01 AND |d| > 0.5

### Tools
PyTorch 2.0, Transformers 4.35, NumPy, SciPy, Matplotlib | Hardware: Colab T4 GPU

---

## 3. Results

### Overall
- **Valid layers:** 28/28 (100%)
- **Strongest layer:** Layer 18 (Cohen's d = 9.05)
- **Pattern:** Progressive build-up from early to late layers

### Top 10 Layers

| Layer | Magnitude | Cohen's d | p-value |
|-------|-----------|-----------|---------|
| 18 | 35.09 | 9.05 | 0.00e+00 |
| 17 | 27.31 | 8.77 | 0.00e+00 |
| 20 | 42.00 | 8.38 | 0.00e+00 |
| 19 | 39.97 | 8.20 | 2.87e-321 |
| 21 | 53.78 | 8.16 | 2.66e-320 |
| 26 | 99.75 | 8.12 | 3.62e-319 |
| 25 | 93.00 | 8.05 | 4.94e-317 |
| 24 | 84.19 | 7.85 | 5.46e-312 |
| 27 | 79.19 | 7.83 | 2.01e-311 |
| 22 | 67.94 | 7.77 | 1.48e-309 |

### Layer Patterns

![Direction Analysis](direction_analysis.png)

**Figure 1.** Left: Separation distance grows exponentially after layer 15. Middle: Effect size plateaus at d ≈ 8 in layers 17-27. Right: Combined view showing both metrics track together.

**Early layers (0-10):** Weak separation (d = 2.8-4.5) - model begins distinguishing  
**Middle layers (11-17):** Rapid increase (d = 4.3-8.8) - discrimination solidifies  
**Late layers (18-27):** Maximal separation (d = 7.7-9.1) - clusters very far apart

---

## 4. Key Findings

1. **Universal separation:** Even layer 0 shows d = 2.75; refusal discrimination begins immediately
2. **Progressive build-up:** Not a binary switch; strength increases continuously through layers
3. **Extremely large effects:** Peak d = 9.05 (9 standard deviations apart) - among strongest documented geometric separations in LLMs

---

## 5. Comparison to Prior Work

**Arditi et al. (2024):**
- Used PCA showing separation exists across layers
- Selected one direction for behavioral intervention
- Focus: Can one direction jailbreak the model?

**Our work:**
- Systematic statistical validation of all 28 layers
- Quantified separation strength (Cohen's d) at each layer
- Focus: How strong is the representation at each layer?

**Contribution:** First systematic quantification revealing progressive build-up from layer 0 to peak at layer 18.

---

## 6. Limitations

- Single model (Qwen 2.5-1.5B)
- 256 prompts per category (larger sample would strengthen)
- Correlation only; causation untested (planned for Experiment 2)

---

## 7. Next Steps

**Experiment 1 (Planned):**
- 1a: Do different harm levels have different projection magnitudes?
- 1b: Does internal magnitude predict behavioral refusal?

**Experiment 2 (Planned):**
- Layer-specific ablation to test causal effects
- Test if removing directions changes behavior

---

## Conclusion

Refusal directions exist robustly across all layers with exceptionally large effect sizes. The progressive strengthening from early to late layers suggests refusal is a continuous computation rather than a discrete decision. These validated directions provide a solid foundation for testing behavioral predictions and causal interventions.

---

## Files Generated

- `direction_analysis.png` - Visualization (Figure 1)
- `validation_results.csv` - Complete statistical data
- `top10_layers.csv` - Top performers  
- `summary.json` - Numerical summary
- `validated_directions.pkl` - Directions for future experiments

**Code:** `notebooks/experiment_0_validation.ipynb`

---

**References:** Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717
