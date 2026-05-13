# Research overview

Engineering-oriented summary of the thesis work: how the cross-modal dataset is built, how weak labels flow into models, and what the headline results mean (with the same caveats as the paper).

## Problem and approach

The project studies **behavioral and emotional signals** in cats from **short public social-media video**, using **weak supervision** throughout. A **cross-modal dataset** ties video snippets to **audio-derived pseudo-labels** (ten-class vocal behavior taxonomy) and rich metadata from automated filtering and model-assisted review.

The **primary modeling story** is **appearance-free pose dynamics**: body keypoints over time feed spatial–temporal models (ST-GCN and related pairwise pose classifiers) so the video branch does not rely on RGB appearance for pain-related discrimination. **Secondary paths** include vision baselines (e.g. EfficientNet-B3 on cropped cats), GPT-assisted Feline Grimace Scale–style evaluation, and pose extraction tooling (ViTPose, DeepLabCut SuperAnimal). **Operational** pieces cover multi-platform scraping, manifest pipelines, CLI inference, and an optional local web MVP.

This is **preliminary research evidence**, **not** a clinical validation study and **not** a veterinary diagnostic product.

## Dataset (scale and labels)

- **Corpus:** on the order of **7.3k** snippet-level rows in the canonical manifest `final_dataset_v2.jsonl` (e.g. 7,318 rows as reported in `src/dataset_construction/paper_figures/output_pngs/figure_report.txt` when generated from that manifest).
- **Ten-class audio pseudo-labels:** `audio_label_10` from the cat emotion classifier; treated as noisy teaching signal for cross-modal analysis, not ground truth ethology.
- **Five-class target:** `final_label_5` merges toward a simpler behavioral taxonomy where defined; many analyses focus on pain-related **proxy** contrasts (e.g. Paining vs Resting) rather than claiming clinical pain ground truth.

To regenerate manifest-driven figures and tables:

```bash
python src/dataset_construction/paper_figures/generate_plots.py
```

Outputs land in `src/dataset_construction/paper_figures/output_pngs/`.

## Headline modeling results (pose branch)

Reported from the same engineering summary as `src/dataset_construction/reports/video_pipeline_comprehensive_report.txt`:

- **Pairwise P4 pose classifiers** (one-vs-one heads, pain-inclusive pairs): a representative strong pair is **Paining | Resting** with validation **macro F1 ≈ 0.761** and **ROC AUC ≈ 0.821** (single run cited there: `individual_binary_pairs/01_Paining_Resting__restsingsweep_gs0030`).
- **Stack / meta-classifier** and **ablations** (dropped voters, body-region drops) are described in that report; bootstrap intervals for some AUC comparisons are wide—treat rankings as **exploratory**.

## Secondary figures and validation artifacts

Supporting plots are generated locally (not duplicated in git for every clone). Use these when you need charts beyond the root README’s single funnel figure.

- **Dataset characterization and label structure:** bars, heatmaps, funnel tables — `python src/dataset_construction/paper_figures/generate_plots.py` → `src/dataset_construction/paper_figures/output_pngs/` (e.g. `fig_09_audio10_labels_twopanel.png`, `fig_13_pipeline_funnel.png`, companion `table_*.txt`).
- **External audio benchmark (NAYA / CatEmotion-style validation):** `src/dataset_construction/paper_figures/plot_naya_catemotion_confusion_heatmap.py` writes confusion plots such as `fig_23_*` under the same `output_pngs/` directory.
- **Human-validation summaries:** metrics text under `src/human_validation/` (e.g. `naya_catemotion_validation_metrics.txt`).

## Inference stack (what actually runs on one clip)

End-to-end routing is documented in [src/inference/README.md](../src/inference/README.md): audio branch vs pose branch after a lightweight audio gate, optional separation for the audio path, and pose-based scoring for the video path.

## Ethics and limitations (summary)

- **Weak supervision** and **pseudo-labels** imply **label noise** and domain shift.
- **Public video** carries consent, bias, and platform-policy constraints; labels reflect online content, not clinical exams.
- **No claim** of sensitivity/specificity for clinical pain management—see ethics section in the root [README](../README.md) and the thesis text.

## Where to read more

- [Root README](../README.md) — landing page and quickstart pointers
- [Dataset construction](../src/dataset_construction/README.md) — manifest steps
- [Data pipeline v2](../src/scrapers/data_pipeline_v2/README.md) — collection orchestration
- [Training and analysis](training-and-analysis.md) — advanced commands
- [Repository layout](repository-layout.md) — directory tree
