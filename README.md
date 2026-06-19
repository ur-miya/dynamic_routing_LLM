# Dynamic Routing for LLM Distillation

This repository contains code for experiments on dynamic routing between a large teacher LLM and a distilled student, including hybrid routing policies and self-refreshing distillation cycles.

## Overview

The research is about the method how to route user queries between a strong teacher model (Qwen 32B) and a smaller distilled student in order to maximize answer quality under a budget on teacher calls (Teacher Call Rate, TCR). The project includes:
- Training two students (Soft KD + LoRA and Distillm2 student).
- Designing several routing policies (A: classifier, B: uncertainty-based LR, C: IRT router, D: hybrid router).
- Iterative refinement where routing is used to identify student failure regions and drive further distillation on teacher-beneficial examples.

- ## Models and Data

- **Teacher**: Qwen 32B, used to generate answers and token-level logits/log-probs on OASST1 subsets (train/val/test). Reference answers from OASST1 are used for quality evaluation and routing signals.
- **Students**:
  - **Soft KD + LoRA**: Qwen 1.5B with LoRA adapters trained via token-wise KL with temperature and an additional supervised term on references.
  - **Distillm2**: A student trained with a modified objective that incorporates teacher uncertainty, token-wise errors vs references, and regularizers to make the student confident on easy examples and conservative on hard ones.

- **Dataset**: OASST1 (train/val/test splits) with teacher outputs and reference answers.
- **Difficulty / UQ signals**: token-level log-probs, entropies, sequence NLL, and IRT-based difficulties (`irt_difficulties_er.csv`).

- ## Routing Policies

There is the implementation of four routing policies between the student and the teacher:

- **Router A: hand-crafted feature classifier**  
  Binary classifier over features capturing example difficulty (errors, log-probs, UQ features) and static task features (length, prompt type, etc.). Target TCR ≈ 20% on validation, but ends up around 11% on test for Distillm2, with a noticeable ROUGE-1 gain over the student-only baseline.

- **Router B: uncertainty-based logistic regression**  
  Logistic regression over uncertainty features from Distillm2: mean, max and first-token entropies plus sequence NLL. In the current configuration, the model ranks examples reasonably.

- **Router C: IRT-based router**  
  Uses IRT-estimated difficulty scores to route higher-difficulty examples to the teacher and simpler ones to the student. For Distillm2, it achieves TCR ≈ 17.2% and the best ROUGE-1 ≈ 0.368 among current routers.

- **Router D: hybrid router**  
  Combines IRT difficulty with classifier/UQ signals in a single decision rule trained over hybrid features. It reads `router_hybrid_config.csv` (threshold, selection mode, target TCR, path to `router_hybrid.joblib`) and is calibrated to a TCR comparable to C (~17%).

  ## Installation

```bash
git clone dynamic_routing_LLM
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

<!-- Describe any external dependencies (e.g., access to Qwen 32B, cluster-specific launchers). -->

## Data Preparation

1. Download OASST1 and place it under `data/` (see `data/README.md`).
2. Run teacher inference to generate answers and token-level logits/log-probs (script: `src/data/run_teacher_inference.py`).
3. Compute IRT difficulties:
   ```bash
   python scripts/routing/04_compute_irt.py \
       --input outputs/teacher_outputs.jsonl \
       --output outputs/irt_difficulties_er.csv
   ```

## Running Distillation Experiments

### Train students

Soft KD + LoRA:
```bash
python distillation/train_soft_kd_lora.py
```

Distillm2:
```bash
python distillation/train_distillm2.py
```

Both scripts should:
- load teacher logits and references on OASST1 train split,
- train the student,
- save checkpoints and training logs into `outputs/distilled_*/`.

## Running Routing Experiments

### Routers A–D

```bash
python scripts/routing/05_train_router_classifier.py
python scripts/routing/06_calibrate_uncertainty_router.py
python scripts/routing/07_calibrate_irt_router.py
python scripts/routing/07b_train_hybrid_router.py
```

Each script:
- loads the chosen student model (typically Distillm2),
- applies the routing policy on OASST1 test split,
- reports TCR, routing accuracy, F1(teacher), AUC-ROC and ROUGE-1.

## Self-Refreshing Distillation Cycle

Hybrid router D is used to identify regions where the student underperforms the teacher and then fine-tune the student on these examples.

1. **Quality estimation**  
   For each query, we estimate the quality scores of teacher and student with automatic metrics (ROUGE, BERTScore) and optionally an LLM-as-a-judge on the cluster.

2. **Improvement signal**  
   We define a target for the router in terms of quality improvement:
   \[
   \Delta Q = Q(\text{teacher}) - Q(\text{student})
   \]
   The router sends a query to the teacher only if the expected \(\Delta Q\) exceeds a threshold.

3. **Hard example mining**  
   We accumulate examples where the teacher significantly outperforms the student (high \(\Delta Q\)) and add them to a distillation buffer.

4. **Student refresh**  
   Periodically, we fine-tune the student on the buffer and recalibrate router D on the new student, closing the loop.

<!--Scripts for this cycle live under `scripts/run_cycle_distill_refresh.sh` and corresponding modules in `src/distillation/` and `src/routing/`.-->

<!--## Results  -->

<!--On the Distillm2 student and OASST1 test split:  -->

<!--- **Student-only baseline**: TCR = _%, ROUGE-1 ≈ _.  -->
<!--- **Router A (classifier)**: TCR ≈ _%, ROUGE-1 ≈ _  -->
<!--- **Router B (uncertainty LR)**: TCR = _%, ROUGE-1 ≈ _  -->
<!--- **Router C (IRT)**: TCR ≈ _%, ROUGE-1 ≈ _, F1(teacher) ≈ _  -->
<!--- **Router D (hybrid)**: TCR ≈ _%, ROUGE-1 ≈ _, higher AUC-ROC than C (≈ _ vs _).  -->
