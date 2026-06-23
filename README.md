# Dynamic Routing for LLM Distillation

This repository contains code for experiments on dynamic routing between a large teacher LLM and a compact distilled student, including hybrid routing, gain-aware escalation, and a student refresh stage driven by teacher-beneficial examples.

## Overview

This project studies how to route user queries between a strong teacher model and a smaller distilled student in order to maximize answer quality under a budget on teacher calls, measured by Teacher Call Rate (TCR). The framework combines teacher–student knowledge distillation with several routing strategies and extends routing from a static inference-time decision to a mechanism for identifying student failure regions and supporting targeted student improvement.

The current implementation consists of two connected stages:

- **Phase 1: static routing**
  - Train and evaluate routing policies on top of a fixed distilled student.
  - Compare hand-crafted, uncertainty-based, IRT-based, and hybrid routers under a quality–cost trade-off.

- **Phase 2: gain-aware refresh**
  - Introduce a delta-margin routing target based on the expected teacher-over-student gain.
  - Use high-gain teacher-beneficial examples to refresh the student and recalibrate the hybrid router.

## Models and Data

### Teacher

- **Teacher model**: Qwen 32B.
- The teacher is used to generate:
  - free-form responses,
  - token-level logits or log-probabilities,
  - supervision signals for student training,
  - response-quality and routing-related features.

### Students

Two student configurations are considered in the project.

#### 1. Soft KD + LoRA student

- Base model: Qwen 1.5B with LoRA adapters.
- Training objective:
  - token-wise KL distillation with temperature scaling,
  - supervised loss on reference answers.

This student is part of the earlier distillation setup and is retained for comparison and baseline experimentation.

#### 2. DistiLLM-2-style student

- Compact Qwen-based student trained with a DISTILLM-2-style contrastive objective.
- Main training components:
  - **SKL** on teacher-generated outputs,
  - **SRKL** on student-generated outputs,
  - **CE** term for stabilization.

Two versions of this student are used:

- **student v1**
  - The fixed distilled backbone used in Phase 1 routing experiments.

- **student v2**
  - An improved student obtained after distilling on hard-examples (teacher-beneficial after phase 1).
  - Used in Phase 2 gain-aware routing experiments.

In the routing experiments, the DistiLLM-2-style student is the main student because it provides the best balance between quality and informative routing features.

### Dataset

- **Dataset**: OASST1 with train / validation / test splits.
- Reference answers from OASST1 are used for:
  - response evaluation,
  - student supervision,
  - routing signals,
  - teacher–student quality-gap estimation.

### Feature sources

The routing pipeline uses several families of features:

- **Quality and difficulty features**
  - response-level metrics,
  - prompt descriptors,
  - error-related signals.

- **Uncertainty features**
  - mean token entropy,
  - maximum token entropy,
  - first-token entropy,
  - sequence negative log-likelihood.

- **IRT-based difficulty**
  - scalar or structured difficulty estimates computed from Item Response Theory models.

- **Judge-based features**
  - automatic response-quality assessments from an external LLM-as-a-judge pipeline for both teacher and student outputs.

## Routing Policies

The repository includes four routing strategies for deciding whether a query should be answered by the student or escalated to the teacher.

### Router A: hand-crafted feature classifier

Binary classifier over manually constructed routing features, including:

- quality and difficulty indicators,
- uncertainty statistics,
- static prompt features such as length or prompt type.

This router serves as a supervised feature-based baseline.

### Router B: uncertainty-based logistic regression

Logistic regression over uncertainty features extracted from the student, such as:

- mean entropy,
- max entropy,
- first-token entropy,
- sequence NLL.

This router is intended as a transparent uncertainty-only baseline.

### Router C: IRT-based router

Difficulty-aware router based on Item Response Theory estimates.

The core idea is:

- higher-difficulty examples are more likely to be escalated to the teacher,
- lower-difficulty examples are more likely to remain with the student.

This router provides a more interpretable routing signal than uncertainty alone.

### Router D: hybrid router

Hybrid router that combines multiple signal families into one routing score, including:

- classifier-derived features,
- uncertainty features,
- IRT-based difficulty,
- judge-based quality signals.

In Phase 1, Router D is used as the strongest static router. 
In Phase 2, Router D is extended to a **gain-aware** version trained on teacher-over-student quality gain rather than only a binary routing label.

## Gain-Aware Routing

The main extension in the second stage is a shift from difficulty-only routing to **gain-aware routing**.

For each query \(x\), the framework estimates:

\[
\Delta Q(x) = Q_T(x) - Q_S(x)
\]

where:

- \(Q_T(x)\) is the estimated quality of the teacher output,
- \(Q_S(x)\) is the estimated quality of the student output.

The routing rule becomes:

- send the query to the **teacher** if the expected gain is above a margin threshold,
- keep the query with the **student** otherwise.

This means that a difficult query is not automatically escalated; escalation happens only when the teacher is expected to provide a sufficiently larger benefit than the student.

## Iterative Student Refresh

A central idea of this project is that routing is not only a deployment-time controller, but also a way to identify where the student still needs improvement.

The refresh cycle is organized as follows:

1. Train or refresh the current student with the DistiLLM-2-style objective.
2. Generate student outputs on the routing dataset.
3. Compute quality metrics, uncertainty features, IRT difficulty, and judge-based scores.
4. Estimate the teacher-over-student gain \(\Delta Q(x)\) for each query.
5. Select high-gain teacher-beneficial examples.
6. Add these examples to a targeted distillation buffer.
7. Refresh the student on the augmented data.
8. Recompute routing features and recalibrate Router D for the updated student.

At the current stage, the repository reflects a **single refresh step** from student v1 to student v2, rather than a fully repeated multi-round loop.

## Repository Workflow

A typical workflow in this repository is:

1. Prepare OASST1 data.
2. Generate teacher outputs and token-level statistics.
3. Train the student model.
4. Generate student outputs.
5. Compute routing features.
6. Train or calibrate routers.
7. Evaluate routing quality and answer quality.
8. Run the gain-aware refresh stage for student improvement.

## Installation

```bash
git clone dynamic_routing_LLM
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Additional environment-specific dependencies may be required for:
- access to the teacher model,
- cluster execution,
- judge-based scoring.

## Data Preparation

1. Download OASST1 and place it under the project data directory.
2. Prepare train / validation / test splits.
3. Run teacher inference to generate:
   - teacher responses,
   - token-level logits / log-probs,
   - output files used in student training and routing feature extraction.

```bash
python scripts/01_generate_teacher_outputs.py \
  --input_file data/raw/oasst1/train.csv \
  --output_dir outputs/teacher_outputs \
  --num_workers 10 \
  --max_tokens 512 \
  --temperature 0.7 \
  --no_think

python scripts/03_generate_teacher_logprobs.py \
  --input_file data/raw/oasst1/train.csv \
  --output_dir outputs/teacher_logprobs \
  --num_workers 10 \
  --max_tokens 512 \
  --top_logprobs 10

python scripts/04_generate_sgo_logprobs.py \
  --input_file data/raw/oasst1/train.csv \
  --output_dir outputs/sgo_logprobs \
  --num_workers 5 \
  --max_tokens 512 \
  --top_logprobs 10 \
  --no_think
```

4. Compute IRT-based difficulty estimates.

```bash
python scripts/routing/04_compute_irt.py \
  --features_csv outputs/routing/features_er.csv \
  --baseline_csv outputs/evaluation/baseline_detailed.csv \
  --output_dir outputs/routing
```

5. Compute routing and response-level features.

```bash
python scripts/routing/01_generate_student_responses_er.py \
  --lora_path outputs/distillm2_model5k \
  --input_file data/raw/oasst1/trainer.csv \
  --output_dir outputs/routing \
  --batch_size 4 \
  --device cuda:0

python scripts/routing/02_compute_features_er.py \
  --input_csv outputs/routing/student_responses_er.csv \
  --output_dir outputs/routing \
  --rouge_threshold 0.15 \
  --bert_threshold 0.82 \
  --device cuda:0
```

6. If used in the current experiment setup, compute judge-based scores for student and teacher outputs.

```bash
python scripts/routing/06b_compute_judge_scores.py \
  --input_csv outputs/routing/features_er.csv \
  --response_col student_response \
  --output_csv outputs/routing/judge_scores_student.csv \
  --batch_size 16

python scripts/routing/06b_compute_judge_scores.py \
  --input_csv outputs/routing/features_er.csv \
  --response_col reply \
  --output_csv outputs/routing/judge_scores_teacher.csv \
  --batch_size 16
```

## Running Distillation Experiments

### Train Soft KD + LoRA student

```bash
python distillation/train_soft_kd_lora.py \
  --teacher_logprobs_file outputs/teacher_logprobs/teacher_logprobs_full.jsonl \
  --output_dir outputs/distilled_softkd \
  --max_samples 5000 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --temperature 2.0 \
  --top_k 10
```

Expected output:
- trained LoRA checkpoints,
- training logs,
- evaluation artifacts.

### Train DistiLLM-2-style student

```bash
python distillation/train_distillm2.py \
  --teacher_logprobs_file outputs/teacher_logprobs/teacher_logprobs_full.jsonl \
  --student_logprobs_file outputs/sgo_logprobs/sgo_logprobs_full.jsonl \
  --output_dir outputs/distillm2_model5k \
  --max_samples 5000 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3
```

Expected output:
- student checkpoints,
- training logs,
- generated outputs for subsequent routing experiments.

### Refresh student to obtain student v2

```bash
python distillation/train_distillm2.py \
  --teacher_logprobs_file outputs/teacher_logprobs/teacher_logprobs_full.jsonl \
  --student_logprobs_file outputs/sgo_logprobs/sgo_logprobs_full.jsonl \
  --output_dir outputs/distillm2_model_v2 \
  --max_samples 2000 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 1 \
  --beta_max 0.7 \
  --beta_min 0.1 \
  --alpha_ce 0.3
```

This stage should use:
- the updated SKL/SRKL-based training setup,
- teacher-beneficial or high-gain examples,
- stabilization constraints used in the refined student v2 pipeline.[file:108]

## Running Routing Experiments

### Train / calibrate Router A

```bash
python scripts/routing/05_train_router_classifier.py \
  --features_csv outputs/routing/features_er.csv \
  --output_dir outputs/routing \
  --model_name answerdotai/ModernBERT-base \
  --num_epochs 3 \
  --target_teacher_rate 0.3 \
  --device cuda:0
```

### Train / calibrate Router B

```bash
python scripts/routing/06_calibrate_uncertainty_router.py \
  --features_csv outputs/routing/features_er.csv \
  --output_dir outputs/routing \
  --target_teacher_rate 0.3
```

### Train / calibrate Router C

```bash
python scripts/routing/07_calibrate_irt_router.py \
  --features_csv outputs/routing/features_er.csv \
  --irt_csv outputs/routing/irt_difficulties_er.csv \
  --output_dir outputs/routing \
  --target_teacher_rate 0.3
```

### Train / calibrate Router D

```bash
python scripts/routing/07b_train_hybrid_router_margin.py \
  --features_csv outputs/routing/features_er.csv \
  --output_dir outputs/routing_distilled/router_v2_margin \
  --target_teacher_rate 0.3 \
  --margin_delta 0.25
```

### Evaluate routing quality

```bash
python scripts/routing/08_evaluate_routers.py \
  --test_csv data/raw/oasst1/test.csv \
  --student_responses_csv outputs/evaluation/distilled_detailed.csv \
  --lora_path outputs/distillm2_model5k \
  --routing_dir outputs/routing \
  --output_dir outputs/routing \
  --device cuda:0
```

Typical reported routing metrics include:

- Routing Accuracy,
- F1-score for the teacher class,
- AUC-ROC,
- Teacher Call Rate (TCR),
- blended answer quality metrics such as ROUGE-1 / ROUGE-2 / ROUGE-L,
- optionally BERTScore and judge-based metrics.

## Running the Gain-Aware Phase

The gain-aware stage extends Router D with an explicit teacher-over-student gain target.

### Step 1. Estimate quality scores

Compute quality estimates for teacher and student responses using:
- ROUGE,
- BERTScore,
- judge-based scores, if available.

```bash
python scripts/routing/02_compute_features_er.py \
  --input_csv outputs/evaluation/distilled_detailed.csv \
  --output_dir outputs/evaluation \
  --rouge_threshold 0.15 \
  --bert_threshold 0.82 \
  --device cuda:0
```

### Step 2. Build delta-margin targets

Construct gain labels using:

\[
\Delta Q(x) = Q_T(x) - Q_S(x)
\]

and define the routing target using a margin threshold.

```bash
python scripts/routing/08c_evaluate_hybrid_margin.py \
  --student_responses_csv outputs/evaluation/distillm2_v2_full/distilled_detailed.csv \
  --routing_dir outputs/routing_distilled_v2/router_v2_margin \
  --classifier_model_dir outputs/routing_distilled_v2/router_v2_margin/router_classifier/best_model \
  --margin_delta 0.25 \
  --max_samples 0 \
  --output_suffix _margin_target
```

### Step 3. Train gain-aware Router D

```bash
python scripts/routing/07b_train_hybrid_router_margin.py \
  --features_csv outputs/routing_distilled/features_er.csv \
  --output_dir outputs/routing_distilled/router_v2_margin \
  --margin_delta 0.25 \
  --target_teacher_rate 0.3 \
  --device cuda:0
```

### Step 4. Mine teacher-beneficial examples

```bash
python scripts/routing/09_select_hurd_examples.py \
  --features_csv outputs/routing_distilled/features_er.csv \
  --output_csv outputs/routing_distilled/teacher_beneficial_examples.csv \
  --margin_delta 0.25
```

### Step 5. Refresh the student and recalibrate the router

```bash
python distillation/train_distillm2.py \
  --teacher_logprobs_file outputs/teacher_logprobs/teacher_logprobs_full.jsonl \
  --student_logprobs_file outputs/sgo_logprobs/sgo_logprobs_full.jsonl \
  --output_dir outputs/distillm2_model_v2_refresh \
  --max_samples 2000 \
  --num_epochs 1 \
  --beta_max 0.7 \
  --alpha_ce 0.3

python scripts/routing/07b_train_hybrid_router_margin.py \
  --features_csv outputs/routing_distilled_v2/features_er.csv \
  --output_dir outputs/routing_distilled_v2/router_v2_margin \
  --margin_delta 0.25 \
  --target_teacher_rate 0.3 \
  --device cuda:0
```

## Experimental Structure

### Phase 1

- Fixed student: **student v1**.
- Goal:
  - compare Routers A–D,
  - evaluate static routing under a quality–cost trade-off,
  - identify the strongest router for low-budget deployment.

### Phase 2

- Updated student: **student v2**.
- Goal:
  - evaluate gain-aware Router D,
  - validate the student refresh idea,
  - test the transition from static routing to a self-refreshing pipeline.

## Results

The current repository reflects the following main conclusions:

- Hybrid routing is the strongest static routing strategy in the completed first-stage experiments.
- The refined student v2 is stronger than student v1 on reference-based quality metrics.
- Gain-aware routing is a proof of concept for linking routing and student refresh, but the current implementation corresponds to a single refresh step rather than a fully repeated closed-loop pipeline.
