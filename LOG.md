# Development Log - Project 14
# Efficient Extractive Question Answering on SQuAD with Distilled Models

**Team Members:** Sherouk | Mostafa | Salma  
**Course:** CISC 867 Deep Learning  
**Repository:** https://github.com/Salma-21/Lightweight-Extractive-Question-Answering-on-SQuAD-Using-DistilBERT#

---

## Week 1

### Date: 03/05/2026 - 09/05/2026

#### Progress
- [x] Sherouk: Loaded SQuAD v1.1, inspected JSON structure, displayed 5 examples, and ran span alignment audit on all answer offsets
- [x] Sherouk: Implemented tokenizer with special tokens, truncation, and stride/window approach (doc_stride=128, max_length=512) and verified chunk shapes
- [x] Sherouk: Implemented answer start and end offset mapping across all chunks and fixed early-return bug that corrupted labels for multi-chunk examples
- [x] Sherouk: Split dataset at article level into 80/10/10 train/val/test, subset train to 50k QA pairs, and encoded all three splits
- [x] Sherouk: Implemented BM25 baseline using rank_bm25 with official SQuAD scoring (max over all gold answers per question)
- [x] Sherouk: Computed and reported baseline results (BM25: EM=0.12%, F1=14.19%) and ran diagnostic confirming low scores are inherent to sentence retrieval not a bug
- [x] Sherouk: Saved BM25 predictions to bm25_baseline_predictions.json for Student C error analysis and official eval script compatibility
- [x] Mostafa: Read DistilBERT paper (Sanh et al., 2019) and studied HuggingFace QA API (DistilBertForQuestionAnswering, TrainingArguments, Trainer)
- [x] Mostafa: Ran sanity-check forward pass on DistilBertForQuestionAnswering with dummy batch (batch_size=2, seq_length=384); verified output shapes: start_logits [2, 384], end_logits [2, 384]
- [x] Salma: Studied the SQuAD evaluation metrics, Exact Match and token-level F1
- [x] Salma: Used the SQuAD-style evaluation script functions to compute EM and F1
- [x] Salma: Built results-logging utilities to save baseline/model predictions, scores, and summaries to CSV

#### Key Decisions
- Chose subset size of 50K QA pairs because it fits within CPU memory and time budget
- Decided on an 80% / 10% / 10% train/val/test split following the standard SQuAD dataset approach (Rajpurkar et al., 2016)
- Chose doc_stride=128 and max_length=512 for the strided tokenization window because DistilBERT has a hard architectural limit of 512 tokens, and a stride of 128 ensures consecutive windows overlap enough so the answer is never accidentally cut between two chunks
- Chose BM25 (BM25Okapi) as the retrieval baseline over TF-IDF following established practice in QA literature, where it is used as the standard strong retrieval baseline on SQuAD (Karpukhin et al., 2020)

#### Issues & How They Were Resolved
- **Issue:** `offset_mapping` function contained a `return` statement inside the chunk loop, causing it to always exit after the first chunk regardless of whether the answer was found. For multi-chunk examples this produced incorrect `(0, 0)` positions pointing to `[CLS]` instead of the real answer span, silently corrupting training labels.  
  **Resolution:** Moved the `return` outside the loop by collecting all chunk results in a `results` list and returning it after the loop completes. Updated all three encoding loops (train, val, test) to iterate over `chunk_positions` using `enumerate` instead of a fixed range, ensuring each chunk gets its own correctly computed start and end position.  
  **Resolved by:** Sherouk

- **Issue:** BM25 baseline returned very low scores (EM: 0.12%, F1: 14.19%) which initially suggested a preprocessing or implementation bug.  
  **Investigation:** Ran a single example diagnostic that revealed BM25 was correctly identifying the relevant sentence in most cases. For example, given the question "When was Gaddafi born, and when did he die?", BM25 correctly retrieved the sentence containing the gold answer but returned the full sentence, resulting in EM=0 and F1=0.45 for that example.  
  **Root Cause:** BM25 is a sentence retriever not a span extractor. SQuAD gold answers are short precise spans so a full sentence prediction will almost never produce an exact match and will always have low F1 due to the extra words. This is an inherent limitation of retrieval-based baselines on extractive QA tasks, not a bug.  
  **Resolution:** Confirmed the implementation is correct. The low numbers are expected and explainable. This finding motivates the use of DistilBERT which learns to extract the exact answer span at token level rather than returning the whole sentence.  
  **Resolved by:** Sherouk

---

## Week 2

### Date: 10/05/2026 - 16/05/2026

#### Progress
- [x] Mostafa: Fixed critical integration bug — val_dataset was never defined by Student A; wrapped all preprocessing lists into HuggingFace Dataset objects using Dataset.from_dict() and applied .set_format('torch')
- [x] Mostafa: Wrote full training loop using HuggingFace Trainer API with TrainingArguments (lr=2e-5, batch=16, grad_accum=2, epochs=3, weight_decay=0.01, warmup_ratio=0.10, fp16=True)
- [x] Mostafa: Fine-tuned DistilBERT (66.4M params) on 50,068 training chunks for 3 epochs on Tesla T4 GPU; training completed in 0.59 hours (35 min 31 sec)
- [x] Mostafa: Logged training loss per step and validation loss per epoch; Epoch 1 (train=1.4506, val=1.1979), Epoch 2 (train=1.1537, val=1.0980 ← best), Epoch 3 (train=0.9677, val=1.0917)
- [x] Mostafa: Saved best checkpoint (Epoch 2, val_loss=1.0980) to ./distilbert-squad-final (252 MB); model restored automatically via load_best_model_at_end=True
- [x] Mostafa: Wrote inference script; tested on 3 examples: "Who introduced BERT?" → "Devlin" (0.41), "How many params in DistilBERT?" → "66 million" (0.73), "What dataset?" → "SQuAD v1.1" (0.73); CPU latency: 140.13 ms/example
- [x] Mostafa: Created hyperparameter config file (config.yaml) documenting all training settings for reproducibility
- [x] Salma: Ran evaluation on BM25 baseline and fine-tuned DistilBERT outputs using 500 validation examples
- [x] Salma: Produced comparison table and EM/F1 bar chart
- [x] Salma: Completed first-pass qualitative error analysis on 15 wrong DistilBERT predictions
- [x] Salma: Added a small robustness check using 10 manually reworded questions

#### Key Decisions
- Chose learning rate of 2e-5 because it is the standard for BERT/DistilBERT fine-tuning (Devlin et al., 2019); tested 5e-5 (unstable loss) and 1e-5 (too slow to converge); 2e-5 achieved steady convergence
- Chose batch size of 16 (effective 32 with gradient_accumulation_steps=2) due to Tesla T4 GPU memory limit (15.6 GB VRAM); batch_size=32 caused OOM error
- Decided on max_length=512 because DistilBERT max position embeddings = 512; covers ~99% of SQuAD examples without truncation; stride=128 maintained from Student A preprocessing
- Chose 3 epochs because validation loss plateaued at Epoch 2 (1.0980) and slightly increased at Epoch 3 (1.0917); used load_best_model_at_end=True to restore best checkpoint automatically

#### Issues & How They Were Resolved
- **Issue:** The evaluation functions `compute_exact` and `compute_f1` were not available when the Student C section was run separately.  
  **Resolution:** Confirmed that the SQuAD-style evaluation script must be downloaded/imported before running the Student C evaluation cells. After importing the metric functions, the baseline and model evaluation ran correctly.  
  **Resolved by:** Salma

- **Issue:** val_dataset was not defined; Trainer raised NameError when Student B attempted to run training.  
  **Resolution:** Wrapped Student A's preprocessing lists into HuggingFace Dataset objects using Dataset.from_dict() and applied .set_format('torch') to enable Trainer compatibility.  
  **Resolved by:** Mostafa

- **Issue:** Initial training run crashed with OOM (Out of Memory) error with batch_size=32 on Tesla T4 GPU.  
  **Resolution:** Reduced batch_size to 16 and added gradient_accumulation_steps=2 (effective batch = 32). Maintained gradient stability within GPU memory limits.  
  **Resolved by:** Mostafa

- **Issue:** Initial automatic paraphrases for the robustness check produced unnatural questions such as "Which thing role...".  
  **Resolution:** Replaced them with 10 manually written paraphrases to make the robustness check more realistic and readable.  
  **Resolved by:** Salma

#### Preliminary Results
| Model | Exact Match | F1 Score |
|-------|-------------|----------|
| Baseline (BM25) | 0.00% | 12.75% |
| DistilBERT (fine-tuned) | 67.40% | 78.42% |

---

## Week 3

### Date: 17/05/2026 - 23/05/2026

#### Progress
- [x] Mostafa: Read TinyBERT paper (Jiao et al., 2020) and set up TinyBERT fine-tuning pipeline using huawei-noah/TinyBERT_General_4L_312D checkpoint
- [x] Mostafa: Fine-tuned TinyBERT on the same 50,068 training chunks; required 20 epochs to converge due to compressed architecture
- [x] Mostafa: Logged TinyBERT training and validation loss across all 20 epochs; confirmed stable convergence with no overfitting
- [x] Salma: Ran full evaluation of TinyBERT on validation and test sets; logged EM, F1, and inference latency
- [x] Salma: Produced side-by-side comparison of DistilBERT and TinyBERT results

#### Key Decisions
- Chose 20 epochs for TinyBERT because the model required significantly more training steps to converge compared to DistilBERT, consistent with findings in the original TinyBERT paper
- Kept all other hyperparameters consistent with DistilBERT training for fair comparison

#### Preliminary Results
| Model | Exact Match | F1 Score | Latency (ms) | Params |
|-------|-------------|----------|--------------|--------|
| TinyBERT (fine-tuned) | 27.40% | 36.02% | 12.46 | 14.5M |
| DistilBERT (fine-tuned) | 66.80% | 77.93% | 104.0 | 66M |

---

## Week 4

### Date: 24/05/2026 - 30/05/2026

#### Progress
- [x] Mostafa: Designed and ran sequence-length ablation for both DistilBERT and TinyBERT across three input lengths: 128, 256, and 384 tokens
- [x] Mostafa: Logged EM, F1, and CPU inference latency for each sequence length configuration
- [x] Salma: Designed and ran TinyBERT layer-freezing ablation across three configurations: freeze layer 0 only, freeze layers 0-1, and freeze layers 0-3
- [x] Salma: Logged EM, F1, validation loss, trainable parameters, training time, and inference latency for each TinyBERT freezing configuration
- [x] Salma: Designed and ran TinyBERT stratified data-size ablation across three subsets: 25%, 50%, and 70%
- [x] Salma: Logged EM, F1, and training time for each TinyBERT data-size configuration

#### Key Decisions
- Used stratified sampling by article title for TinyBERT data-size ablation to preserve topic coverage across subsets
- Kept all other hyperparameters fixed across ablation configurations to isolate the effect of each variable

#### Results — Sequence Length Ablation

| Model | EM @ 128 | EM @ 256 | EM @ 384 | Latency @ 128 (ms) | Latency @ 256 (ms) | Latency @ 384 (ms) |
|-------|----------|----------|----------|--------------------|--------------------|-------------------|
| DistilBERT | 65.40% | 65.60% | 65.60% | 56.43 | 49.87 | 71.35 |
| TinyBERT | 34.20% | 27.80% | 27.40% | 10.03 | 9.21 | 12.46 |

#### Results — TinyBERT Layer-Freezing Ablation

| Configuration | EM (%) | F1 (%) | Val Loss | Trainable Params (M) | Training Time (hrs) | Latency (ms) |
|---|---|---|---|---|---|---|
| Freeze layer 0 | 25.00 | 35.12 | 1.79 | 13.11 | 1.38 | 3.27 |
| Freeze layers 0-1 | 13.00 | 18.23 | 2.01 | 11.97 | 1.38 | 3.15 |
| Freeze layers 0-3 | 1.50 | 7.30 | 4.13 | 9.68 | 1.31 | 3.22 |

#### Results — TinyBERT Data-Size Ablation

| Data Size | Chunks | EM (%) | F1 (%) | Training Time (hrs) |
|---|---|---|---|---|
| 25% | 12,631 | 18.00 | 26.52 | 0.42 |
| 50% | 25,263 | 23.50 | 33.06 | 0.74 |
| 70% | 35,368 | 24.00 | 34.41 | 1.00 |

---

## Week 5

### Date: 31/05/2026 - 06/06/2026

#### Progress
- [x] Sherouk: Implemented layer-freezing ablation for DistilBERT across three configurations: full fine-tuning, freeze embedding + 2 transformer layers, and freeze embedding + 4 transformer layers
- [x] Sherouk: Ran all three freezing configurations and logged EM, F1, trainable parameters, training time, and inference latency for each
- [x] Sherouk: Documented freezing results and identified emb+2layers as the recommended configuration based on cost-benefit analysis

#### Key Decisions
- Tested three freezing levels to isolate the point where frozen layers start hurting answer quality
- Due to compute limitations, ran a coarse search only — a fine-grained hyperparameter search was planned but not feasible within available resources
- Chose to evaluate on the same validation split used in the baseline to ensure fair comparison

#### Issues & How They Were Resolved
- **Issue:** Compute budget did not allow for a full fine-grained hyperparameter search across freezing configurations.  
  **Resolution:** Ran a coarse search only, testing three fixed configurations. Results were still sufficient to draw clear conclusions about the trade-off between frozen layers and answer quality.  
  **Resolved by:** Sherouk

#### Results — DistilBERT Layer-Freezing Ablation

| Configuration | EM (%) | F1 (%) | Trainable Params (M) | Training Time (hrs) | Latency (ms) |
|---|---|---|---|---|---|
| Full fine-tuning | 63.0 | 72.8 | 66.36 | 0.79 | 7.67 |
| Freeze emb + 2 layers | 60.5 | 72.3 | 28.35 | 0.62 | 7.23 |
| Freeze emb + 4 layers | 48.0 | 58.4 | 14.18 | 0.47 | 7.29 |

---

## Week 6

### Date: 07/06/2026 - 13/06/2026

#### Progress
- [x] Sherouk: Implemented data-size ablation for DistilBERT by training on three subsets of the training data: 15%, 25%, and 35%
- [x] Sherouk: Ran all three data-size configurations and logged EM, F1, and training time for each
- [x] Sherouk: Identified 25% as the practical sweet spot based on the clear drop in returns beyond that threshold
- [x] Sherouk: Documented findings showing diminishing returns pattern consistent across both DistilBERT and TinyBERT

#### Key Decisions
- Chose 15%, 25%, and 35% as the three data size levels to cover a meaningful range without exceeding compute budget
- Due to compute limitations, each configuration was run once only — the original plan was to aggregate results across 3 runs per configuration to report mean and standard deviation, but this was not feasible

#### Issues & How They Were Resolved
- **Issue:** Compute budget did not allow running each data-size configuration multiple times, which was the original plan to ensure statistical reliability.  
  **Resolution:** Each configuration was run once only. Results show a clear and consistent trend, but confidence intervals could not be reported. This is acknowledged as a limitation in the final report.  
  **Resolved by:** Sherouk

#### Results — DistilBERT Data-Size Ablation

| Data Size | Examples | EM (%) | F1 (%) | Training Time (hrs) |
|---|---|---|---|---|
| 15% | 7,510 | 38.5 | 46.8 | 0.16 |
| 25% | 12,517 | 55.5 | 66.0 | 0.23 |
| 35% | 17,523 | 58.0 | 67.8 | 0.30 |

---

## Week 7

### Date: 14/06/2026 - 20/06/2026

#### Progress
- [x] Sherouk: Consolidated all DistilBERT ablation results and wrote the methodology and results sections of the final report covering layer-freezing and data-size experiments
- [x] Sherouk: Reviewed all ablation findings across both models and contributed to the discussion section on cost-benefit trade-offs
- [x] Mostafa: Wrote related work section covering DistilBERT, TinyBERT, and knowledge distillation literature
- [x] Mostafa: Wrote discussion and conclusion sections synthesizing findings across all experiments
- [x] Mostafa: Compiled and formatted all references
- [x] Salma: Wrote TinyBERT results and analysis sections covering layer-freezing and data-size ablations
- [x] Salma: Collected and verified all TinyBERT efficiency metrics for the final comparison table
- [x] Salma: Proofread and formatted the full report for submission

#### Key Decisions
- Decided to acknowledge single-seed limitation explicitly in the limitations section rather than omitting it
- Chose to present all ablation results in unified tables for easier cross-model comparison

#### Final Results Summary

| Model | EM (%) | F1 (%) | Latency (ms) | Params |
|-------|--------|--------|--------------|--------|
| BM25 Baseline | 0.00 | 12.75 | N/A | N/A |
| TF-IDF Baseline | 0.11 | 15.25 | N/A | N/A |
| TinyBERT | 27.40 | 36.02 | 12.46 | 14.5M |
| DistilBERT | 66.80 | 77.93 | 104.0 | 66M |

#### Key Takeaways
- Freezing the embedding layer plus two transformer layers retains 99.3% of DistilBERT F1 while reducing trainable parameters by 57.3%
- 25% of training data captures most of DistilBERT performance; returns diminish significantly beyond that threshold
- DistilBERT is the right choice when accuracy matters; TinyBERT is 8.3x faster and better suited for latency-critical deployment
- Layer freezing saves training time but does not reduce inference latency in either model

#### Issues & How They Were Resolved
- **Issue:** Inconsistent formatting across sections written by different team members.  
  **Resolution:** Salma did a full pass to unify formatting, table styles, and citation format across the entire report.  
  **Resolved by:** Salma
