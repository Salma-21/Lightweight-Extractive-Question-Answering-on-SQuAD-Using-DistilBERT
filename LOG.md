# Development Log - Project 14
# Efficient Extractive Question Answering on SQuAD with Distilled Models

**Team Members:** Sherouk | Mostafa | Salma  
**Course:** CISC 867 Deep Learning  
**Repository:** https://github.com/Salma-21/Lightweight-Extractive-Question-Answering-on-SQuAD-Using-DistilBERT#

---

## Week 1

### Date: 03/05/2026 - 09/05/2026

#### Progress
#### Progress
- [x] Sherouk: Loaded SQuAD v1.1, inspected JSON structure, displayed 5 examples, and ran span alignment audit on all answer offsets
- [x] Sherouk: Implemented tokenizer with special tokens, truncation, and stride/window approach (doc_stride=128, max_length=512) and verified chunk shapes
- [x] Sherouk: Implemented answer start and end offset mapping across all chunks and fixed early-return bug that corrupted labels for multi-chunk examples
- [x] Sherouk: Split dataset at article level into 80/10/10 train/val/test, subset train to 50k QA pairs, and encoded all three splits
- [x] Sherouk: Implemented BM25 baseline using rank_bm25 with official SQuAD scoring (max over all gold answers per question)
- [x] Sherouk: Computed and reported baseline results (BM25: EM=0.12%, F1=14.19%) and ran diagnostic confirming low scores are inherent to sentence retrieval not a bug
- [x] Sherouk: Saved BM25 predictions to bm25_baseline_predictions.json for Student C error analysis and official eval script compatibility
- [x] Mostafa: Read DistilBERT paper (Sanh et al., 2019) and studied HuggingFace QA API (DistilBertForQuestionAnswering, TrainingArguments, Trainer)
- [x] Mostafa: Set up Google Colab environment; installed transformers==4.32.1, torch==2.0.1, datasets==2.13.0; resolved CUDA version conflict (CUDA 12.1 only)
- [x] Mostafa: Ran sanity-check forward pass on DistilBertForQuestionAnswering with dummy batch (batch_size=2, seq_length=384); verified output shapes: start_logits [2, 384], end_logits [2, 384]
- [x] Salma: Studied the SQuAD evaluation metrics, Exact Match and token-level F1
- [x] Salma: Used the SQuAD-style evaluation script functions to compute EM and F1
- [x] Salma: Built results-logging utilities to save baseline/model predictions, scores, and summaries to CSV

#### Key Decisions
- Chose subset size of 50K QA pairs because [reason, e.g., fits within CPU memory/time budget]
- Decided on an 80% / 10% / 10% train/val/test split following the standard SQuAD dataset approach ([Rajpurkar et al., 2016](https://stanford.edu)).
- Chose doc_stride=128 and max_length=512 for the strided tokenization window because
  DistilBERT has a hard architectural limit of 512 tokens, and a stride of 128 ensures
  consecutive windows overlap enough so the answer is never accidentally cut between two
  chunks.
- Chose BM25 (BM25Okapi) as the retrieval baseline over TF-IDF following established practice
  in QA literature, where it is used as the standard strong retrieval baseline on SQuAD
  ([Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)).

#### Issues & How They Were Resolved
- **Issue:** `offset_mapping` function contained a `return` statement inside the chunk loop,
  causing it to always exit after the first chunk regardless of whether the answer was found.
  For multi-chunk examples this produced incorrect `(0, 0)` positions pointing to `[CLS]`
  instead of the real answer span, silently corrupting training labels.<br>
  **Resolution:** Moved the `return` outside the loop by collecting all chunk results in a
  `results` list and returning it after the loop completes. Updated all three encoding loops
  (train, val, test) to iterate over `chunk_positions` using `enumerate` instead of a fixed
  range, ensuring each chunk gets its own correctly computed start and end position.<br>
  **Resolved by:** Sherouk.<br>

- **Issue:** BM25 baseline returned very low scores (EM: 0.12%, F1: 14.19%) which initially
  suggested a preprocessing or implementation bug.<br>
  **Investigation:** Ran a single example diagnostic that revealed BM25 was correctly
  identifying the relevant sentence in most cases. For example, given the question
  "When was Gaddafi born, and when did he die?", BM25 correctly retrieved the sentence
  containing the gold answer "1942 â€“ 20 October 2011" but returned the full sentence
  "1942 â€“ 20 October 2011), commonly known as Colonel Gaddafi..." resulting in EM=0
  and F1=0.45 for that example.<br>
  **Root Cause:** BM25 is a sentence retriever not a span extractor. SQuAD gold answers
  are short precise spans so a full sentence prediction will almost never produce an exact
  match and will always have low F1 due to the extra words. This is an inherent limitation
  of retrieval-based baselines on extractive QA tasks, not a bug.<br>
  **Resolution:** Confirmed the implementation is correct. The low numbers are expected
  and explainable. This finding motivates the use of DistilBERT which learns to extract
  the exact answer span at token level rather than returning the whole sentence.<br>
  **Resolved by:** Sherouk

- **Issue:** [No issues encountered in Week 1]

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

Student C evaluation was run on 500 validation examples for a fair baseline/model comparison under runtime limits.






