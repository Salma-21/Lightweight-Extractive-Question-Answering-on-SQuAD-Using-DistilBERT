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
- [x] Sherouk: Computed and reported baseline results (BM25: EM=0.18%, F1=14.96%) and ran diagnostic confirming low scores are inherent to sentence retrieval not a bug
- [x] Sherouk: Saved BM25 predictions to bm25_baseline_predictions.json for Student C error analysis and official eval script compatibility
- [ ] Mostafa: Read DistilBERT paper and studied HuggingFace QA API
- [ ] Mostafa: Set up environment (dependencies, CPU config)
- [ ] Mostafa: Ran sanity-check forward pass on DistilBertForQuestionAnswering
- [ ] Salma: Studied SQuAD evaluation metrics (EM, F1)
- [ ] Salma: Implemented/adapted official SQuAD evaluation script
- [ ] Salma: Built results-logging utility (predictions to CSV)

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

- **Issue:** BM25 baseline returned very low scores (EM: 0.18%, F1: 14.96%) which initially
  suggested a preprocessing or implementation bug.<br>
  **Investigation:** Ran a single example diagnostic that revealed BM25 was correctly
  identifying the relevant sentence in most cases. For example, given the question
  "When was Gaddafi born, and when did he die?", BM25 correctly retrieved the sentence
  containing the gold answer "1942 – 20 October 2011" but returned the full sentence
  "1942 – 20 October 2011), commonly known as Colonel Gaddafi..." resulting in EM=0
  and F1=0.45 for that example.<br>
  **Root Cause:** BM25 is a sentence retriever not a span extractor. SQuAD gold answers
  are short precise spans so a full sentence prediction will almost never produce an exact
  match and will always have low F1 due to the extra words. This is an inherent limitation
  of retrieval-based baselines on extractive QA tasks, not a bug.<br>
  **Resolution:** Confirmed the implementation is correct. The low numbers are expected
  and explainable. This finding motivates the use of DistilBERT which learns to extract
  the exact answer span at token level rather than returning the whole sentence.<br>
  **Resolved by:** Sherouk

- **Issue:** [e.g., HuggingFace version conflict on one machine]  
  **Resolution:** [e.g., Pinned transformers==4.x.x in requirements.txt]  
  **Resolved by:** Mostafa

---

## Week 2

### Date: 10/05/2026 - 16/05/2026

#### Progress
- [ ] Mostafa: Wrote full training loop with loss, optimizer, scheduler
- [ ] Mostafa: Fine-tuned DistilBERT on subset; logged metrics per epoch
- [ ] Mostafa: Saved checkpoints and wrote inference script
- [ ] Mostafa: Created hyperparameter config file
- [ ] Salma: Ran evaluation against baseline and model outputs
- [ ] Salma: Produced comparison tables and plots
- [ ] Salma: Completed first-pass qualitative error analysis (10-15 examples)

#### Key Decisions
- Chose learning rate of [X] because [reason, e.g., default from HuggingFace examples, then tuned]
- Chose batch size of [X] due to CPU memory constraints
- Decided on [X] max sequence length due to [reason]
- [Any other decisions made this week]

#### Issues & How They Were Resolved

- **Issue:** [e.g., Evaluation script crashing on empty predictions]  
  **Resolution:** [e.g., Added null-answer fallback in inference script]  
  **Resolved by:** Salma

#### Preliminary Results
| Model | Exact Match | F1 Score |
|-------|-------------|----------|
| Baseline (BM25) | 0.12% | 14.19% |
| DistilBERT (fine-tuned) | [X]% | [X]% |



