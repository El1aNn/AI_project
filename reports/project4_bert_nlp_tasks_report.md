# Project 4 Report: BERT and NLP Task Evaluation

Student name(s): ____________________

Student ID(s): ____________________

## 1. Overview

This report presents the BERT-based experiments for two NLP tasks:

- Task 1: SST-2 single-sentence sentiment analysis.
- Task 2: MRPC sentence-pair paraphrase detection.

The final outputs are stored under `experiment_data/submission/project4/`. The result directory contains final metrics, validation predictions, misclassified examples, and report examples for both tasks.

The model used in the final run is `prajjwal1/bert-mini`, with `bert-base-uncased` as the tokenizer. BERT-mini is smaller than full BERT, so it trains faster while still demonstrating the BERT fine-tuning workflow required by the project.

## 2. Experimental Setup

The implementation uses Hugging Face `datasets` and `transformers`. The model is loaded as `BertForSequenceClassification` with two output labels. Since the classification head is task-specific, the final classifier weights are initialized before fine-tuning.

Main settings:

| Item | SST-2 | MRPC |
|---|---:|---:|
| Task type | Sentiment classification | Paraphrase classification |
| Input | One sentence | Sentence pair |
| Labels | negative / positive | not equivalent / equivalent |
| Max sequence length | 64 | 100 |
| Epochs | 2 | 2 |
| Batch size | 32 | 32 |
| Learning rate | 3e-5 | 3e-5 |
| Warmup ratio | 0.1 | 0.1 |
| Model | BERT-mini | BERT-mini |

The final result files are:

- `experiment_data/submission/project4/sst2/final_metrics.json`
- `experiment_data/submission/project4/sst2/validation_predictions.csv`
- `experiment_data/submission/project4/sst2/validation_predictions_misclassified.csv`
- `experiment_data/submission/project4/sst2/report_examples.md`
- `experiment_data/submission/project4/mrpc/final_metrics.json`
- `experiment_data/submission/project4/mrpc/validation_predictions.csv`
- `experiment_data/submission/project4/mrpc/validation_predictions_misclassified.csv`
- `experiment_data/submission/project4/mrpc/report_examples.md`
- `experiment_data/submission/project4/summary.json`

## 3. Task 1: SST-2 Sentiment Analysis

SST-2 is a single-sentence sentiment classification task. The model predicts whether a movie review sentence is positive or negative.

Final SST-2 validation results:

| Metric | Value |
|---|---:|
| Validation loss | 0.3826 |
| Accuracy | 0.8589 |
| F1 | 0.8604 |
| Validation examples | 872 |
| Misclassified examples | 123 |

The final accuracy and F1 are both above 0.85, showing that BERT-mini learned the sentiment classification task effectively. The model performs well because many SST-2 examples contain direct sentiment cues such as positive adjectives, negative adjectives, or clear opinion phrases.

### 3.1 Correct SST-2 Examples

| Sentence | True label | Prediction | Explanation |
|---|---|---|---|
| it 's a charming and often affecting journey . | positive | positive | The words `charming` and `affecting` are direct positive sentiment cues. |
| unflinchingly bleak and desperate | negative | negative | The words `bleak` and `desperate` strongly indicate negative sentiment. |

These examples are easy for the model because the sentiment-bearing words are explicit and not heavily dependent on long context.

### 3.2 Incorrect SST-2 Examples

| Sentence | True label | Prediction | Explanation |
|---|---|---|---|
| we root for ( clara and paul ) , even like them , though perhaps it 's an emotion closer to pity . | positive | negative | The phrase contains both positive verbs such as `root for` and negative or ambiguous emotion such as `pity`, which may confuse the model. |
| pumpkin takes an admirable look at the hypocrisy of political correctness , but it does so with such an uneven tone that you never know when humor ends and tragedy begins . | negative | positive | The word `admirable` is positive, but the whole sentence criticizes the uneven tone. The model may over-focus on the local positive cue. |

The errors show that sentiment classification is harder when sentences contain contrast, irony, mixed opinions, or sentiment words whose meaning depends on the full sentence.

## 4. Task 2: MRPC Paraphrase Detection

MRPC is a sentence-pair classification task. The model predicts whether two sentences have the same meaning.

Final MRPC validation results:

| Metric | Value |
|---|---:|
| Validation loss | 0.5647 |
| Accuracy | 0.7157 |
| F1 | 0.8242 |
| Validation examples | 408 |
| Misclassified examples | 116 |

The F1 score is higher than the accuracy because MRPC is label-imbalanced: many pairs are paraphrases. The model is good at identifying equivalent pairs, but it sometimes predicts `equivalent` when two sentences share many surface words but differ in important details.

### 4.1 Correct MRPC Examples

| Sentence pair | True label | Prediction | Explanation |
|---|---|---|---|
| Sentence 1: He said the foodservice pie business doesn't fit the company's long-term growth strategy. Sentence 2: The foodservice pie business does not fit our long-term growth strategy. | equivalent | equivalent | The second sentence paraphrases the same claim with slightly different wording. |
| Sentence 1: The AFL-CIO is waiting until October to decide if it will endorse a candidate. Sentence 2: The AFL-CIO announced Wednesday that it will decide in October whether to endorse a candidate before the primaries. | equivalent | equivalent | Both sentences describe the same decision timing and organization. |

These examples are correctly classified because the sentence pairs preserve the same event and relation even though the wording changes.

### 4.2 Incorrect MRPC Examples

| Sentence pair | True label | Prediction | Explanation |
|---|---|---|---|
| Sentence 1: Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war. Sentence 2: His wife said he was "100 percent behind George Bush" and looked forward to using his years of training in the war. | not equivalent | equivalent | The two sentences share the phrase about training in the war, but the speaker and key claim are different. |
| Sentence 1: The dollar was at 116.92 yen against the yen, flat on the session, and at 1.2891 against the Swiss franc, also flat. Sentence 2: The dollar was at 116.78 yen, virtually flat on the session, and at 1.2871 against the Swiss franc, down 0.1 percent. | not equivalent | equivalent | The sentences look similar, but the numeric values and final movement differ. |

These errors show that paraphrase detection requires attention to factual details. BERT-mini can capture general semantic similarity, but it sometimes misses small numeric or attribution differences.

## 5. Comparison of the Two BERT Tasks

| Task | Accuracy | F1 | Main difficulty |
|---|---:|---:|---|
| SST-2 | 0.8589 | 0.8604 | Mixed or contrastive sentiment |
| MRPC | 0.7157 | 0.8242 | Fine-grained factual differences |

SST-2 has higher accuracy because the task is simpler: one sentence is classified into positive or negative sentiment. MRPC is more difficult because the model must compare two sentences and decide whether their meanings are equivalent. Surface similarity is not enough; the model must also track numbers, speakers, dates, and relations.

The results are still strong for a small model. BERT-mini fine-tunes quickly and reaches good performance on SST-2 and acceptable performance on MRPC. A larger pretrained BERT model would likely improve MRPC accuracy, but it would require more computation.

## 6. Output Files and Reproducibility

The code saves both full predictions and misclassified examples:

| Task | Full prediction file | Misclassified file |
|---|---|---|
| SST-2 | `validation_predictions.csv` | `validation_predictions_misclassified.csv` |
| MRPC | `validation_predictions.csv` | `validation_predictions_misclassified.csv` |

The saved CSV files make it possible to inspect every validation example, label, prediction, and correctness flag. The `report_examples.md` files contain selected examples used in this report.

## 7. Conclusion

Both BERT NLP tasks ran successfully:

- SST-2 final accuracy is 0.8589 and F1 is 0.8604.
- MRPC final accuracy is 0.7157 and F1 is 0.8242.
- Misclassified validation examples were computed and saved for both tasks.
- Representative correct and incorrect examples were analyzed.

The results show that BERT fine-tuning is effective for both single-sentence sentiment classification and sentence-pair paraphrase detection. SST-2 is easier and achieves higher accuracy, while MRPC is more sensitive to detailed semantic differences between two sentences.
