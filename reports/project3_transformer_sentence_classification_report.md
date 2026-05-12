# Project 3 Report: Transformer Sentence Classification

Student name(s): ____________________

Student ID(s): ____________________

## 1. Overview

This report presents the implementation and evaluation of the Transformer sentence classification experiment. The assignment requires two settings:

- Task A: run the original character-level Transformer classifier.
- Task B: perform text processing with Chinese word segmentation and run the Transformer classifier again.

The final outputs are stored under `experiment_data/submission/project3/`. The directory includes model metrics, prediction files, misclassified documents, and the required comparison examples between Task A and Task B.

## 2. Data and Experimental Settings

The dataset is the THUCNews sentence classification data included in the project source. The original data contains Chinese news titles and category labels. The final run used:

| Split | Number of documents |
|---|---:|
| Train | 45,000 |
| Dev | 2,500 |
| Test | 2,500 |

The model is the Transformer classifier from the provided project source. Both Task A and Task B used the same model architecture and training hyperparameters, so the comparison mainly reflects the effect of input tokenization.

Main settings:

| Item | Task A | Task B |
|---|---:|---:|
| Input unit | Chinese characters | Jieba word segmentation |
| Epochs | 6 | 6 |
| Batch size | 128 | 128 |
| Pad size | 32 | 32 |
| Learning rate | 5e-4 | 5e-4 |
| Vocabulary size | 4,205 | 10,002 |
| Test documents | 2,500 | 2,500 |

Task B first segments the original Chinese titles with `jieba`, then trains the same Transformer model on the segmented word sequence. This increases the vocabulary size because multi-character words and named entities become independent tokens.

## 3. Task A: Character-Level Transformer

Task A treats each Chinese character as a token. This approach has a smaller vocabulary and is robust to rare words, because unseen multi-character words can still be represented through their component characters.

Training curve:

| Epoch | Train loss | Train accuracy | Dev loss | Dev accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.2229 | 0.5938 | 1.1584 | 0.6744 |
| 2 | 0.7580 | 0.7550 | 0.9139 | 0.7484 |
| 3 | 0.6389 | 0.7915 | 0.7567 | 0.7908 |
| 4 | 0.5700 | 0.8133 | 0.8894 | 0.7716 |
| 5 | 0.5211 | 0.8285 | 0.7927 | 0.7840 |
| 6 | 0.4661 | 0.8452 | 0.6834 | 0.8112 |

Final Task A test result:

| Metric | Value |
|---|---:|
| Test loss | 0.5075 |
| Test accuracy | 0.8480 |
| Misclassified documents | 380 |

The full prediction file is `experiment_data/submission/project3/taskA_char/test_predictions.csv`, and the misclassified documents are saved in `test_predictions_misclassified.csv`.

## 4. Task B: Word-Segmented Transformer

Task B uses word-level tokens after Chinese segmentation. This allows the model to preserve complete domain terms such as school names, exam terms, and topic keywords. However, it also increases vocabulary sparsity and can introduce segmentation errors.

Training curve:

| Epoch | Train loss | Train accuracy | Dev loss | Dev accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.6431 | 0.4450 | 1.2036 | 0.6208 |
| 2 | 1.1501 | 0.6226 | 1.0009 | 0.6756 |
| 3 | 0.9503 | 0.6879 | 0.8797 | 0.7248 |
| 4 | 0.8286 | 0.7296 | 0.7851 | 0.7596 |
| 5 | 0.7497 | 0.7548 | 0.7426 | 0.7744 |
| 6 | 0.6690 | 0.7826 | 0.6890 | 0.7836 |

Final Task B test result:

| Metric | Value |
|---|---:|
| Test loss | 0.5356 |
| Test accuracy | 0.8256 |
| Misclassified documents | 436 |

The full prediction file is `experiment_data/submission/project3/taskB_word/test_predictions.csv`, and the misclassified documents are saved in `test_predictions_misclassified.csv`.

## 5. Comparison Between Task A and Task B

The final test comparison is:

| Setting | Test accuracy | Test loss | Misclassified documents |
|---|---:|---:|---:|
| Task A: character-level | 0.8480 | 0.5075 | 380 |
| Task B: word-level | 0.8256 | 0.5356 | 436 |

Task A performs slightly better overall. The character-level model reaches higher test accuracy and makes fewer mistakes. This suggests that, for this dataset and model size, character-level input is more robust. Chinese news titles are short, and character-level tokenization avoids segmentation errors and out-of-vocabulary word problems.

Task B is still successful and useful. It correctly classifies some documents that Task A misses, especially when word segmentation preserves topic-specific phrases.

The comparison files are:

- `A_wrong_B_right.csv`: 176 documents misclassified in Task A but correct in Task B.
- `B_wrong_A_right.csv`: 232 documents misclassified in Task B but correct in Task A.
- `comparison_examples.md`: selected examples for the report.

## 6. Documents Misclassified in Task A but Correct in Task B

The following three examples were wrong in Task A but correct in Task B:

| Index | Text | True label | Task A prediction | Task B prediction |
|---:|---|---|---|---|
| 5 | 本科未录取还有这些路可以走 | 教育 | 房产 | 教育 |
| 7 | 去新西兰体验舌尖上的饕餮之旅(组图) | 教育 | 房产 | 教育 |
| 28 | 调查显示：29.5%的人不满意当年所选高考专业 | 教育 | 股票 | 教育 |

Brief explanation:

In these examples, word segmentation helps preserve topic words related to education, such as admission, undergraduate study, college entrance examination, and major selection. At the character level, the model may focus on isolated characters that also appear in other domains. With word-level segmentation, the model can treat multi-character education-related terms as complete semantic units, so Task B predicts the correct label.

## 7. Documents Misclassified in Task B but Correct in Task A

The following three examples were wrong in Task B but correct in Task A:

| Index | Text | True label | Task A prediction | Task B prediction |
|---:|---|---|---|---|
| 12 | 研究生办替考网站续：幕后老板年赚近百万(图) | 教育 | 教育 | 社会 |
| 15 | 公共英语(PETS)写作中常见的逻辑词汇汇总 | 教育 | 教育 | 游戏 |
| 17 | 九成外国人愿继续在日生活 六成留学生未返校 | 教育 | 教育 | 房产 |

Brief explanation:

These examples show the weakness of word segmentation. Some titles contain mixed signals, rare names, acronyms, or terms that may be segmented in a way that weakens the education topic. Character-level input is more robust because it can still use meaningful individual characters such as `生`, `考`, `学`, and `校`. Therefore, Task A avoids some segmentation-related errors.

## 8. Confusion Matrix Discussion

The test split mainly contains three labels with nonzero support: `财经`, `教育`, and `娱乐`. Other classes appear in the model label list but have zero support in this particular test subset, so their precision and recall are reported as zero by the classification report.

For Task A:

- `财经`: 422 out of 500 were correctly predicted.
- `教育`: 914 out of 1,000 were correctly predicted.
- `娱乐`: 784 out of 1,000 were correctly predicted.

For Task B:

- `财经`: 423 out of 500 were correctly predicted.
- `教育`: 900 out of 1,000 were correctly predicted.
- `娱乐`: 741 out of 1,000 were correctly predicted.

Both models perform well on `财经` and `教育`. The largest weakness is `娱乐`, where many examples are confused with technology, games, real estate, or education. This may happen because entertainment titles often include people, events, and commercial terms that overlap with other news domains.

## 9. Conclusion

Both required tasks ran successfully:

- Task A successfully trained and evaluated the character-level Transformer.
- Task B successfully performed Chinese word segmentation and trained the word-level Transformer.
- Misclassified documents for both tasks were computed and saved.
- Required A-wrong/B-right and B-wrong/A-right examples were identified.

Task A achieved better final performance with 0.8480 test accuracy, compared with 0.8256 for Task B. Although Task B improves some education-related examples by preserving complete word units, the character-level model is more stable overall on this dataset. The final conclusion is that character-level Transformer classification is the better setting for this particular THUCNews experiment, while word segmentation remains useful for interpreting some individual cases.
