# NLP Projects 1-4 Report

Student name(s):

Student ID(s):

## 1. CBOW and Skip-gram Training

Describe the corpus, tokenization, vocabulary size, embedding dimension, window
size, negative sampling setting, epochs, and hardware.

Training result files:

- `project1_2_solution/outputs/cbow.vec`
- `project1_2_solution/outputs/skipgram.vec`

## 2. KNN Evaluation

Report the top-10 nearest neighbors for all 50 query words. Discuss the 4 words
selected by the student ID seed.

CBOW mean top-10 cosine:

Skip-gram mean top-10 cosine:

Comments:

## 3. SimLex-999 Golden Standard

Explain the formula:

```text
score = (cosine(word1, word2) + 1) * 5
```

CBOW covered pairs and Spearman correlation:

Skip-gram covered pairs and Spearman correlation:

Discuss the 20 sampled word pairs from each model.

## 4. Analogy Reasoning

Use:

```text
vec(D) = vec(B) - vec(A) + vec(C)
```

CBOW covered questions, correct answers, and accuracy:

Skip-gram covered questions, correct answers, and accuracy:

Discuss the 10 sampled examples from each model.

## 5. Project 3 Transformer

Task A: original character-level input.

Task B: word-level input after Chinese segmentation.

Task A test loss and accuracy:

Task B test loss and accuracy:

Include confusion matrices and classification reports from:

- `project3_solution/outputs/taskA_char/metrics.json`
- `project3_solution/outputs/taskB_word/metrics.json`

List 3 documents misclassified in Task A but correctly classified in Task B, with
brief explanation.

List 3 documents misclassified in Task B but correctly classified in Task A, with
brief explanation.

## 6. Project 4 BERT

SST-2 final validation accuracy/F1:

MRPC final validation accuracy/F1:

Discuss 4 SST-2 sentences: 2 correct and 2 incorrect.

Discuss 4 MRPC sentence pairs: 2 paraphrase and 2 non-paraphrase.

## 7. Conclusion

Compare CBOW vs Skip-gram, Transformer Task A vs Task B, and the two BERT
classification tasks. Summarize which model/task setting worked better and why.
