# Project 1 and 2 Report: CBOW and Skip-Gram Word Embeddings

Student name(s): ____________________

Student ID(s): ____________________

## 1. Overview

This report describes the implementation, training, and evaluation of two word embedding models: Continuous Bag-of-Words (CBOW) and Skip-Gram. Both models were trained successfully and evaluated with the three required parts of the assignment: K-nearest-neighbor evaluation, SimLex-999 golden standard evaluation, and analogy reasoning.

The final experimental outputs are stored under `experiment_data/submission/project1_2/`. The most important files are:

- `cbow.vec` and `skipgram.vec`: final word vectors.
- `cbow_knn_top10.csv` and `skipgram_knn_top10.csv`: KNN results for all covered query words.
- `cbow_simlex999.csv` and `skipgram_simlex999.csv`: golden-standard similarity evaluation.
- `cbow_analogy.csv` and `skipgram_analogy.csv`: analogy reasoning predictions.
- `evaluation_summary.json`: final metrics.

Both vector files have the header `11587 100`, meaning that the final vocabulary size is 11,587 and the embedding dimension is 100. This confirms that the submitted results are from the final run, not from the quick smoke test.

## 2. Data and Preprocessing

The training corpus is the NLTK Reuters corpus. The corpus was tokenized into lowercase English words and numbers with a regular-expression tokenizer. Tokens whose corpus frequency was lower than 5 were filtered out. The `<UNK>` token was used for out-of-vocabulary words. The final training setting used a maximum vocabulary of 30,000, but after frequency filtering the actual vocabulary size was 11,587.

Main preprocessing and training settings:

| Item | Value |
|---|---:|
| Corpus | NLTK Reuters |
| Token count | about 1,473,743 |
| Vocabulary size | 11,587 |
| Minimum count | 5 |
| Embedding dimension | 100 |
| Window size | 2 |
| Negative samples | 5 |
| Training samples per model | 1,000,000 |
| Epochs | 10 |
| Optimizer | Adam |

The Reuters corpus is mainly financial and news text, so many learned associations are influenced by economic and political language. This is useful for topic-related nearest-neighbor evaluation, but it also explains why some general semantic tasks, especially analogy reasoning, are difficult.

## 3. Model Implementation

The CBOW model predicts the center word from surrounding context words. Given the context words in the window, their input embeddings are averaged and used to predict the center word. The Skip-Gram model uses the opposite direction: it predicts surrounding context words from the center word.

Both models were trained with negative sampling. For a positive pair and several sampled negative words, the objective encourages the positive word-context score to be high and the negative scores to be low. Negative words are sampled from a unigram distribution raised to the 0.75 power, which is a common Word2Vec setting.

The two models differ in what kind of information they emphasize:

- CBOW averages context words, so it is usually more stable on frequent words and small corpora.
- Skip-Gram treats each center-context pair separately, so it often captures rarer word relations better when enough training data is available.

In this experiment, CBOW produced better overall scores on SimLex and analogy reasoning, while Skip-Gram still produced reasonable KNN neighborhoods.

## 4. Training Success

Both models trained successfully and produced final `.vec` files:

| Model | Vector file | Header | Status |
|---|---|---|---|
| CBOW | `experiment_data/submission/project1_2/cbow.vec` | `11587 100` | Success |
| Skip-Gram | `experiment_data/submission/project1_2/skipgram.vec` | `11587 100` | Success |

The existence of all evaluation CSV files also shows that the trained vectors could be loaded and used for downstream evaluation.

## 5. KNN Evaluation

The KNN evaluation computes cosine similarity between a query word and all words in the embedding vocabulary. For each query, the top 10 most similar words were output. The assignment query list contains 50 words; 46 of them were covered by the final vocabulary.

Overall KNN results:

| Model | Covered query words | Mean top-10 cosine |
|---|---:|---:|
| CBOW | 46 | 0.5942 |
| Skip-Gram | 46 | 0.5264 |

CBOW has a higher mean top-10 cosine than Skip-Gram. This means that, on average, the nearest neighbors of CBOW vectors are more tightly clustered around the query word. Higher cosine alone does not guarantee better semantic quality, but the KNN examples show that CBOW learned relatively coherent local neighborhoods.

The following four query words were selected by the student-ID sampling rule. The full top-10 results are saved in `cbow_knn_4_words_for_report.csv` and `skipgram_knn_4_words_for_report.csv`.

| Query | CBOW top neighbors | Skip-Gram top neighbors |
|---|---|---|
| gradually | inclined, wind, productivity, notable, trimming | shifting, liberalize, input, stimulating, postpone |
| near | short, medium, long, lowest, grounded | sideways, stored, hectic, middle, relation |
| road | contrary, degree, focus, specify, recommend | vigorously, texstyrene, viermetz, revalue, soaring |
| police | scientists, practical, zinn, spethmann, goria | fisheries, powder, yeo, sabah, charts |

Some neighborhoods are semantically meaningful. For example, Skip-Gram connects `gradually` with words such as `shifting`, `liberalize`, and `slowed`, which are plausible in economic news text. CBOW connects `near` with `short`, `medium`, and `long`, showing that it learned a length or distance-related local region. Other examples are less semantically clean, such as `police`, whose neighbors include names or unrelated domain words. This reflects the limited corpus size and the domain bias of Reuters.

## 6. Golden Standard Evaluation with SimLex-999

The SimLex-999 evaluation compares model similarity scores with human similarity judgments. For each covered pair, cosine similarity is converted to a 0-10 scale using:

```text
score = (cosine(word1, word2) + 1) * 5
```

The final metric is Spearman correlation between the human score and the model score. Spearman correlation is suitable because it measures whether the ranking of word-pair similarity is consistent with human ranking.

Results:

| Model | Covered pairs | Spearman correlation |
|---|---:|---:|
| CBOW | 390 | 0.1950 |
| Skip-Gram | 390 | 0.0951 |

CBOW performs better on this golden-standard evaluation. Its correlation is still moderate rather than high, but it is clearly above Skip-Gram in this run. The lower score is expected because the Reuters corpus is domain-specific and much smaller than the corpora usually used to train high-quality general-purpose embeddings.

Selected examples:

| Pair | Human score | CBOW scaled score | Skip-Gram scaled score |
|---|---:|---:|---:|
| depth - magnitude | 6.12 | 7.50 | 6.55 |
| accept - deny | 1.75 | 7.33 | 6.48 |
| say - participate | 3.82 | 6.65 | 6.31 |
| go - sell | 0.97 | 6.49 | 5.88 |
| liquor - band | 0.68 | 4.34 | 5.88 |
| winter - summer | 2.38 | 8.13 | 7.05 |
| acquire - obtain | 8.57 | 7.39 | 6.72 |
| borrow - want | 1.77 | 7.03 | 6.78 |

The pair `acquire - obtain` is a good example where both models assign a relatively high score, which agrees with the high human similarity score. In contrast, `accept - deny` and `winter - summer` receive high model scores even though SimLex gives relatively low similarity scores. This is a known distinction between semantic similarity and topical relatedness: antonyms and contrastive words often occur in similar contexts, so distributional models may assign them high cosine similarity even though humans judge them as semantically different.

## 7. Analogy Reasoning

The analogy task uses vector arithmetic:

```text
vec(D) = vec(B) - vec(A) + vec(C)
```

For each analogy question `A : B :: C : D`, the model predicts the word with the highest cosine similarity to `vec(B) - vec(A) + vec(C)`, excluding the words already present in the question.

Results:

| Model | Covered questions | Correct | Accuracy |
|---|---:|---:|---:|
| CBOW | 4,226 | 142 | 0.0336 |
| Skip-Gram | 4,226 | 5 | 0.0012 |

CBOW is much stronger than Skip-Gram on analogy reasoning in this experiment, although the absolute accuracy is still low. Analogy reasoning is a difficult evaluation because it requires consistent linear structure in the embedding space. With a relatively small Reuters corpus and only 10 epochs, the model learns useful local similarity but not enough global regularity for many analogy categories.

Selected analogy examples:

| Model | Category | A | B | C | Expected | Prediction | Correct |
|---|---|---|---|---|---|---|---|
| CBOW | gram5-present-participle | increase | increasing | go | going | going | Yes |
| CBOW | gram7-past-tense | seeing | saw | slowing | slowed | slows | No |
| CBOW | gram8-plural | computer | computers | machine | machines | please | No |
| CBOW | gram6-nationality-adjective | spain | spanish | netherlands | dutch | notably | No |
| Skip-Gram | gram5-present-participle | increase | increasing | go | going | chinese | No |
| Skip-Gram | gram8-plural | computer | computers | machine | machines | barbara | No |

The successful CBOW example `increase : increasing :: go : going` shows that the model can learn some morphological regularities. However, most analogy errors are unrelated words or corpus-specific proper nouns. This suggests that the embeddings are not regular enough for reliable analogy solving.

## 8. Discussion

The evaluation results show different strengths:

- CBOW performs better on KNN mean cosine, SimLex correlation, and analogy accuracy.
- Skip-Gram learns some useful neighbors, especially for `gradually`, but performs worse on global analogy structure.
- Both models cover 46 query words, 390 SimLex pairs, and 4,226 analogy questions, so evaluation coverage is sufficient.

The low analogy accuracy is not necessarily a training failure. Analogy reasoning is much harder than nearest-neighbor retrieval, and the training corpus is relatively small and domain-specific. Reuters text strongly emphasizes business, companies, markets, and politics, so embeddings are biased toward those contexts. This causes words that are topically related but not semantically similar to be close in vector space.

## 9. Conclusion

Both CBOW and Skip-Gram were implemented, trained, and evaluated successfully. CBOW achieved better final performance in this experiment:

- KNN mean top-10 cosine: 0.5942 for CBOW versus 0.5264 for Skip-Gram.
- SimLex Spearman: 0.1950 for CBOW versus 0.0951 for Skip-Gram.
- Analogy accuracy: 0.0336 for CBOW versus 0.0012 for Skip-Gram.

Therefore, CBOW is the stronger model for this submission. Skip-Gram still completed training and produced valid vectors, but its analogy performance was weak. For future improvement, I would train on a larger and more balanced corpus, increase epochs, tune the learning rate, and try a larger embedding dimension.
