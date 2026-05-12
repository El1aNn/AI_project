# MRPC Report Examples

## Two Correct Paraphrase Pairs

- Sentence 1: He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .
  Sentence 2: " The foodservice pie business does not fit our long-term growth strategy .
  Model prediction: equivalent; correct: True.
  Comment: the two sentences describe the same event or claim with different wording.
- Sentence 1: The AFL-CIO is waiting until October to decide if it will endorse a candidate .
  Sentence 2: The AFL-CIO announced Wednesday that it will decide in October whether to endorse a candidate before the primaries .
  Model prediction: equivalent; correct: True.
  Comment: the two sentences describe the same event or claim with different wording.

## Two Incorrect Paraphrase Pairs

- Sentence 1: Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war .
  Sentence 2: His wife said he was " 100 percent behind George Bush " and looked forward to using his years of training in the war .
  Model prediction: equivalent; correct: False.
  Comment: the pair shares some surface words but differs in facts, relation, or meaning.
- Sentence 1: The dollar was at 116.92 yen against the yen , flat on the session , and at 1.2891 against the Swiss franc , also flat .
  Sentence 2: The dollar was at 116.78 yen JPY = , virtually flat on the session , and at 1.2871 against the Swiss franc CHF = , down 0.1 percent .
  Model prediction: equivalent; correct: False.
  Comment: the pair shares some surface words but differs in facts, relation, or meaning.
