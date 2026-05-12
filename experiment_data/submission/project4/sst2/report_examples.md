# SST2 Report Examples

## Two Correct Sentences

- it 's a charming and often affecting journey . 
  True label: positive; prediction: positive.
  Comment: the sentiment cue is relatively direct, so the fine-tuned model can classify it reliably.
- unflinchingly bleak and desperate 
  True label: negative; prediction: negative.
  Comment: the sentiment cue is relatively direct, so the fine-tuned model can classify it reliably.

## Two Incorrect Sentences

- we root for ( clara and paul ) , even like them , though perhaps it 's an emotion closer to pity . 
  True label: positive; prediction: negative.
  Comment: the sentence may contain ambiguity, contrast, or weak sentiment words that make the label harder to infer.
- pumpkin takes an admirable look at the hypocrisy of political correctness , but it does so with such an uneven tone that you never know when humor ends and tragedy begins . 
  True label: negative; prediction: positive.
  Comment: the sentence may contain ambiguity, contrast, or weak sentiment words that make the label harder to infer.
