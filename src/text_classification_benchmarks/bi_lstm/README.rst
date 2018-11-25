Bidirectional LSTM Classifier
-----------------------------

Bidirectional LSTMs are an extension of traditional LSTMs that can improve model performance
on sequence classification problems.

In problems where all timesteps of the input sequence are available, Bidirectional LSTMs train
two instead of one LSTMs on the input sequence. The first on the input sequence as-is and the
second on a reversed copy of the input sequence. This can provide additional context to the
network and result in faster and even fuller learning on the problem.


--test --checkpoint=clf-6000
--data-dir=balanced --model-name=model_bal --vocab-name=vocab_bal --summaries-name=summaries_bal