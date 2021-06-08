# ORF: OCaml Random Forests

Random Forests are one of the workhorse of modern machine learning.
Especially, they cannot over-fit to the training set, are
fast to train, predict fast and give you a reasonable model
even without optimizing the model's default hyper-parameters.

Using out of bag (OOB) samples, you can even get an idea
of a model's performance, without the need for a held out
(test) dataset.

Their only drawback is that RFs, being an ensemble model,
cannot predict values which are outside of the training set
range of values (this _is_ a serious limitation in case you
are trying to optimize or minimize something in order to discover
outliers, compared to your training set samples).

For the moment, this implementation will only consider a sparse vector of
integers as features. i.e. categorical variables would need to be
one-hot-encoded.

# Bibliography

Breiman, Leo. (2001). Random forests. Machine learning, 45(1), 5-32.
