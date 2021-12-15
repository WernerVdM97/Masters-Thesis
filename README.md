# MEng Industrial Engineering Thesis
## A genetic algorithm based model tree forest 

This thesis presents an ensemble approach that reduces the high variance error exhibited by
model trees that comprise multivariate non-linear models and increases their overall robustness.
The ensemble approach is conceptualised, tuned for, and evaluated against competing regression
models on ten separate benchmarking datasets. The ensemble, referred to as the model tree
forest (MTF), incorporates a hybrid genetic algorithm approach to construct structurally optimal
polynomial expressions (GASOPE) within the leaf nodes of greedy induced model trees that
form the base learners of the ensemble. Bootstrap aggregation, together with randomised
splitting feature spaces during tree induction, sufficiently decorrelates the base learners within
the ensemble, thereby reducing the variance error of MTF compared to that of a single model
tree whilst retaining the favourable low bias error that model trees exhibit. The multivariate
non-linear models that predicts the output enable MTF to produce approximations of highly
non-linear data The addition of ensembling methods passively combat overfitting brought forth
from the increased model complexity, compared to a previous implementation of GASOPE
within a tree structure which is shown to exhibit overfitting in specific cases. MTF produced
similar performance to an artificial feed-forward neural network and outperformed the M5 model
tree, an ensemble of M5 model trees and support vector regression.
