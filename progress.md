## Early Versions - Testing with Tiny dataset
1. Tested MutlinomialNB. Even with parameter optimization, achieved a score of ~30%
1. Tried experimental HistGradientBoostingClassifier. Crashed. I think the kind of data I have is wrong for it.
1. RandomForest, SVM and GradientBoosting took a while
1. RandomForest (~5 minutes) achieved ~35%
1. GradientBoosting is infeasible; just one pass of the algorithm took half an hour.
1. SVM (~10 minutes) achieved ~31%
