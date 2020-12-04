## Early Versions - Testing with Tiny dataset
1. Tested MutlinomialNB. Even with parameter optimization, achieved a score of ~30%
1. Tried experimental HistGradientBoostingClassifier. Crashed. I think the kind of data I have is wrong for it.
1. RandomForest, SVM and GradientBoosting took a while
1. RandomForest (~5 minutes) achieved ~35%
1. GradientBoosting is infeasible; just one pass of the algorithm took half an hour.
1. SVM (~10 minutes) achieved ~31%

Going with Random Forest from this point on

## Version 1 - Simple Tokenization - Tiny Dataset (All Languages)
- CV Scores: 0.36999321 0.37185657 0.36429861 0.3705796  0.36239384
- Average 0.36782436

Realized at this point that more complex linguistic analysis would be language-specific. Regenerated datasets to keep only English tweets

## Version 1 - Simple Tokenization - Tiny Dataset (English)
- CV Scores: 0.48410175 0.48186647 0.47763799 0.48405212 0.47794783
- Average 0.4811212298411828
