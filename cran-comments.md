This is a resubmission of version 0.3.1. 

Re-submission notes:
- Fixed the formatting of the NEWS.md file using standard Markdown bullet points to resolve the "No news entries found" NOTE from the previous automated incoming check.

## Test environments
* local Windows install, R 4.4.0

## R CMD check results
0 errors | 0 warnings | 1 note

* Note: `checking for future file timestamps ... NOTE unable to verify current time`
  * Explanation: This is a harmless local network/firewall timeout issue when checking file modification dates. It does not affect the package build, source code, or functionality.

## Major package updates in 0.3.1:
- Removed the 'caret' dependency entirely to reduce installation footprint.
- Refactored the pipeline to use memory-efficient dgCMatrix sparse matrices.
- Fixed a namespace issue by renaming prediction() to predict_sentiment().
- Added new underlying functions evaluation() and route_prediction() to modularize and streamline the package architecture.
- Added ROC/AUC threshold guidance and fixed bugs related to Naive Bayes and TF-IDF+RF combinations.
