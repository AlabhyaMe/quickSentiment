## Test environments
* local Windows install, R 4.4.0

## R CMD check results
0 errors | 0 warnings | 1 note

* Note: `checking for future file timestamps ... NOTE unable to verify current time`
  * Explanation: This is a harmless local network/firewall timeout issue when checking file modification dates. It does not affect the package build, source code, or functionality.

## Reverse dependencies
There are currently no reverse dependencies for this package.

## Major Changes in v0.3.1
* **Removed Dependency:** Dropped `caret` entirely. Replaced it with lightweight, native base-R logic for data splitting and confusion matrix evaluation to reduce installation footprint.
* **Bug Fixes:** Resolved a matrix alignment bug in the Random Forest TF-IDF prediction engine. 
* **Optimizations:** Migrated to native `dgCMatrix` structures and implemented strict column-subsetting inside the main pipeline to prevent memory spikes on massive datasets.