This is a submission of version 0.3.6


## Test environments
* local Windows install, R 4.5.3

## R CMD check results
0 errors | 0 warnings | 1 note



## Major package updates in 0.3.6

- rebuilt pre_process() to ensure pipeline resilience on massive, messy datasets. The function now uses a chunked processing architecture (default chunk_size = 5000).
