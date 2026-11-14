| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 3B BF16                  |   5.98 GiB |     3.21 B | MTL,BLAS   |      12 |           pp512 |      2288.01 ± 47.43 |
| llama 3B BF16                  |   5.98 GiB |     3.21 B | MTL,BLAS   |      12 |           tg128 |         50.10 ± 9.97 |
| llama 3B Q4_K - Medium         |   1.96 GiB |     3.21 B | MTL,BLAS   |      12 |           pp512 |     2005.99 ± 116.60 |
| llama 3B Q4_K - Medium         |   1.96 GiB |     3.21 B | MTL,BLAS   |      12 |           tg128 |        154.95 ± 3.75 |

build: 620d222 (1)
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 3B BF16                  |   5.98 GiB |     3.21 B | MTL,BLAS   |      12 |           tg128 |         66.69 ± 1.42 |
| llama 3B Q4_K - Medium         |   1.96 GiB |     3.21 B | MTL,BLAS   |      12 |           tg128 |        143.25 ± 7.12 |

build: 620d222 (1)
