# H1b: FEM-Corrected Time-Resolved Noise Correlations

**Session:** Allen_2022-04-13  
**Neurons:** 49  
**Valid time bins:** 55  

## Tercile Analysis

| Tercile | fz_U | fz_C | Dz |
|---------|------|------|----|
| early | 0.095158 | 0.031376 | -0.063782 |
| mid | 0.087607 | -0.006946 | -0.094553 |
| late | 0.050061 | 0.002747 | -0.047314 |

## Sliding Window Spearman Correlations

Window size: 15 time bins, 41 windows

| Metric | rho | p-value | sig |
|--------|-----|---------|-----|
| fz_U (uncorrected) | -0.9531 | 7.4831e-22 | *** |
| fz_C (corrected) | -0.1049 | 5.1403e-01 | n.s. |
| Dz (correction magnitude) | 0.3838 | 1.3242e-02 | * |

## Key Findings

- Dz (correction magnitude) early=-0.063782, late=-0.047314
- FEM correction magnitude significantly increases with time (rho=0.3838, p=1.3242e-02)
- Uncorrected noise corr trend: rho=-0.9531, p=7.4831e-22
- Corrected noise corr trend: rho=-0.1049, p=5.1403e-01
