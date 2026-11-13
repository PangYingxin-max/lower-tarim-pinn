# Lower Tarim water-record audit

## Bottom line

The water workbook you supplied is not a clean official annual record of lower-Tarim ecological conveyance. The 2021-2023 rows say `Tower Management Bureau Annual Report + Trend Projection`, and the workbook's cumulative column is only an arithmetic running sum of its own allocation column. It is not an observed cumulative discharge series.

The workbook therefore has three separate problems:

1. The 2021-2023 values are partly trend-extended and cannot be called basin-authority records.
2. The column named `Total water discharge (billion m3)` is actually a running sum of the preceding allocation column, not an independent measured discharge series.
3. The unit label `billion m3` is wrong for the numbers. Values such as 1.02, 3.48 and 7.20 are in `10^8 m3` (亿立方米), not billion cubic metres.

The workbook values also fail simple official checkpoints. For example, the official cumulative release before the 2020 conveyance was 81.6e8 m3, but the workbook already reaches 97.84e8 m3 in 2019. Official cumulative release through 2023 was 98.25e8 m3, but the workbook reaches 128.31e8 m3. This means the whole 2000-2020 block needs source-by-source verification, not only the 2021-2023 rows.

## Verified 2021-2023 values

| Year | Official or verifiable value (10^8 m3) | Basis |
|---|---:|---|
| 2021 | 3.48 | 2021 annual chronicle: the 22nd lower-Tarim conveyance delivered 3.48e8 m3. |
| 2022 | 7.20 | 2022 annual chronicle: full-year ecological water delivered was 7.2e8 m3. |
| 2023 | 3.15 (derived) | Official cumulative total through 2023 was 98.25e8 m3; before the 2022 conveyance the cumulative total was 87.9e8 m3; 2022 was 7.2e8 m3; therefore 2023 = 98.25 - 87.9 - 7.2 = 3.15e8 m3. |

Do not merge the lower-reach conveyance value with the whole-basin Populus forest water figure. The 2021 annual chronicle also reports 14.3e8 m3 of whole-basin Populus forest ecological water, of which the 3.48e8 m3 lower-reach conveyance is one component. The 2023 chronicle reports 11.03e8 m3 of whole-basin Populus forest water; that is not the lower-reach conveyance.

## Where to find the records

1. Tarim River Basin Administration annual chronicles: `政务之窗 -> 大事记` on `https://www.tahe.gov.cn/`. Read the 2021, 2022, 2023 and 2024 pages and record the exact sentence containing the lower-reach conveyance.
2. Xinjiang Water Resources Department: `水利要闻` and `地州水事` pages on the Xinjiang government and water-department sites.
3. Official news reports from Xinhua, People's Daily, CCTV and the Xinjiang government. Use them as corroboration, not as the primary source, unless the bureau page is unavailable.
4. Xinjiang Statistical Yearbook, China Water Resources Yearbook and Xinjiang water-resources bulletins for a tabular cross-check.
5. Peer-reviewed papers from 2024-2025 that report annual 2021-2023 conveyance values.

Useful Chinese search strings:

```text
2021 塔里木河 下游 生态输水 3.48 亿立方米
2022 塔里木河 下游 生态输水 7.2 亿立方米
2023 塔里木河 下游 生态输水 98.25 亿立方米
site:tahe.gov.cn 大事记 生态输水
site:xinjiang.gov.cn 塔里木河 生态输水 2022
塔里木河流域管理局 2023 年大事记 生态输水
```

For each year, archive the source title, URL, page or paragraph, reporting period, and a SHA256 hash of the saved PDF or HTML snapshot. Record whether the number is a direct annual report, a cumulative checkpoint, or a derived value.

## Consequence for the model

The archived model outputs were generated with 7.35, 7.18 and 7.42 for 2021-2023. Replacing them with 3.48, 7.20 and 3.15 changes the holdout forcing and the normalized forcing values. The PINN and downstream holdout, bifurcation and sensitivity results must therefore be re-run before the manuscript can claim that its forcing series is an official record. Do not patch only the supplementary table while leaving the reported model metrics unchanged.
