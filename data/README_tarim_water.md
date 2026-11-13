# Tarim lower-reach ecological water conveyance, 2000-2023

Source file used by `gee_to_inputs.py`: `data/tarim_water_official_2000_2023.csv`.

## What the series is

The annual volume released from Daxihaizi Reservoir (大西海子水库总下泄量) for the
ecological water conveyance to the lower Tarim River, in `1e8 m3` (亿立方米).

## Where it comes from

* 2000-2020: Wang, Y., Alimujiang, Zhou, L., Yang, P., Feng, S., Wang, Y. (2023).
  塔里木河下游生态输水效应 [Ecological water conveyance effect in the lower reaches
  of Tarim River]. 中国沙漠 (Journal of Desert Research), 43(4), 231-240.
  doi:10.7522/j.issn.1000-694X.2023.00009. Table 2 lists all 21 conveyance events
  with the Daxihaizi release and the Taitema Lake inflow; the release column is
  aggregated here by year. The paper states that the water data were supplied by
  the Tarim River Basin Administration.
* 2021-2022: Tarim River Basin Administration annual chronicles (大事记).
  2021 = 3.48e8 m3 (22nd conveyance), 2022 = 7.20e8 m3 (full-year total).
* 2023: no annual chronicle entry was located. The value 3.15e8 m3 is retained
  as the reported volume for the 24th conveyance and is labelled `reported`.
  China News/China Xinjiang Net (26 April 2023) reports a cumulative release of
  95.13e8 m3 before that conveyance. The cumulative totals do not reconcile
  exactly with the event-by-event table because the accounting boundaries and
  rounding differ, so the row is not described as a direct station observation.

## Cross-checks that passed

| Check | Value |
| --- | --- |
| 2018 (19th conveyance), Table 2 | 70006 x1e4 m3 = 7.00e8 m3 |
| TRBA 2018 chronicle | "全年累计生态输水7亿立方米" |
| 2019 (20th conveyance), Table 2 | 46482 x1e4 m3 = 4.65e8 m3 |
| TRBA 2019 chronicle | "完成塔河下游生态输水4.64亿立方米" |
| 2020 (21st conveyance), Table 2 | 27934 x1e4 m3 = 2.79e8 m3 |
| TRBA 2020 chronicle | "第二十一次向塔河下游生态输水2.81亿立方米" |
| **Selected model value for 2020** | **2.81e8 m3**, the official annual chronicle value |

## Known gaps

* 2008 has no conveyance event in Table 2 (the 9th is October 2007, the 10th is
  December 2009). The CSV stores 0.0 and flags the row as `gap`.
* The running sum of the series to 2020 is 84.73e8 m3; the value quoted in the
  literature for the cumulative release before the 2020 conveyance is 81.6e8 m3.
  The two differ by the accounting boundary and by rounding in the event table.
* The 18th conveyance spans April 2017 to January 2018 and is assigned in full to
  2017, following the start year convention used in the source table.

## What this replaces

The 2020 row uses the official annual chronicle value (2.81e8 m3); the
event-table value (2.7934e8 m3) is kept as a cross-check in the note rather than
silently mixing the two accounting summaries.

`Lower Tarim River_R_2000-2023.xlsx` is no longer used anywhere in the pipeline.
Its "Ecological water allocation" column is not the downstream release (2000 =
1.02 against 3.26 here, 2020 = 8.52 against 2.81 here) and its 2021-2023 rows were
trend-extrapolated, not observed.
