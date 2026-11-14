# GitHub release checklist for journal submission

## Must be in the public repository

- Reproducible analysis code.
- Processed, non-third-party input tables.
- GEE JavaScript export scripts and exact asset IDs.
- A CSV and Markdown source register with stable URLs, DOIs, licence notes, and missing-data explanations.
- Result tables and figure files needed to check the manuscript's claims.
- README, requirements, citation metadata, and code/data licences.
- A version tag and, preferably, a Zenodo DOI for the exact archived release.
- The final Data Availability Statement URL in both the manuscript and supplementary material.

## Must not be uploaded

- Raw Sentinel-2, Landsat, or MODIS scenes.
- Licensed PDFs, reports, and web snapshots whose redistribution is not permitted.
- Cover letter, editor correspondence, reviewer comments, or confidential submission material.
- Large repeated intermediate files, checkpoints, temporary build directories, and virtual environments.
- Any exploratory result that is not described in the manuscript without an explicit label.

## Recommended release process

1. Create a public GitHub repository with a neutral name such as lower-tarim-pinn.
2. Upload the contents of this github_release directory, not the whole working directory.
3. Check that no file is larger than 50 MB and that the repository is below 100 MB.
4. Replace placeholder repository links after the real URL is known.
5. Create a release tag, for example v1.0.1.
6. Connect the repository to Zenodo and archive the tagged release.
7. Put the Zenodo DOI, not the moving main branch, in the manuscript Data Availability Statement when possible.
8. Put the same URL/DOI in the supplementary Code Availability section.
9. Verify the repository from a fresh clone: install requirements, run the lightweight commands that do not need raw rasters, and open the archived result tables.
10. If the journal requires anonymous review, delay the public release or use an anonymized archive link until acceptance.

## Current limitations to disclose in the repository

- The water records are compiled from multiple sources.
- Tarim 2021--2023 are report/trend records, not a continuous station download.
- Heihe Zhengyixia water release covers 11 of 24 water years.
- The temporal PINN has negative holdout R2 and is not a forecasting model.
- The full calibrated system has no validated fold or Hopf point.
- The spatial result is pattern-level evidence on three fixed windows, not a basin-wide prediction.
