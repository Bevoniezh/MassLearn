# Sample Template Guide

This guide explains the purpose of the sample template used in MassLearn's untargeted metabolomics pipeline and how to complete it for consistent downstream analysis.

## Purpose of the Sample Template
- Captures experimental metadata for every sample so that multivariate statistical analyses can separate variation driven by treatments from variation introduced by other experimental factors.
- Enables consistent interpretation of feature abundance changes by tracking who prepared samples, when they were prepared, and what treatment levels were applied at each factor.
- Provides a record of experiments that can be reused across studies, ensuring comparable metadata for all samples.

## How the Template Is Processed
- The untargeted metabolomic pipeline loads the template during metadata ingestion to map each input file to its experimental context (see `Sample_Template.csv` for the expected columns and examples).
- The pipeline expects the `LINE` column to categorize samples. At least one blank signal file must be included and labeled with `BLANK` in the `LINE` column; MassLearn uses this designation to target and handle blank signals during processing.
- Treatment columns (e.g., `Treatment_1`, `Treatment_2`, etc.) are read as distinct experimental factors, allowing multilevel designs to be modeled during downstream analyses.

## Completing the Template
1. **Samples**: Provide the unique sample file names that correspond to your raw data.
2. **Experiment_title**: Name the experiment for grouping related runs.
3. **Preparation_date**: Record the date each batch was prepared to track temporal effects.
4. **Investigator / Technician**: Identify who designed and who prepared the samples so personnel-driven variability can be evaluated.
5. **Line**: Categorize each sample (e.g., biological line or condition). Use `BLANK` for blank signal files, and ensure at least one blank is present.
6. **Treatment columns**: Fill in each treatment level applied to the sample across the available columns. Use `NOTHING` when a treatment slot does not apply.

## Importance for Statistical Analysis
- Statistical analysis is mandatory in the pipeline: the captured metadata feeds into multivariate analyses that attribute observed variance to treatments or to other experimental aspects (preparer, preparation date, batch line, etc.).
- By supplying complete metadata, the pipeline can highlight which factors drive differences in feature abundances and distinguish treatment effects from confounding sources.

## Usage Tips
- Keep naming consistent across batches to make cross-experiment comparisons straightforward.
- Include blank samples early in acquisition so blank-related signals can be accurately targeted and removed.
- Review `Sample_Template.csv` before editing to follow the expected formatting and delimiters.
