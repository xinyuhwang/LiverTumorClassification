# MCT-LTDiag Dataset Audit

- Dataset root: `/projects/insightx-lab/Liver_Data`
- Cases audited: 517
- Cases discovered on disk: 517
- Cases listed in metadata: 517
- Full completeness check: True
- Cases with errors: 11
- Cases with warnings: 450

## Tumor Types

- BCLM: 115
- CRLM: 103
- HCC: 103
- HH: 96
- ICC: 100

## Outputs

- `cases.csv`: one row per case
- `issues.csv`: machine-readable validation failures
- `case_details.json`: complete geometry and intensity details
- `qc/`: registered four-phase overlays
