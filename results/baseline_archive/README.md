# Baseline archive

Four completed original-backbone baselines; images are not copied.
source_snapshot.tar.gz captures project source at archive time, not an asserted historical checkout.
AP columns retain each task's category set. Only common20_AP is a B/C common-category comparison.
Validation is not an independent test. Check baseline_audit.json for actual classifier widths.
M3FD legacy classifier width may exceed 7; preserve it in the backbone comparison.
SATR trained on M3FD has a different pretraining history from the official COCO detector.
Only checkpoint_best_total.pth is copied; live intermediate/resume checkpoints remain in place.
