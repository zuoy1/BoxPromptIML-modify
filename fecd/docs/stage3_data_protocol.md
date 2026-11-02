# FECD Stage 3 Internal Data and Evaluation Protocol

Version: `fecd-stage3-split-v1`; seed `20261029`; target diagnostic fraction 20% per dataset. The frozen identity manifest is `fecd/assets/stage3_split_manifest.json`, SHA-256 `7f18b459cded7cc2a8577167c5e2958cfac715691217a32bf457c8ae611cd66e`. It is a derived manifest and does not replace the locked 5,737-row training manifest.

## Split and source isolation

| Dataset | Locked train | train_fit | teacher_diag |
|---|---:|---:|---:|
| CASIAv2 | 5,123 | 4,098 | 1,025 |
| NIST16 | 414 | 332 | 82 |
| Columbia | 130 | 104 | 26 |
| Coverage | 70 | 56 | 14 |
| Total | 5,737 | 4,590 | 1,147 |

Grouping is dataset-scoped and transitive. CASIAv2 filename donor tokens are joined into connected components; the largest component contains 3,020 samples and is pinned to `train_fit`, because holding it out would exceed the planned diagnostic fraction. The selected CASIAv2 diagnostic groups contain 1,025 samples. This is a conservative filename proxy, not verified provenance. NIST16 uses the existing `group` base-source field (414 records, 52 groups); Columbia uses the `_sub_NN` scene number and deliberately ignores camera-pair text; Coverage uses its existing base-image `group`. Missing/ambiguous metadata is assigned a unique unresolved singleton and explicitly does not establish source isolation. The current manifest has all 5,737 rows assigned; no source token proxy was silently replaced with a claimed identity.

The known NIST16 cross-original-split risk remains: `world/NC2016_0359.jpg` includes historical train `NC2016_5290`/`NC2016_7882` and historical test `NC2016_0796`/`NC2016_9257`. Stage 3 only partitions historical train rows; this known relationship means `teacher_diag` is independent of `train_fit` by the recorded source group but the historical corpus itself is not independent of the locked test set.

The split manifest records sample key, image SHA-256, declared image path, source-group proxy, method and subset. It contains no GT paths or masks. Stable ordering and SHA-256 hashes are frozen in the JSON. Re-running the preparation command must reproduce the hash before diagnosis.

All future candidate student comparisons using `train_fit` must use identical images and steps. Any B0 trained on all 5,737 historical samples is only a historical reference, not a same-condition baseline. Restoring full-data training later would require a separate protocol and would no longer support this held-out diagnostic as an untouched validation set.

## Frozen analysis decisions (before teacher GT access)

Teacher mask threshold is `mu >= 0.5`; GT is nearest-neighbor resized from the prepared native binary mask to 224x224, then foreground is `>127`. K1 and K8 evaluate on exactly the same pixels. Primary high-consensus subset is K8 `c_sam >= 0.90` and `var <= 0.01`; these absolute cutoffs are fixed here and will not be optimized on GT. Box interior is union box mask eroded by 3 pixels; box edge is the remainder of the union box; box exterior is a 5-pixel dilation ring, with the remainder reported separately.

Error detection uses error score `1 - reliability` so higher means more likely wrong. Report AUROC, non-interpolated AUPRC and error prevalence; single-class subsets are N/A with reason. Report exact per-image metrics and pooled micro metrics using 4,096 uniform reliability bins (within-bin ties). A fixed hash-selected first 8 images per dataset also receive exact pooled sensitivity checks. Reliability coverage is selected by descending score with fractional selection within ties, at 10%, 25%, 50%, 75% and 90% of each method's identical candidate set; lower risk is better. AURC integrates the tie-aware expected risk curve over coverage. Positive and negative candidate sets are analyzed separately; positive coverage is always relative to SAM predicted-positive pixels. The five fixed methods are S0 `c_sam`, S1 `c_pos` with noise, S2 `c_pos` with JPEG, S3 `c_pos` with FFT, S4 `c_pos` with median fused evidence; S5 evidence alone scores positive-candidate errors as `1-r` and is diagnostic only. For negative candidates, the Rule-FECD score is exactly `c_sam`; S5 uses reliability `1-r`, because anomalous negative pixels are candidate misses. HP is additionally reported as a separate ablation and never enters S4.

The primary decision endpoint is the paired per-image AUROC difference S4 minus S0 among high-consensus predicted-positive pixels. Secondary endpoint is same-coverage error risk difference at 50% trusted-positive coverage. Bootstrap resamples whole source groups with replacement (2,000 replicates, seed `20261029`) and reports image-weighted macro difference with percentile 95% CI. Micro metrics are descriptive, not pixel-independent inference. Dataset-specific results are all reported; AUPRC is accompanied by error prevalence.

GO requires (1) positive primary AUROC difference with 95% CI lower bound above zero, (2) lower S4 error risk at matched 50% positive coverage with CI upper bound below zero, and (3) nonnegative matched-coverage effect in every dataset with at least 20 diagnostic images and 50 candidate errors. CONDITIONAL GO requires positive pooled point estimates but an interval crossing zero, adequate power limitations, or a domain-specific reversal that has a concrete follow-up test. Otherwise NO-GO. Severe source/codec/texture confounding or no evidence of incremental value is NO-GO. No rule or cutoff may be changed after teacher_diag GT access.

No IND/OOD test set is read for feature or threshold selection. GT access is restricted to the dedicated `audit_pseudo.py` diagnostic entry point and visualization outputs under `fecd/assets/stage3_diagnostics/`.

JPEG sensitivity uses Q75 and Q95 only on the same fixed 8 images per dataset; it is a diagnostic of codec dependence and never changes Q90, normalization or the candidate formulas. Area strata are GT fractions <1%, 1–10%, 10–30%, >=30%, descriptive only. Absolute trusted-score threshold is 0.90, with coverage reported alongside precision. Coverage entries selecting fewer than 10 expected pixels are N/A with reason. Machine-readable choices and their canonical hash are frozen in `fecd/diagnostics/protocol.py` and `fecd/assets/stage3_analysis_protocol.json`.

## Execution plan and stage gates

1. **3A**: freeze identities and analysis protocol; tests first for four descriptors, normalization, cache identity and scores; run a 12-image train_fit pilot; then generate and verify complete GT-free raw evidence; fit only train_fit and check distributions. Continue only with finite maps, unchanged source identity and zero unresolved cache errors.
2. **3B**: only after 3A passes, enable teacher_diag GT access in the dedicated audit entry point; evaluate all fixed methods, positive/negative/high-consensus and box conditions, same-coverage risks and K1/K8 segmentation.
3. **3C**: report every dataset and per-image result, source-group paired CIs, diagnostic successes/failures and source/texture/codec limitations; preserve protected-file hashes; run the existing regression checks and all stage3 tests/static checks; stop after the report, without student training.
