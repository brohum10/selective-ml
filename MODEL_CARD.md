# Model card

## Intended use

An educational, CPU-only example of supervised learning and uncertainty evaluation. Input is a batch of flattened 8×8 grayscale digits with values in [0,16]. Output is a digit, confidence, and accept/review decision. Not intended for scanned documents, photographs, identity checks, medical use, or automated consequential decisions.

## Training and evaluation

A 4,810-parameter ReLU MLP trained with categorical cross-entropy and Adam. Fixed hyperparameters and stratified random-row splits are specified in the README and source. Pixel scaling uses a known constant, not statistics fit to the complete dataset. The calibration and policy splits are separate from training and test. The dataset has no exact duplicate images, but writer identity is unavailable here: random row splits cannot establish unseen-writer generalization.

Seeds 42/43/44 are all reported; seed 42 is the exported artifact by convention, not because it performed best. At seed 42, test accuracy is 97.45%; the 2,000-resample percentile bootstrap interval is approximately 95.27–98.91%. This interval is conditional on that test sample/model and excludes training and dataset-selection uncertainty.

## Failure modes

- Only 275 test observations per seed; estimates and ten-bin ECE are noisy.
- Calibration improved the calibration objective by construction, but worsened test NLL in two seeds. It must not be advertised as universally improving confidence.
- The empirical 98% accepted-accuracy target is missed on two held-out splits. There is no distribution-free risk guarantee.
- High confidence can occur on unfamiliar inputs. Range/schema checks are not out-of-distribution detection.
- Fixed Gaussian corruption reduces accuracy to 90.18–94.18%; other distortions were not evaluated.
- Digit class balance does not measure demographic fairness. No demographic fairness claims are made.
- JSON avoids pickle code execution, but this CLI is not a hardened network service; callers must impose file-size/resource limits if serving untrusted requests.

## Reproducibility

Version-pinned dependency, checksum-verified data, recorded Python/NumPy versions, deterministic seeds, exported row IDs and model parameters. Floating-point results may vary slightly by BLAS/platform. Re-running experiments writes only the requested output directory. Inspect all metrics rather than selecting favorable seeds or silently editing the reference output.
