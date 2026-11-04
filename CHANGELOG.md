# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-11-04

### Added
- Bootstrap confidence intervals for AUC (`roc.bootstrap_auc`), with
  single-class resamples skipped and quantile-based 95% intervals.
- Multi-method majority-vote fusion (`ensemble.py`): per-cell vote collection,
  confusion-matrix metrics over all voting rules, and a fusion performance plot.
- Measurement-noise robustness analysis (`robustness.py`): AUC with bootstrap
  intervals and fusion sensitivity/specificity across configurable voltage-noise
  levels, delimiting the robustness boundary of each method.
- Synthetic-file tests for the Arbin, NASA, and Oxford readers that require no
  external dataset.
- Project tooling: `pre-commit` config, Dependabot, issue templates, and a pull
  request template.

### Changed
- CLI tests rewritten to invoke `main()` directly in addition to the subprocess
  wrapper, improving CLI coverage.
- Noise levels for the robustness sweep are centralized in the configuration
  (`synthetic.robustness_noise_levels`) instead of being hard-coded.

## [1.0.0]

### Added
- Classical signal-processing detection methods M1–M4 (relaxation shoulder,
  dQ/dV feature, coulombic efficiency, charge kink) with no deep learning.
- Synthetic battery generator with known plating ground truth and deterministic
  per-cell seeding.
- Data readers and preprocessing for Arbin, NASA, and Oxford datasets.
- ROC/AUC analysis, degradation alignment, and stress-condition sensitivity.
- CLI, reporting, PEP 621 packaging, Ruff linting, pytest suite, and CI.
