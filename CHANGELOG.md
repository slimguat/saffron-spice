# Changelog

## [0.3.1] - 2026-05-04
### Added
- NaN-aware convolution helpers: `get_convolution_size` and `get_expanded_convolution_list` in `saffron/utils/nan_convolution.py`.
- Model visualization improvements and APIs in `saffron/fit_models/model_diagram.py`.

### Changed
- Replaced ad-hoc convolution sizing with shared, NaN-aware helpers used in raster fitting and postprocessing (`saffron/fit_functions/fit_raster.py`, `saffron/postprocessing/compo_reader.py`).
- Default convolution settings changed: `convolution_extent_list` -> `[1]`, `t_convolution_index` -> `1`; manager validates positive values.
- Refactor and cleanup in model generation and initialization routines (`saffron/init_handler/gen_inits.py`).
- Utilities cleaned and improved (verbosity helper `_vprint`, formatting, plotting helpers) in `saffron/utils/utils.py`.

### Fixed
- Multiple bug fixes in postprocessing (radiance/error estimation), Doppler gradient correction, and FIP map calculation.
- Memory-leak and data-handling fixes in L2/L3 pipeline and convolution/denoising paths.

### Notes
- This release focuses on reliability and consistency of convolution and postprocessing pipelines; numerical outputs for convolutions and error estimates should be validated against representative L2 inputs before bulk reprocessing.

## [0.3.0] - 2026-04-01
### Added
- Initial public release (0.3.0).

---