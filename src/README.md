# Source Code

This directory contains the Python scripts used for evaluating segmentation results, analyzing annotations, and processing DICOM images.

## Directory Structure

### Top-level scripts

| Script | Description |
|--------|-------------|
| `segmentation_evaluation.py` | Main pipeline for evaluating kidney segmentation results (DSC, Hausdorff distance, overlays, compound images) |
| `clean_segmentation.py` | Post-processing for segmentation masks (restricts regions to capsule, keeps largest contour) |
| `compare_mutual_information.py` | Computes mutual information between images and Nakagami parameters |

### `tools/` -- Utility modules

Shared utility functions used by the evaluation and analysis scripts.

| Module | Description |
|--------|-------------|
| `basic_utils.py` | File/directory operations and CSV formatting helpers |
| `metric_utils.py` | Segmentation metrics: Dice coefficient, Hausdorff distance, precision, recall |
| `image_utils.py` | Image visualization: compound images and color-coded mask overlays |
| `size_tools.py` | Kidney size measurements (length, diameter, area, volume) from masks |
| `distribution_utils.py` | Nakagami distribution fitting, SNR computation, KL divergence |
| `variability_shared.py` | Shared helpers for inter/intra-rater variability analysis |

### `annotation_analysis/` -- Annotation evaluation scripts

Scripts for analyzing the manual annotations provided by expert sonographers.

| Script | Description |
|--------|-------------|
| `create_masks.py` | Generates binary masks from VGG Image Annotator polygon annotations |
| `cohens_kappa.py` | Cohen's Kappa inter-rater reliability for quality scores |
| `interrater_dsc.py` / `interrater_hd.py` | Inter-rater Dice and Hausdorff distance |
| `intrarater_dsc.py` / `intrarater_hd.py` | Intra-rater Dice and Hausdorff distance |
| `compare_distributions.py` | Nakagami parameter comparison between ground truth and predictions |
| `compare_snr.py` | SNR comparison between ground truth and predictions |
| `nakagami_kl_divergence.py` | KL divergence of Nakagami distributions |
| `sensitivity_analysis.py` | Sensitivity of metrics to morphological erosion/dilation |
| `summative_stats.py` | Summary statistics and histograms for annotation coverage |
| `point_stats.py` | Statistics on annotation polygon point counts per class |
| `view_counter.py` | Counts ultrasound view types (transverse, longitudinal, other) |
| `overlay_masks.py` | Creates color-coded mask overlays on original images |
| `create_compound_image.py` | Side-by-side composite images for visual comparison |
| `create_compound_image_from_csv.py` | Compound images from repeated annotations |
| `convert_nifty_to_png.py` / `convert_png_to_nifty.py` | Format conversion between NIfTI and PNG |

### `echotools/` -- DICOM processing (reference only)

These scripts were used to process the original DICOM files. Since only anonymized PNG images are shared with the dataset, these are provided as reference for reproducibility.

| Module | Description |
|--------|-------------|
| `dicom.py` | Core DICOM handling: reading, cleaning, PHI removal, format conversion |
| `clean_dcm_images.py` | Pipeline for batch DICOM processing and metadata extraction |

## Dependencies

See `requirements.txt` in the repository root for required Python packages.
