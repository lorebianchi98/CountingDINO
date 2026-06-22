# CARPK Evaluation Protocol & Reproducibility

This folder contains the official exemplar annotations and configuration details used to evaluate **CountingDINO** on the CARPK dataset, achieving the reported performance of **MAE: 21.26** and **RMSE: 28.20**.

To ensure full reproducibility, we have provided the exact bounding boxes used as exemplars, along with the precise pipeline configurations.

---

## 1. Exemplar Selection Strategy

The differences in reproducibility often stem from how exemplars are sampled. For our CARPK evaluation, we adopted the following protocol:

* **Quantity:** 3 exemplar bounding boxes per image.
* **Sampling:** Randomized using a fixed **seed = 11**.
* **Boundary Constraint:** To avoid truncated or cut-off objects, we only selected bounding boxes that are located **at least 5 pixels away from the image borders**.

The exact annotations selected through this process are saved in this directory.

---

## 2. Model Configuration

The reported test results were achieved using the `DINOv2 L/14 reg.` backbone (`dinov2_vitl14_reg`). Below are the specific hyperparameters and processing flags required for CARPK:

| Parameter | Value | Description |
| --- | --- | --- |
| `divide_et_impera` | **True** | Enables the patch-based divide-and-conquer processing. |
| `divide_et_impera_twice` | **True** | Applies the divide-and-conquer strategy hierarchically/twice for finer resolution. |
| `filter_background` | **True** | Filters out background noise from the density map. |
| `ellipse_normalization` | **True** | Applies elliptical normalization to the target shapes. |
| `ellipse_kernel_cleaning` | **True** | Post-processing morphological cleaning via ellipse kernels. |

### Reported Test Metrics

* **MAE:** 21.26
* **RMSE:** 28.20

---

## 3. Inference Specifics & Pipeline Integration

When running inference on CARPK, ensure that your evaluation script integrates the specific flags listed above.

No images or ground truth annotations were excluded from the official test set.
