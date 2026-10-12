# Airbus Ship Detection

This project reproduces the main experiments from **“Airbus Ship Detection – Traditional v.s. Convolutional Neural Network Approach”** using the Airbus Ship Detection Challenge dataset from Kaggle.

The task is **binary image classification**: predict whether a satellite image contains at least one ship. This project does not perform pixel-level ship segmentation.

## Environment

The notebook is configured for **Kaggle Notebooks**.

- Dataset: [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/data)
- Notebook: `airbus_ship_detection_completed_classical_experiments.ipynb`
- GPU: enable a Kaggle GPU accelerator for CNN training
- Internet: enable it if needed to download the ImageNet-pretrained DenseNet-169 weights and install packages
- Python packages: `mahotas`, `opencv-python-headless`, `scikit-learn`, `pandas`, `matplotlib`, and `seaborn`; PyTorch is used for the CNNs and is normally available in Kaggle

## Dataset setup

1. Open the notebook in Kaggle.
2. Use **Add Data** to attach the *Airbus Ship Detection Challenge* dataset.
3. Confirm the attached dataset contains:
   - `train_ship_segmentations_v2.csv`
   - `train_v2/` (the satellite images)
4. In the notebook settings, enable a GPU and Internet if available.
5. Run the notebook cells **from top to bottom**. The setup cell searches `/kaggle/input/` for the CSV and identifies the dataset directory automatically.

The CSV contains run-length-encoded ship masks. For this classification task, an image is labelled **ship present** if at least one non-null `EncodedPixels` entry exists for its `ImageId`; otherwise it is labelled **no ship**.

## What the notebook does

### 1. Install/import dependencies and configure paths

The notebook installs the additional image-feature packages and sets the dataset, cache, output, and model directories. The current configuration uses random seed `0`.

### 2. Prepare the data

- Randomly sample **7,000 training**, **3,000 development/validation**, and **5,000 test** images (15,000 total).
- Use uniform random sampling so the class imbalance is not manually removed.
- Resize CNN images to **256 × 256 RGB**.
- Cache the resized image array to avoid repeating the expensive preprocessing step.

The image cache is written under `/kaggle/temp/airbus_paper_reproduction`. Preparing and resizing all 15,000 images can take time and requires substantial temporary disk space.

### 3. Run traditional machine-learning experiments

The notebook evaluates five classifiers across eight feature configurations (40 experiments):

- Classifiers: Linear Discriminant Analysis (LDA), K-Nearest Neighbours (KNN, `k=5`), Gaussian Naive Bayes, Random Forest (`10` trees, maximum depth `5`), and Support Vector Machine (SVM).
- Feature groups: colour histogram (`His`), Haralick texture (`Ha`), Hu moments (`Hu`), the three pairwise combinations, all features, and the raw-pixel baseline.
- The raw-pixel baseline is reduced to **32 × 32** pixels to keep memory use manageable; the paper does not specify a separate resolution for that baseline.
- The notebook also runs 10-fold cross-validation for the all-feature comparison.

Results are saved incrementally to `table1_test_accuracy.csv`; cross-validation results are saved to `cv_scores.csv`. The Random Forest model is saved as `random_forest.joblib`.

### 4. Train the CNN models

Both CNNs are trained for **30 epochs** with batch size **64**.

- **TL-CNN:** DenseNet-169 pretrained on ImageNet, with a task-specific classification head. The notebook reports 216,577 trainable parameters.
- **SIM-CNN:** a CNN trained from scratch with four convolutional and four max-pooling layers, followed by fully connected layers and a sigmoid output. The notebook reports 4,292,001 trainable parameters.

The notebook evaluates thresholds from **0.1 to 0.9** using development-set results and reports test metrics at the paper's thresholds: **0.1 for TL-CNN** and **0.6 for SIM-CNN**. It reports accuracy, precision, recall, and specificity.

CNN training outputs:
- `tl_cnn.pth`
- `sim_cnn.pth`
- `cnn_results.json`

### 5. Package the outputs

The final notebook cell gathers the trained weights, result tables, and figures into:

`/kaggle/working/airbus_models.zip`

Download the ZIP from the notebook's output/link after the final cell completes. Files under `/kaggle/temp/` are temporary, so download the packaged results if you need to keep them.

## Running the notebook again

1. Attach the dataset and enable the required accelerator/settings.
2. Run all cells in order.
3. If the data-preparation cell has already run in the current session, its cache may be reused by the notebook; if the Kaggle session or temporary storage is reset, preprocessing may need to run again.
4. Wait for the classical experiments and both CNNs to finish.
5. Check the printed metrics and generated figures, then download `airbus_models.zip`.

The notebook saves the classical accuracy table incrementally so completed classical experiments can survive an interruption. CNN training progress and output files should still be checked after a session interruption; do not assume an interrupted model run completed.

## Results from the current run

These are the metrics shown in the run logs currently available for this project. They are **observed reproduction results**, not the paper's reported results.

| Model | Evaluation threshold | Test accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|
| TL-CNN | 0.1 | 0.524 | 0.31 | 0.99 | 0.39 |
| SIM-CNN | 0.6 | 0.922 | 0.87 | 0.75 | 0.97 |

The paper reports test accuracy of approximately **0.90 for TL-CNN** and **0.94 for SIM-CNN** at those thresholds. SIM-CNN is relatively close in accuracy in the current run. TL-CNN's current test result differs substantially from the paper, so the reproduction should be treated as **in progress rather than an exact reproduction**. Differences can arise from the sampled split, preprocessing, training/checkpoint behaviour, or inference/threshold handling; these causes have not yet been confirmed.

## Important reproduction notes

- The notebook uses the specified split sizes, but a random sample with seed `0` is not necessarily the same exact set of images used by the paper.
- The notebook's raw-pixel baseline uses 32 × 32 images for memory reasons; this is a documented deviation.
- Feature extraction details and CNN implementation choices may differ from the original authors' implementation where the paper does not specify every parameter.
- Matching the reported trainable parameter counts does not, by itself, prove that the architectures and training procedures match exactly.
- Always report the actual metrics produced by your run. Do not replace them with the paper's results.

## Reference

Ying Chen, Junwen Zheng, and Zhengqing Zhou, **“Airbus Ship Detection – Traditional v.s. Convolutional Neural Network Approach.”** The paper compares traditional feature-based classifiers with a transfer-learning CNN and a CNN trained from scratch on the Airbus ship-image classification task.
