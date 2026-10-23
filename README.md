# Shoplifting Detection: Video Classification from Scratch (PyTorch)

Binary classification of short surveillance clips (**shoplifting** vs **normal behaviour**) using two models trained from scratch, with no pretrained weights:

- **CNN + GRU**: a per-frame 2D CNN encoder followed by a bidirectional GRU over time.
- **3D CNN**: spatio-temporal convolutions over the whole clip.

Everything lives in a single notebook, `shoplifting.ipynb`.

## Highlights

- **Uniform frame sampling.** Each video is cut into `T` equal segments and one frame is taken from each. Training picks a random position inside each segment (jitter), while validation and test use the segment midpoint. It works for any video length, and frames are repeated if a video has fewer than `T` frames.
- **Clip-consistent augmentation.** Crop box, horizontal flip, brightness, contrast and saturation are drawn **once per clip** and applied identically to every frame, so motion is not corrupted. A built-in check verifies this.
- **Class-weighted loss** to handle the imbalance between the two classes.
- **Best-checkpoint selection** by validation F1 on the positive class (shoplifting).
- **Sanity checks** for the sampler, the dataset shapes and the augmentation consistency, plus a saved visual grid (`aug_check.png`).

## Dataset

| | Clips | Duration (mean) | Frames (mean) | Resolution |
|---|---|---|---|---|
| Non shoplifters | 531 | 13.9 s | 348 | 704 × 576 |
| Shoplifters | 324 | 12.2 s | 305 | 704 × 576 |
| **Total** | **855** | | | |

The notebook downloads the dataset zip from Google Drive with `gdown` and extracts it to `shop_data/`. Expected layout:

```
shop_data/Shop DataSet/
├── non shop lifters/   # class 0
└── shop lifters/       # class 1 (positive class)
```

Labels are the index of the alphabetically sorted class folders. Subfolders are searched recursively for common video extensions (`.mp4`, `.avi`, `.mov`, `.mkv`, ...).

**Split** (stratified, seed 42): 80% train+val / 20% test, then 15% of the train portion as validation. This gives **581 train / 103 val / 171 test** clips.

## Pipeline

| Step | Details |
|---|---|
| Frame sampling | 16 frames per clip, uniformly spread |
| Loading | Frames resized to 128 × 128 when decoded |
| Train augmentation | RandomResizedCrop to 112 × 112 (scale 0.7–1.0, ratio 0.9–1.1), horizontal flip (p = 0.5), brightness / contrast / saturation jitter ±0.3 |
| Eval preprocessing | Resize to 112 × 112 |
| Normalisation | mean 0.45, std 0.225 per channel |

## Models

**CNN + GRU** (about 1.47M parameters)
- Frame encoder: 4 blocks of (Conv3×3 → BN → ReLU) × 2 → MaxPool, with channels 32 → 64 → 128 → 256, then global average pooling, giving a 256-d vector per frame.
- Temporal model: bidirectional GRU (hidden size 128), averaged over time.
- Head: Dropout (0.4) → Linear.

**3D CNN**
- 4 blocks of Conv3D(3×3×3) → BN → ReLU → MaxPool3D, with channels 32 → 64 → 128 → 256. The first block pools spatially only, so the temporal resolution is kept early on.
- Global average pooling → Dropout (0.4) → Linear.

## Training setup

| Setting | Value |
|---|---|
| Optimiser | AdamW (lr 1e-3, weight decay 1e-2) |
| Scheduler | Cosine annealing over all epochs |
| Loss | Cross-entropy with inverse-frequency class weights |
| Epochs / batch size | 20 / 8 |
| Gradient clipping | 5.0 |
| Seed | 42 |

## Results

**3D CNN** on the held-out test set (171 clips):

| Metric | Value |
|---|---|
| Accuracy | 0.696 |
| Precision (shoplifting) | 0.563 |
| Recall (shoplifting) | 0.892 |
| F1 (shoplifting) | 0.691 |

Confusion matrix (rows = true class, columns = predicted):

|  | Pred. normal | Pred. shoplifting |
|---|---|---|
| **True normal** | 61 | 45 |
| **True shoplifting** | 7 | 58 |

The model catches most shoplifting clips (high recall) but produces many false alarms (low precision).

**CNN + GRU**: validation F1 reached 0.765 by epoch 18 (val accuracy 0.816). The test report for this run is not included in the saved notebook output, so re-run the training cell to get it.

### Notes and limitations

- Training is unstable in the early epochs. Validation predictions flip between "all normal" and "all shoplifting" until the loss starts to drop (around epoch 12 for CNN + GRU), so more epochs would likely help.
- The validation set is small (103 clips), so epoch-to-epoch metrics are noisy and checkpoint selection is not very reliable.
- Splits are random at the clip level. If several clips come from the same footage or scene, test scores can be optimistic.
- Models are trained from scratch on about 600 clips. Pretrained backbones would likely improve results considerably.

## How to run

1. Open `shoplifting.ipynb` in Kaggle or Colab with a **GPU** runtime. The notebook was run on Kaggle (`/kaggle/working`).
2. Install the dependencies if they are missing:
   ```bash
   pip install torch torchvision opencv-python numpy scikit-learn matplotlib gdown
   ```
3. Run the cells in order:
   - **0** downloads and unzips the data (about 1.8 GB). Adjust `DATA_ROOT` if your folder layout differs.
   - **2–3** list and explore the videos and save `explore.png`.
   - **4–6** frame sampling, augmentation and dataset.
   - **7** models.
   - **8–9** training and evaluation helpers.
   - **10** `check(args)` runs the sanity checks.
4. Train a model by setting the architecture and calling `train`:
   ```python
   args.arch = 'cnn_gru'   # or 'cnn3d'
   train(args)
   ```
   The best checkpoint is saved as `best_<arch>.pt`, and the final test report is printed at the end.

## Configuration

Edit the `args` namespace in the "Run" section:

```python
args = SimpleNamespace(data_root=DATA_ROOT, arch='cnn_gru', frames=16, size=112,
                       batch=8, epochs=20, lr=1e-3, workers=2, seed=42)
```

## Outputs

| File | Description |
|---|---|
| `explore.png` | Clips per class and duration histograms |
| `aug_check.png` | Original frames (top row) vs augmented frames (bottom row) |
| `best_cnn_gru.pt`, `best_cnn3d.pt` | Best checkpoints by validation F1 |
