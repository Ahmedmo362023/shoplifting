# Shoplifting Detection: Video Classification (PyTorch)

Binary classification of short surveillance clips (shoplifting vs normal behaviour) with two approaches, compared on the same split:

1. **Part 1 - From scratch (no pretrained weights)**
   - **CNN + GRU:** a per-frame 2D CNN encoder followed by a bidirectional GRU over time.
   - **3D CNN:** spatio-temporal convolutions over the whole clip.
2. **Part 2 - Pretrained model (transfer learning)**
   - **VideoMAE-B** from Hugging Face (`MCG-NJU/videomae-base-finetuned-kinetics`), with the classifier head replaced by a 2-class head and fine-tuned on this dataset.

Everything lives in a single notebook: `shoplifting_week5_week6.ipynb` (Part 1 cells first, Part 2 cells after).

## Highlights
- **Uniform frame sampling.** Each video is cut into T equal segments and one frame is taken from each. Training picks a random position inside each segment (jitter), while validation and test use the segment midpoint. It works for any video length, and frames are repeated if a video has fewer than T frames.
- **Clip-consistent augmentation.** Crop box, horizontal flip, brightness, contrast and saturation are drawn once per clip and applied identically to every frame, so motion is not corrupted. A built-in check verifies this.
- **Same pipeline and same split for both parts**, so the comparison is fair. Part 2 reuses the Part 1 sampler, augmentation, dataset class, split and `evaluate()`.
- **Class-weighted loss** to handle the imbalance between the two classes.
- **Best-checkpoint selection** by validation F1 on the positive class (shoplifting).
- **Sanity checks** for the sampler, the dataset shapes and the augmentation consistency, plus a saved visual grid (`aug_check.png`).

## Dataset

| | Clips | Duration (mean) | Frames (mean) | Resolution |
|---|---|---|---|---|
| Non shoplifters | 531 | 13.9 s | 348 | 704 x 576 |
| Shoplifters | 324 | 12.2 s | 305 | 704 x 576 |
| **Total** | **855** | | | |

The notebook downloads the dataset zip from Google Drive with `gdown` and extracts it to `shop_data/`. Expected layout:

```
shop_data/Shop DataSet/
├── non shop lifters/   # class 0
└── shop lifters/       # class 1 (positive class)
```

Labels are the index of the alphabetically sorted class folders. Subfolders are searched recursively for common video extensions (`.mp4`, `.avi`, `.mov`, `.mkv`, ...).

**Split** (stratified, seed 42): 80% train+val / 20% test, then 15% of the train portion as validation. This gives **581 train / 103 val / 171 test** clips. Both parts use exactly this split.

## Pipeline

| Step | Part 1 (scratch) | Part 2 (VideoMAE) |
|---|---|---|
| Frame sampling | 16 frames per clip, uniformly spread | same |
| Loading size | frames resized to 128 x 128 when decoded | 256 x 256 |
| Train augmentation | RandomResizedCrop to 112 x 112 (scale 0.7-1.0, ratio 0.9-1.1), horizontal flip (p = 0.5), brightness / contrast / saturation jitter +/-0.3 | same, but crop to **224 x 224** |
| Eval preprocessing | Resize to 112 x 112 | Resize to 224 x 224 |
| Normalisation | mean 0.45, std 0.225 per channel | **ImageNet** mean/std (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225), as the pretrained model expects |

In Part 2 the Part 1 dataset class is reused through a thin subclass (`VideoDatasetVMAE`) that only converts the Part 1 normalisation to ImageNet normalisation.

## Models

### Part 1: from scratch

**CNN + GRU** (about 1.47M parameters)
- Frame encoder: 4 blocks of (Conv3x3 -> BN -> ReLU) x 2 -> MaxPool, with channels 32 -> 64 -> 128 -> 256, then global average pooling, giving a 256-d vector per frame.
- Temporal model: bidirectional GRU (hidden size 128), averaged over time.
- Head: Dropout (0.4) -> Linear.

**3D CNN**
- 4 blocks of Conv3D(3x3x3) -> BN -> ReLU -> MaxPool3D, with channels 32 -> 64 -> 128 -> 256. The first block pools spatially only, so the temporal resolution is kept early on.
- Global average pooling -> Dropout (0.4) -> Linear.

### Part 2: pretrained VideoMAE-B

**Checkpoint:** `MCG-NJU/videomae-base-finetuned-kinetics` (ViT-Base video transformer, about 86M parameters).
- Self-supervised pretraining with masked autoencoding (about 90% of the space-time patches are masked and reconstructed), then supervised fine-tuning on Kinetics-400 (400 human-action classes).
- Input: **16 frames x 224 x 224**. The clip is cut into 2 x 16 x 16 "tubelets", giving 8 x 14 x 14 = 1568 tokens, processed by 12 transformer layers with full space-time self-attention.

**Head adaptation (2-class output).** The original 768 -> 400 classifier is dropped and a freshly initialised 768 -> 2 layer is created:

```python
vmae = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2, id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True)          # backbone loaded, classifier re-created
assert vmae.classifier.out_features == 2
```

The notebook asserts the head size and checks that a dummy forward pass returns logits of shape `(1, 2)`. A small wrapper (`VMAELogits`) lets the Part 1 `evaluate()` function work with the Hugging Face model unchanged.

## Training setup

| Setting | Part 1 (scratch) | Part 2 (VideoMAE fine-tuning) |
|---|---|---|
| Optimiser | AdamW (lr 1e-3, weight decay 1e-2) | AdamW (lr 3e-5, weight decay 0.05) |
| Scheduler | Cosine annealing over all epochs | OneCycleLR (10% warm-up, then decay) |
| Loss | Cross-entropy, inverse-frequency class weights | same |
| Epochs / batch size | 20 / 8 | 8 / 4 |
| Precision | fp32 | mixed precision (fp16 autocast + GradScaler) |
| Gradient clipping | 5.0 | 1.0 |
| Trainable parameters | all | all (small learning rate, full fine-tuning) |
| Checkpoint selection | best validation F1 | best validation F1 (`best_videomae.pt`) |
| Seed | 42 | 42 |

## Results

All models are evaluated on the same held-out test set (171 clips); the positive class is *shoplifting*.

### Part 1: from scratch

**3D CNN**

| Metric | Value |
|---|---|
| Accuracy | 0.696 |
| Precision (shoplifting) | 0.563 |
| Recall (shoplifting) | 0.892 |
| F1 (shoplifting) | 0.691 |

Confusion matrix (rows = true class, columns = predicted):

| | Pred. normal | Pred. shoplifting |
|---|---|---|
| True normal | 61 | 45 |
| True shoplifting | 7 | 58 |

The model catches most shoplifting clips (high recall) but produces many false alarms (low precision).

**CNN + GRU:** validation F1 reached 0.765 by epoch 18 (val accuracy 0.816). The test report for this run is not included in the saved notebook output, so re-run the training cell to get it.

### Part 2: comparison table (scratch vs pretrained)

The notebook generates this table automatically (section 15) and saves it as `model_comparison.csv`. Replace the `TBD` values with your run's output:

| Model | Input | Accuracy | Precision | Recall | **F1** | Params (M) | Size (MB, fp32) | Latency (ms/clip) |
|---|---|---|---|---|---|---|---|---|
| CNN + GRU (scratch) | 16x112x112 | TBD | TBD | TBD | TBD | 1.47 | TBD | TBD |
| 3D CNN (scratch) | 16x112x112 | 0.696 | 0.563 | 0.892 | 0.691 | TBD | TBD | TBD |
| **VideoMAE-B (pretrained, fine-tuned)** | 16x224x224 | TBD | TBD | TBD | **TBD** | ~86 | ~345 | TBD |

Latency is measured on a single clip with batch size 1 on the GPU used for the run, so absolute numbers depend on the hardware; only the relative differences matter.

## Trade-offs: scratch vs pretrained

| Aspect | From scratch | Pretrained VideoMAE-B |
|---|---|---|
| **Accuracy / F1** | Limited by the small dataset (about 600 training clips); features must be learned from nothing, and training is unstable | Starts from Kinetics-400 features (people, hands, objects, motion), so it converges quickly and generalises better with little data |
| **Model size** | About 1-1.5M parameters (a few MB) | About 86M parameters (about 345 MB in fp32), roughly 60x larger |
| **Inference speed** | Fast: 112 x 112 input, shallow network | Slower: self-attention over 1568 space-time tokens at 224 x 224, with higher latency and GPU memory use |
| **Training cost** | Cheap per epoch, but many epochs and sensitive to learning rate and initialisation | Few epochs (about 8), but each epoch is heavy |
| **Data needed** | A large labelled set | A small set is enough thanks to transfer learning |

**Takeaway:** if missing a theft is costly, the fine-tuned VideoMAE is the better detector. For real-time monitoring of many camera streams or on-device inference, the small scratch models are far cheaper. A practical compromise is a cascade: a light model screens every clip and only suspicious ones go to VideoMAE. Other options are fp16 / INT8 quantisation, distilling VideoMAE into a small student, or a lighter pretrained backbone such as X3D-S.

## Notes and limitations
- Part 1 training is unstable in the early epochs. Validation predictions flip between "all normal" and "all shoplifting" until the loss starts to drop (around epoch 12 for CNN + GRU), so more epochs would likely help.
- The validation set is small (103 clips), so epoch-to-epoch metrics are noisy and checkpoint selection is not very reliable.
- Splits are random at the clip level. If several clips come from the same footage or scene, test scores can be optimistic for both parts.
- VideoMAE needs a GPU with enough memory. If you get a CUDA out-of-memory error, lower `VM_BATCH` (default 4) to 2.
- If VideoMAE does not beat the scratch F1, try 12-15 epochs, a learning rate between 1e-5 and 5e-5, and average results over several seeds.

## How to run
Open `shoplifting_week5_week6.ipynb` in Kaggle or Colab with a GPU runtime. The notebook was run on Kaggle (`/kaggle/working`).

Install the dependencies if they are missing:

```bash
pip install torch torchvision opencv-python numpy scikit-learn matplotlib gdown transformers accelerate pandas
```

Run the cells in order:

**Part 1**
1. Section 0 downloads and unzips the data (about 1.8 GB). Adjust `DATA_ROOT` if your folder layout differs.
2. Sections 1-7 define the imports, listing/exploration, frame sampling, augmentation, dataset, models and train/evaluate helpers.
3. Section 8-9: `check(args)` runs the sanity checks.
4. Train a model by setting the architecture and calling `train`:
   ```python
   args.arch = 'cnn_gru'   # or 'cnn3d'
   train(args)
   ```
   The best checkpoint is saved as `best_<arch>.pt`, and the final test report is printed at the end.

**Part 2** (run Part 1 first, at least up to the `args` cell)
1. Section 10 recreates the exact Part 1 split.
2. Section 11 builds the 224 x 224 data loaders.
3. Section 12 loads VideoMAE and adapts the 2-class head (with assertions).
4. Section 13 fine-tunes the model and saves `best_videomae.pt`.
5. Section 14 loads the Part 1 checkpoints (or retrains them if missing).
6. Section 15 evaluates everything on the test set and produces the comparison table, confusion matrices and F1 check.
7. Section 16 contains the written discussion of the trade-offs.

## Configuration
Edit the `args` namespace in the "Run" section (Part 1), and `VM_BATCH`, `VM_SIZE` and the `finetune(...)` arguments (Part 2):

```python
args = SimpleNamespace(data_root=DATA_ROOT, arch='cnn_gru', frames=16, size=112,
                       batch=8, epochs=20, lr=1e-3, workers=2, seed=42)

VM_BATCH = 4                                   # lower to 2 on out-of-memory
hist_vm, train_time_vm = finetune(vm, epochs=8, lr=3e-5)
```

## Outputs

| File | Description |
|---|---|
| `explore.png` | Clips per class and duration histograms |
| `aug_check.png` | Original frames (top row) vs augmented frames (bottom row) |
| `best_cnn_gru.pt`, `best_cnn3d.pt` | Best Part 1 checkpoints by validation F1 |
| `best_videomae.pt` | Best fine-tuned VideoMAE checkpoint by validation F1 |
| `model_comparison.csv` | Test metrics, size and latency of all three models |
