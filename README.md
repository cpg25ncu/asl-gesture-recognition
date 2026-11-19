<h1 align="center">Real-time ASL Hand-Gesture Recognition (A–J)</h1>

<p align="center">
  <b>English</b> &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/cpg25ncu/asl-gesture-recognition/actions/workflows/ci.yml"><img src="https://github.com/cpg25ncu/asl-gesture-recognition/actions/workflows/ci.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/python-3.11--3.12-blue" alt="python 3.11-3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
  <a href="https://github.com/cpg25ncu/asl-gesture-recognition/releases/latest"><img src="https://img.shields.io/github/v/release/cpg25ncu/asl-gesture-recognition" alt="latest release"></a>
</p>

Recognises the American Sign Language letters **A–J** from a webcam in real time.
MediaPipe tracks 21 hand landmarks per frame. The landmark geometry is then
**normalised** for position, size and frame aspect ratio, and classified with
classical machine learning, including a **k-nearest-neighbours classifier
written from scratch in pure Python**.

<p align="center"><img src="reports/figures/demo.gif" width="560" alt="Real-time demo"></p>
<p align="center"><sub>Demo clip generated from the author's own photos (moving hand, plain and cluttered backgrounds), processed frame by frame by the streaming pipeline.</sub></p>

> **Background.** This started as a group project in an undergraduate AI module at
> the University of East Anglia (2025–26), where I built the whole technical
> pipeline: feature extraction, cleaning, from-scratch kNN, model tuning and
> clustering. I later rebuilt it as a reproducible package and extended it with a
> real-time pipeline, landmark normalisation and a robustness study.

## Highlights

Accuracy is reported two ways. **End-to-end** counts every held-out test image,
and an image where MediaPipe finds no hand is counted as an error. **Classification**
accuracy counts only the images where a hand was detected.

| | |
|---|---|
| **93.8 %** end-to-end test accuracy | All 1,117 held-out test images (MediaPipe finds the hand in 96.8 % of them). On the 1,081 images with a detected hand, the classifier (Logistic Regression) is **96.9 %** accurate, macro-F1 96.9 % |
| kNN written from scratch | Standard library only (`math`, `heapq`); unit-tested to match scikit-learn's predictions exactly. 92.7 % end to end, 95.7 % classification |
| Normalisation helps every model | Classification accuracy, Decision Tree 89.8 → 94.2 %, kNN 95.0 → 95.7 %, LR 96.3 → 96.9 %. End to end with the hand shifted and resized: Decision Tree 51.8 → 89.6 %, kNN 68.3 → 90.5 % |
| Robust to unseen conditions | End to end: 82 % on cluttered backgrounds. Pre-processing rescues hard cases: low light 43 → 87 % (adaptive gamma), sensor noise 4 → 84 % (denoising) |
| Real time on a laptop CPU | **129 FPS** (7.8 ms per 640×480 frame) with landmark tracking, 2.1× faster than detecting on every frame |
| Reproducible | One CLI, fixed seeds, a saved train/test split, 23 unit tests, GitHub Actions CI |

## How it works

<p align="center"><img src="reports/figures/pipeline.png" width="900" alt="Pipeline"></p>

1. **Landmarks.** MediaPipe Hands returns 21 (x, y, z) keypoints, a 63-value vector.
   In video mode, MediaPipe runs its palm detector only when tracking is lost, which
   is the main reason the pipeline runs in real time.
2. **Normalisation** ([`features.py`](src/aslgesture/features.py)). MediaPipe's x and y
   are fractions of the frame width and height, so the raw vector encodes where the
   hand is and how big it looks as much as the gesture itself. The pipeline rescales
   to isotropic units (so a 640×480 frame doesn't squash the hand), moves the wrist
   to the origin and divides by the palm length (wrist → middle-finger knuckle).
3. **Classification.** Three model families are tuned with stratified 5-fold
   cross-validation. Every grid includes the library defaults as a baseline.
4. **Temporal smoothing.** Class probabilities are averaged over the last 5 frames,
   so the displayed letter doesn't flicker. Predictions below 50 % confidence are
   shown as "?".

## Results

### Data

| | |
|---|---|
| Images | 3,781 photos of ASL A–J (256×256) from about 125 contributors, various skin tones, backgrounds and lighting |
| Hand detected by MediaPipe | 3,661 of 3,781 (96.8 %). The other 120 (119 unique) have no landmarks, so they can't be used for training. **They are still part of the test set for every end-to-end figure, counted as errors** |
| Exact duplicates removed | 59 images with identical landmarks, plus 1 duplicate among the undetected images. One duplicate group even carried conflicting labels. Left in, duplicates can leak between the training and test sets |
| Split | Stratified 70/30 by letter, fixed seed. **Train**: 2,521 images with landmarks. **Test**: 1,117 images, 1,081 with a detected hand plus 36 without |
| Extraction speed | 67 s for all 3,781 images (56 images/s, CPU only) |

<p align="center"><img src="reports/figures/class_distribution.png" width="640"></p>

### Supervised learning (held-out test set)

| Model | Feature set | Best hyper-parameters (5-fold CV) | CV accuracy | **End-to-end** accuracy (1,117 images) | Classification accuracy (1,081 detected) | Classification macro-F1 |
|---|---|---|---|---|---|---|
| kNN (from scratch) | raw | k=5, distance-weighted | 95.2 ± 1.1 % | 91.9 % | 95.0 % | 94.9 % |
| Decision Tree | raw | max_depth=None, min_samples_split=2 | 90.1 ± 1.2 % | 86.9 % | 89.8 % | 89.4 % |
| Logistic Regression | raw | C=1, L1 | 95.9 ± 1.2 % | 93.2 % | 96.3 % | 96.2 % |
| kNN (from scratch) | **norm** | k=5, distance-weighted | 95.5 ± 0.5 % | 92.7 % | 95.7 % | 95.6 % |
| Decision Tree | **norm** | max_depth=15, min_samples_split=2 | 93.3 ± 0.7 % | 91.1 % | 94.2 % | 94.0 % |
| **Logistic Regression** | **norm** | C=100, L2 | 95.7 ± 0.9 % | **93.8 %** | **96.9 %** | **96.9 %** |
| kNN (from scratch) | norm_rot | k=1 | 92.0 ± 1.0 % | 88.9 % | 91.9 % | 91.6 % |
| Decision Tree | norm_rot | max_depth=15, min_samples_split=20 | 88.4 ± 0.7 % | 85.9 % | 88.8 % | 88.6 % |
| Logistic Regression | norm_rot | C=1, L2 | 91.4 ± 1.1 % | 89.2 % | 92.1 % | 91.9 % |
| kNN (from scratch) | norm_aug | k=5, distance-weighted | 95.8 ± 0.8 % | 92.3 % | 95.4 % | 95.3 % |
| Decision Tree | norm_aug | max_depth=10, min_samples_split=5 | 93.9 ± 0.4 % | 89.5 % | 92.5 % | 92.3 % |
| Logistic Regression | norm_aug | C=1, L1 | 95.7 ± 1.1 % | 93.3 % | 96.4 % | 96.3 % |

CV accuracy is measured on the training images, all of which have landmarks.
`norm_aug` uses the same features as `norm`, but the training set gets 3 extra
copies of each hand, rotated at random by up to ±30°. Augmentation is applied
inside each CV fold, after the split, so rotated copies of a validation hand never
leak into its training fold.

<p align="center"><img src="reports/figures/feature_set_comparison.png" width="640"></p>

* **Normalisation (`norm`) improves every model.** It also lowers the CV variance for
  kNN and the Decision Tree. The tree gains the most (+4.4 points): it splits on one
  coordinate at a time, so it suffers most when the same gesture appears at different
  positions and sizes.
* **Rotation alignment (`norm_rot`) hurts (−4 to −5 points).** Some letters differ
  mainly in orientation. I and J are the clearest case (J is drawn by tilting the
  I-hand), so rotating every hand upright throws away a useful cue.
* **Rotation augmentation (`norm_aug`) is a trade-off.** It makes models more tolerant
  of tilted hands (end to end under ±30° rotation: Logistic Regression 85.0 → 88.5 %,
  Decision Tree 72.2 → 84.7 %). It also weakens the orientation cue for J: J recall falls
  from 90.5 to 87.6 % on the test set, and on the demo clip J is held for 17 of 40
  frames instead of 31. **The demo therefore ships Logistic Regression + `norm`**.
  Selection used cross-validation, where `norm` and `norm_aug` tie at 95.7 %, plus the
  I/J behaviour, not test accuracy.
* On clean, centred images, raw and normalised Logistic Regression are within CV
  noise of each other. The real benefit of normalisation shows up when the hand
  moves around the frame (see the robustness results below).
* **Where the full-system errors come from.** The deployed model gets 69 of the 1,117
  test images wrong. 36 of those are images where MediaPipe finds no hand; the other
  33 are classification mistakes. Among the mistakes, **I ↔ J** dominates (14 of 33):
  J is a *motion* sign, and a single frame of it looks very much like I. The confusion
  matrix shows the missed detections as a separate "no hand" column.

<p align="center">
<img src="reports/figures/confusion_lr_norm.png" width="420">
</p>

Hyper-parameter search, one figure per feature set ([raw](reports/figures/cv_tuning_raw.png) · [norm](reports/figures/cv_tuning_norm.png) · [norm_rot](reports/figures/cv_tuning_norm_rot.png) · [norm_aug](reports/figures/cv_tuning_norm_aug.png)):

<p align="center"><img src="reports/figures/cv_tuning_norm.png" width="900"></p>

### Robustness to unseen capture conditions

All 1,117 **test** images were re-rendered under conditions the dataset doesn't
cover, and the full pipeline (detection + features + classifier) was run end to end.
Every figure below is end-to-end accuracy over all 1,117 images, so an image counts
as wrong if MediaPipe loses the hand.

Complex backgrounds were made by segmenting the hand with **OpenCV GrabCut**, seeded
from its landmarks, and placing it on cluttered scenes. The three conditions that
cut out the hand first need its landmarks from the original photo, so they can't be
generated for the 36 images where MediaPipe never found a hand. Those 36 count as
errors there. Examples, using only the author's own photos:

<p align="center"><img src="reports/figures/robustness_examples.png" width="900"></p>

End-to-end test accuracy (%) over all 1,117 test images. The "MediaPipe detection" column is the share of images in which a hand was found at all:

| Condition | MediaPipe detection | Decision Tree raw → norm | kNN raw → norm | LR raw → norm | LR norm_aug |
|---|---|---|---|---|---|
| clean (original photos) | 96.8 % | 86.9 → 91.1 | 91.9 → 92.7 | 93.2 → **93.8** | 93.3 |
| complex background | 86.6 % | 75.7 → 79.7 | 80.7 → 81.3 | 82.4 → 82.4 | 82.3 |
| blur (σ = 2.5 px) | 95.1 % | 84.0 → 88.0 | 89.8 → 89.4 | 90.7 → 91.3 | 91.3 |
| rotation ±15–30° | 96.5 % | 77.1 → 72.2 | 85.5 → 77.2 | 87.0 → 85.0 | **88.5** |
| shift & scale (hand 55–75 % size, off-centre) | 94.5 % | **51.8 → 89.6** | **68.3 → 90.5** | 88.5 → 91.9 | 91.8 |
| webcam-like (background + rotation + shift + blur) | 81.8 % | 51.7 → 72.5 | 66.2 → 74.0 | 74.1 → 75.7 | 76.6 |
| low light | 47.3 % | 39.8 → 42.9 | 43.2 → 43.0 | 43.1 → 43.2 | 42.8 |
| low light + CLAHE | 30.8 % | 25.6 → 27.6 | 28.2 → 28.1 | 28.1 → 28.2 | 28.4 |
| low light + **adaptive gamma** | 90.5 % | 80.1 → 85.3 | 85.9 → 85.5 | 86.4 → **87.0** | 86.1 |
| sensor noise (σ = 15) | 4.7 % | 3.3 → 3.8 | 3.8 → 3.8 | 3.8 → 4.0 | 3.8 |
| sensor noise + **median denoise** | 87.7 % | 77.0 → 82.2 | 82.9 → 83.2 | 84.1 → **84.0** | 83.6 |

<p align="center"><img src="reports/figures/robustness.png" width="900"></p>

What this shows:

* **Normalisation is what makes the classifier position- and scale-independent.**
  When the hand is shifted and shrunk, the raw-coordinate Decision Tree collapses to
  51.8 % and kNN to 68.3 %. With normalised features they stay at about 90 %.
* **The bottleneck under hard conditions is detection, not classification.** With a
  cluttered background, 95 % of the hands MediaPipe *does* find are classified
  correctly. The losses come from the 13.4 % of images where no hand is detected.
  Low light and noise mostly stop MediaPipe from finding the hand at all.
* **Cheap OpenCV pre-processing fixes most of that.** MediaPipe's palm detector turned
  out to be very sensitive to per-pixel noise: at σ = 15 it finds the hand in only
  4.7 % of images. A 5×5 median filter (≈1 ms) restores detection to 87.7 %. For dark
  frames, adaptive gamma (only ever brightens, < 1 ms) restores it from 47.3 to
  90.5 %. **CLAHE made things worse** (30.8 %): it amplifies noise in dark regions.
* Both kept steps were also checked on the original photos. Gamma only changes the
  darker ones and finds 1 more of the 36 missed hands (93.9 vs 93.8 %). Denoising
  finds 2 more but misclassifies a few others, a net −0.2 points end to end. That is why both are options (`--auto-gamma`,
  `--denoise`) rather than always on.
* Rotation is the one case where raw features hold up better than `norm` for kNN and
  the tree. Rotation augmentation closes that gap; see the trade-off discussed above.

### Real-time performance

Measured on a 400-frame 640×480 clip (`asl-gesture benchmark`) on a laptop CPU (AMD Ryzen AI 9 HX 370, no GPU).
The time covers the whole per-frame pipeline except camera capture and drawing:

| Setting | Throughput | Mean latency | Pre-processing | MediaPipe | Features + classifier |
|---|---|---|---|---|---|
| Detect hand on every frame | 62 FPS | 16.1 ms | – | 15.3 ms | 0.7 ms |
| **Track landmarks across frames (default)** | **129 FPS** | **7.8 ms** | – | 7.1 ms | 0.6 ms |
| + adaptive gamma | 129 FPS | 7.7 ms | 0.4 ms | 6.8 ms | 0.5 ms |
| + median denoise | 114 FPS | 8.8 ms | 1.4 ms | 6.8 ms | 0.6 ms |
| + both | 108 FPS | 9.3 ms | 1.8 ms | 6.9 ms | 0.6 ms |

Tracking roughly halves the per-frame cost, because the palm detector only runs
when the hand is lost. Even with both pre-processing steps on, the pipeline runs at
over 3× a typical webcam's 30 FPS. For comparison, the pure-Python kNN needs
≈1.8 ms per prediction versus 0.2 ms for Logistic Regression. kNN would also fit the
real-time budget, but Logistic Regression is faster, more accurate and stores no
training data.

### Unsupervised learning: does the data cluster into letters?

K-Means and agglomerative (Ward) clustering were run on the landmarks without
labels, for k = 2…20. Labels were used only afterwards, to score the clusters.
Clustering needs landmarks, so these figures cover the 3,602 images with a detected hand.

| Features | Method (k=10) | Silhouette | Purity | ARI | NMI |
|---|---|---|---|---|---|
| raw | K-Means | 0.14 | 0.47 | 0.28 | 0.42 |
| raw | Agglomerative (Ward) | 0.12 | 0.49 | 0.37 | 0.52 |
| norm | K-Means | 0.20 | 0.46 | 0.30 | 0.47 |
| norm | Agglomerative (Ward) | 0.14 | 0.46 | 0.28 | 0.45 |

The silhouette peaks at k = 2–4, not at 10. The letters don't form ten separate
natural clusters. The cluster/letter table shows why: clustering merges letters
with almost the same hand shape (**B + F**, **G + H**, **I + J**). These are the same
pairs that labelled classifiers sometimes confuse. With labels, the classifier reaches
96.9 % on the same kind of images (detected hands). Without labels, mapping each
cluster to its majority letter gives under 50 %.

<p align="center">
<img src="reports/figures/clustering_crosstab.png" width="460">
<img src="reports/figures/clustering_silhouette.png" width="520">
</p>
<p align="center"><img src="reports/figures/pca_by_letter.png" width="820"></p>

## Quick start

```bash
git clone https://github.com/cpg25ncu/asl-gesture-recognition.git
cd asl-gesture-recognition
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
```

Python 3.11 or 3.12 is required (MediaPipe 0.10.14; versions pinned in `requirements.txt`). A trained model (8 KB) is included in [`models/`](models).

```bash
asl-gesture demo                       # webcam, press q to quit
asl-gesture demo --left-hand           # mirror frames for a left hand
asl-gesture demo --auto-gamma --denoise   # dim or noisy cameras
asl-gesture demo --source clip.mp4 --save out.mp4 --no-display
asl-gesture predict examples/own_images/B_1.jpg
```

Reproduce everything from your own images (see [`data/README.md`](data/README.md)):

```bash
asl-gesture extract        # MediaPipe -> data/processed/landmarks.csv
asl-gesture train          # CV tuning + test evaluation, 3 models x 4 feature sets
asl-gesture export-model --model lr --feature-set norm
asl-gesture cluster        # K-Means / agglomerative analysis
asl-gesture robustness     # synthetic capture conditions
asl-gesture make-demo-video && asl-gesture benchmark --source data/demo_input.mp4
pytest
```

## Project structure

```
src/aslgesture/
├── landmarks.py    MediaPipe wrapper (image + tracking modes), dataset extraction
├── features.py     raw / normalised / rotation-aligned features, rotation augmentation
├── knn.py          kNN from scratch (standard library only)
├── supervised.py   cleaning, split, 5-fold CV grid search, evaluation, end-to-end metrics
├── clustering.py   K-Means, agglomerative, purity / ARI / NMI, PCA
├── robustness.py   GrabCut segmentation, cluttered backgrounds, perturbations,
│                   pre-processing (median denoise, adaptive gamma, CLAHE)
├── realtime.py     streaming pipeline, smoothing, overlay, benchmark, demo clip
└── cli.py          `asl-gesture` command line
tests/              kNN vs scikit-learn, normalisation invariances, pre-processing
reports/            all figures and result tables used in this README
models/             trained Logistic Regression model used by the demo
examples/           30 photos of the author's own hand (3 per letter)
```

## Limitations

* **MediaPipe misses about 3 % of hands even in clean photos.** Those images are
  errors for the whole system, whatever classifier follows. This is why end-to-end
  accuracy (93.8 %) is about 3 points below classification accuracy (96.9 %).
* **Static letters only.** J (and Z) are motion signs in ASL. Here J is recognised
  from a single frame, which is why most classification errors are I ↔ J. A
  sequence model over landmark trajectories would be the natural next step.
* The robustness study uses **synthetic** perturbations of real test photos, and the
  FPS benchmark uses a clip animated from still photos. Both isolate one factor at a
  time, but neither replaces a test set recorded live on a webcam.
* Low light remains the weakest condition without pre-processing (43 % end to end).
  With adaptive gamma it reaches 87 %.
* The training images are mostly right hands. Left hands are supported by mirroring
  (`--left-hand`).
* The training dataset can't be redistributed (see [`data/README.md`](data/README.md)).

## Acknowledgements

* [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) by Google for landmark detection.
* The classmates who contributed photos to the course dataset, and my project
  teammate for discussions and presentation work.
* The cluttered-background tests use two photos bundled with scikit-learn
  (`china.jpg`, `flower.jpg`), CC BY 2.0, by Flickr users *danielbuechele* and *vultilion*.
