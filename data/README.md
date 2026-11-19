# Data

The training images are **not included** in this repository. The original
dataset (3,781 photos of ASL letters A-J from about 125 contributors) was
compiled for a university course and is marked *do not distribute*, so
neither the images nor anything derived from them per sample (landmark CSVs,
train/test split, kNN models that store the training set) are published here.

To run the pipeline on your own data, arrange images by letter:

```
data/raw/A/*.jpg
data/raw/B/*.jpg
...
data/raw/J/*.jpg
```

or point the code at another folder with the `ASL_DATA_DIR` environment
variable (it must contain a `raw/` subfolder). Then:

```bash
asl-gesture all          # extract -> train -> cluster -> robustness
```

Everything under `data/` except this file is git-ignored. The folder
[`examples/own_images/`](../examples/own_images) contains 30 photos of the
author's own hand (3 per letter) that can be shared and are used for the
demo and the CI smoke test.
