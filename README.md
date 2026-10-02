# Image classification with PyTorch

Two Jupyter notebooks that train an image classifier on **your own images**, such as Pokémon, plants, or cars, and then use it. You put images into folders; no code changes are needed.

| File | What it does |
|---|---|
| `1_training.ipynb` | trains the network on `data/training`, reports accuracy on `data/testing` after every epoch, saves `model.pt` |
| `2_prediction.ipynb` | loads `model.pt` and prints the 3 most likely classes for every image in `data/predict` |

## Quick start

1. Create one folder per class in `data/training/` and put the images in it. The folder name becomes the class name.
2. *(Optional)* Put test images into the same class folders under `data/testing/`.
3. Run `1_training.ipynb` from top to bottom.
4. Put the images you want classified into `data/predict/` and run `2_prediction.ipynb`.

## Folder layout

```
image-classification/
├── 1_training.ipynb
├── 2_prediction.ipynb
├── model.pt                  ← created by training
└── data/
    ├── training/
    │   ├── bulbasaur/        img001.jpg, img002.png, …
    │   ├── charmander/
    │   └── pikachu/
    ├── testing/              ← the same class folders (optional)
    │   ├── bulbasaur/
    │   ├── charmander/
    │   └── pikachu/
    └── predict/              ← any images, no subfolders
```

- **One folder per class.** Images lying directly in `data/training/` (not in a class folder) are ignored.
- **Class names** are the folder names, sorted alphabetically. Any name works, including spaces and non-English letters.
- **`data/testing/` must contain exactly the same class folders** as `data/training/`. Otherwise training stops with an error. Without this check, labels would be silently mixed up.
- **If `data/testing/` is missing or empty**, 20% of each class in `data/training/` is set aside for testing automatically (`TEST_FRACTION`).

## Preparing images

### Formats

`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.ppm` and `.pgm`. **Files with any other extension are silently skipped**, so check the image counts that training prints for every class.

- **iPhone photos (HEIC) are not supported.** Convert them to JPG, or set the camera to *Settings → Camera → Formats → Most Compatible*.
- **Transparent images** (e.g. PNG artwork) get a white background. This happens in both training and prediction, so the two always match.
- Grayscale images are converted to colour automatically.

### Size: they do not need to be the same

Every image is resized to `IMAGE_SIZE` × `IMAGE_SIZE` (224 × 224) automatically, so you can mix photos, screenshots and downloaded pictures.

- **Recommended:** the shorter side at least ~256 px, and ideally no more than ~1000 px.
- **Too small** (e.g. 64 × 64 sprites): images get enlarged and look blurry. It still works, but fine details are lost.
- **Too large** (e.g. 4000 × 3000 phone photos): accuracy is no better, because the network only sees 224 × 224. Loading is much slower, though, since every large file is decoded again in every epoch. Shrink such photos once beforehand, **on a copy**, because this overwrites the files:

  ```bash
  mogrify -resize '800x800>' data/training/*/*.jpg   # ImageMagick; only shrinks, never enlarges
  ```

### Shape and framing: more important than size

- **Square or nearly square works best.** For testing and prediction, a square is cut from the **centre** of each image, so the edges of a wide 16:9 photo are lost.
- **Keep the subject centred and filling most of the frame.** During training, random crops covering 60–100% of the image are used (data augmentation), so a subject near the edge is sometimes cut out.
- **Training images should look like the images you will classify later.** A model trained on official artwork with white backgrounds will do poorly on photos of plush toys. Include examples of every kind you expect.

### How many images

- **With the pretrained network** (`PRETRAINED = True`): at least **30–50 images per class**. More is better.
- **With the small network from scratch** (`PRETRAINED = False`): hundreds to thousands per class.
- **Keep classes roughly balanced.** A class with 10× more images than the others pulls predictions toward itself.
- **Test images must be different photos from the training images.** Copies, slightly edited versions or burst shots of the same scene make the test accuracy look better than it really is.

## Settings

All settings are at the top of `1_training.ipynb`.

| Setting | Default | Meaning |
|---|---|---|
| `TRAIN_DIR`, `TEST_DIR` | `data/training`, `data/testing` | where the images are |
| `MODEL_PATH` | `model.pt` | where the trained model is saved |
| `PRETRAINED` | `True` | `True`: fine-tune ResNet-18 pretrained on ImageNet. `False`: small CNN trained from scratch |
| `IMAGE_SIZE` | 224 | images are resized to this size (square) |
| `EPOCHS` | 15 | how many times the network goes through all training images |
| `BATCH_SIZE` | 64 | images processed at once |
| `LEARNING_RATE` | 0.0003 / 0.001 | set automatically from `PRETRAINED` |
| `TEST_FRACTION` | 0.2 | fraction set aside for testing when `data/testing` is missing or empty |
| `NUM_WORKERS` | 8 | worker processes that load images in parallel (see *Shared memory*) |
| `SEED` | 42 | makes the automatic test split and training repeatable |

### `PRETRAINED`: which network

- **`True` (recommended):** ResNet-18 already trained on 1.2 million ImageNet photos. Only its last layer is replaced to fit your classes, then the whole network is fine-tuned. It works well even with few images. The weights (~45 MB) are downloaded once on the first run, which needs internet access, and then cached in `~/.cache/torch`.
- **`False`:** a small 4-block convolutional network that learns everything from zero. It needs far more images, and is mainly useful for comparison.

In a quick test (3 classes, 32–40 training images each, 2 epochs), the pretrained network reached 93% test accuracy and the network trained from scratch reached 38%.

### `IMAGE_SIZE`

- **224:** keep this for the pretrained ResNet-18; it was trained at this size.
- **288 or 320:** can help when classes differ only by small details, but training is slower.
- **128:** makes the small network much faster, but costs the pretrained network accuracy.

## Reading the training output

```
Epoch  3/15 | loss 0.412 | accuracy: training  86.2%, testing  91.0% |  5.1 s
```

- **loss:** how wrong the network is on the training images. It should go down.
- **training accuracy:** measured on randomly cropped and flipped images while the network is still learning. It can be lower than testing accuracy; that is normal.
- **testing accuracy:** measured on the test images. **This is the number that matters.**
- If training accuracy keeps rising while testing accuracy stops improving or falls, the network is **overfitting**: it memorises the training images. Use more (or more varied) images or fewer epochs.
- The model saved is the one **after the last epoch**.

## The model file (`model.pt`)

It contains the learned weights, the class names, the network type (`PRETRAINED`) and `IMAGE_SIZE`. `2_prediction.ipynb` needs only this file, not the `data/training` folder, so you can copy `model.pt` to another computer together with `2_prediction.ipynb`.

Adding, removing or renaming a class means training again.

## Prediction

- For each image in `data/predict/`, the 3 most likely classes are printed with percentages.
- **The network always chooses one of its known classes**, even for a photo of something completely different. It cannot say "I don't know". A low top percentage (e.g. below 50%) means it is unsure.
- Images are prepared exactly as test images in training: transparency becomes white, the centre square is cut out and resized to `IMAGE_SIZE`.

## Hardware and speed

- The GPU is used automatically when available, with mixed precision (bfloat16) for speed. Without a GPU everything runs on the CPU, much more slowly.
- **NVIDIA RTX 50xx (Blackwell) GPUs need PyTorch 2.7 or newer built for CUDA 12.8 or later**, e.g.:
  ```bash
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
- **The main speed factor is usually image loading, not the GPU.** More `NUM_WORKERS` and smaller image files (see *Size*) help most.

### Shared memory (Docker / JupyterHub)

The worker processes hand images over through shared memory (`/dev/shm`). The amount needed is roughly:

```
NUM_WORKERS × 2 × BATCH_SIZE × 3 × IMAGE_SIZE × IMAGE_SIZE × 4 bytes
```

With the defaults that comes to about 8 × 2 × 38.5 MB ≈ 620 MB. In practice, allow about 1.5× the formula, so 2 GB is comfortable. Docker gives containers only **64 MB** by default. Check what you have from a notebook cell:

```
!df -h /dev/shm
```

To get more:

- **docker compose** (Jupyter runs in the compose service itself):
  ```yaml
  services:
    jupyter:
      shm_size: "2gb"
  ```
- **JupyterHub with DockerSpawner** (one container per user). In `jupyterhub_config.py`, merge this into any existing `extra_host_config`, such as the GPU settings:
  ```python
  c.DockerSpawner.extra_host_config = {"shm_size": "2g"}
  ```
  Existing user containers must be removed so they are recreated with the new setting.

If you cannot change it, lower `NUM_WORKERS` (or set it to 0, which is slower) or lower `BATCH_SIZE`.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `unable to allocate shared memory`, `bus error`, `DataLoader worker exited unexpectedly` | Not enough shared memory. See *Shared memory*. After this error, restart the kernel before trying again. |
| `Couldn't find any class folder in data/training` | Images must be inside class folders, not directly in `data/training/`. |
| `Found no valid file for the classes …` | A class folder has no supported images. If the class is `.ipynb_checkpoints`, Jupyter created that hidden folder: delete it. |
| `data/testing must contain the same class folders as data/training` | Add the missing class folders to `data/testing/`, or remove `data/testing/` to use the automatic split. |
| A class has fewer images than expected | Unsupported file types (e.g. `.heic`) are skipped. Convert them to JPG. |
| `FileNotFoundError: model.pt` | Run `1_training.ipynb` first, in the same folder. |
| Downloading the ResNet-18 weights fails | No internet access from the server. Allow access to `download.pytorch.org`, or use `PRETRAINED = False`. |
| `no kernel image is available` / GPU not compatible warning | PyTorch is too old for the GPU. See *Hardware and speed*. |
| `nvidia-smi` on the server: `Driver/library version mismatch` | The NVIDIA driver was updated but the old one is still loaded. Reboot (or unload and reload the `nvidia` kernel modules), then run `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` before starting GPU containers again. |
