# CPE 4903 — Real-Time Handwriting Recognition (TensorFlow + Raspberry Pi 5)

End-to-end MNIST digit recognition on a **Raspberry Pi 5**: train a CNN, convert
to TensorFlow Lite, capture digits with the Pi Camera, run on-device inference,
and display the prediction on the **Sense HAT** LED matrix.

> Course: CPE 4903 · Author: Etienne Enzo Sidibe · Kennesaw State University · Dec 2025

## System
1. Train CNN on MNIST (TensorFlow, trained on the Pi 5).
2. Save `.keras` model → convert to TFLite.
3. Capture frame via 1080p Pi Camera.
4. Preprocess: grayscale → resize 28×28 → normalize.
5. Run TFLite inference.
6. Display predicted digit on the Sense HAT.

## Model
Stacked Conv layers with Batch Normalization + Dropout, GlobalAveragePooling
head (~300k params, tuned for the Pi). `tf.data` pipeline (normalize / batch /
cache / prefetch) with `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`.
**Test accuracy ≈ 0.9895** on MNIST.

## Platform
Raspberry Pi 5 · Raspberry Pi OS (Debian 64-bit) · 1080p Camera · Sense HAT ·
Python 3.9 venvs · TensorFlow (CPU), TFLite Runtime, OpenCV.

```bash
python3 -m venv tf-env        # training env: TensorFlow, NumPy, OpenCV
python3 -m venv tflite-env    # separate inference env: tflite-runtime, OpenCV
```
