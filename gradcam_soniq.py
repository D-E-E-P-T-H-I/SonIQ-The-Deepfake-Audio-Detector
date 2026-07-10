"""
Grad-CAM for SonIQ Audio Deepfake Detector
==========================================
Drop this file into your SonIQ project directory alongside soniq.ipynb.
Run after training: python gradcam_soniq.py

Produces:
  - gradcam_outputs/  folder with one PNG per audio file
  - gradcam_summary.png  showing best real + fake examples side by side
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from pathlib import Path


# ─────────────────────────────────────────────
# 1.  PREPROCESSING  (copied from soniq.ipynb)
# ─────────────────────────────────────────────

def preprocess_audio(audio_path, target_sr=16000, duration=5.0):
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
        audio, _ = librosa.effects.trim(audio, top_db=100)
        target_length = int(target_sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        return audio, target_sr
    except Exception as e:
        print(f"[ERROR] Could not load {audio_path}: {e}")
        return None, None


def extract_mel_spectrogram(audio, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=64,
        hop_length=1024
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def audio_to_input(audio_path):
    """Full preprocessing pipeline → (1, 64, 79, 1) tensor ready for CNN."""
    audio, sr = preprocess_audio(audio_path)
    if audio is None:
        return None, None
    spec = extract_mel_spectrogram(audio, sr)
    tensor = np.expand_dims(spec, axis=(0, -1))          # (1, 64, T, 1)
    return tensor, spec                                   # spec kept for plotting


# ─────────────────────────────────────────────
# 2.  GRAD-CAM CORE
# ─────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap for one input spectrogram.

    Parameters
    ----------
    img_array        : np.ndarray  shape (1, H, W, 1)
    model            : trained Keras CNN
    last_conv_layer_name : str     name of the last Conv2D layer
    pred_index       : int or None  class index (0=real, 1=fake).
                       If None, uses the argmax of the model output.

    Returns
    -------
    heatmap : np.ndarray  shape (H_conv, W_conv)  values in [0, 1]
    pred_prob : float     model's fake probability
    """

    # Sub-model that outputs (conv_feature_maps, final_prediction)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        pred_prob = float(predictions[0][0])               # sigmoid output

        if pred_index is None:
            pred_index = 1 if pred_prob > 0.5 else 0

        # We want gradients w.r.t the fake class score
        class_channel = predictions[:, 0]

    # Gradients of the class score w.r.t. last conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)     # (1, H, W, C)

    # Mean gradient across spatial dims → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

    # Weight conv outputs by channel importances
    conv_outputs = conv_outputs[0]                          # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                           # (H, W)

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), pred_prob


def overlay_gradcam(spec_db, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the Mel-spectrogram.

    Returns
    -------
    overlay : np.ndarray  shape (H, W, 3)  RGB image
    """
    # Normalize spectrogram to [0,1] for display
    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
    spec_rgb = np.stack([spec_norm, spec_norm, spec_norm], axis=-1)  # (H, W, 3)

    # Resize heatmap to match spectrogram spatial dims using bilinear
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        [spec_db.shape[0], spec_db.shape[1]],
        method='bilinear'
    ).numpy()[..., 0]

    # Apply jet colormap to heatmap
    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)[..., :3]   # (H, W, 3) drop alpha

    # Blend
    overlay = (1 - alpha) * spec_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    return overlay


# ─────────────────────────────────────────────
# 3.  VISUALIZATION
# ─────────────────────────────────────────────

def plot_gradcam_single(
    audio_path, spec_db, heatmap, pred_prob, true_label,
    save_path=None, sr=16000, hop_length=1024
):
    """
    Three-panel figure:
      Left  — raw Mel-spectrogram
      Middle — Grad-CAM heatmap (standalone)
      Right  — overlay
    """
    overlay = overlay_gradcam(spec_db, heatmap)
    pred_label = "FAKE" if pred_prob > 0.5 else "REAL"
    true_str   = "fake" if true_label == 1 else "real"
    correct    = pred_label.lower() == true_str

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{Path(audio_path).name}  |  True: {true_str.upper()}  "
        f"|  Pred: {pred_label}  ({pred_prob*100:.1f}% fake)  "
        f"|  {'✓ Correct' if correct else '✗ Wrong'}",
        fontsize=11, fontweight='bold',
        color='green' if correct else 'red'
    )

    # Panel 1 — raw spectrogram
    axes[0].set_title("Mel-Spectrogram", fontsize=10)
    img0 = librosa.display.specshow(
        spec_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', ax=axes[0], cmap='magma'
    )
    fig.colorbar(img0, ax=axes[0], format='%+2.0f dB')

    # Panel 2 — raw heatmap (resized to spec dims)
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        [spec_db.shape[0], spec_db.shape[1]]
    ).numpy()[..., 0]
    axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
    im = axes[1].imshow(
        heatmap_resized, cmap='jet', aspect='auto',
        origin='lower', vmin=0, vmax=1
    )
    fig.colorbar(im, ax=axes[1])
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Mel bands")

    # Panel 3 — overlay
    axes[2].set_title("Spectrogram + Grad-CAM Overlay", fontsize=10)
    axes[2].imshow(overlay, aspect='auto', origin='lower')
    axes[2].set_xlabel("Time frames")
    axes[2].set_ylabel("Mel bands")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close()


def plot_gradcam_summary(results, save_path="gradcam_summary.png"):
    """
    2×2 grid showing:
      Top row    — best correctly-classified REAL examples
      Bottom row — best correctly-classified FAKE examples
    """
    real_correct = [r for r in results if r['true'] == 0 and r['correct']]
    fake_correct = [r for r in results if r['true'] == 1 and r['correct']]

    # Sort by confidence
    real_correct.sort(key=lambda x: x['prob'],  reverse=False)  # lowest fake prob = most confident real
    fake_correct.sort(key=lambda x: x['prob'],  reverse=True)   # highest fake prob = most confident fake

    picks = real_correct[:2] + fake_correct[:2]
    if not picks:
        print("[WARN] Not enough correctly classified samples for summary.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Grad-CAM Summary: Most Confident Correct Detections", fontsize=13, fontweight='bold')

    labels = ["REAL (confident)", "REAL (confident)", "FAKE (confident)", "FAKE (confident)"]

    for col, (result, label) in enumerate(zip(picks, labels)):
        spec_db  = result['spec']
        heatmap  = result['heatmap']
        overlay  = overlay_gradcam(spec_db, heatmap)
        prob     = result['prob']

        # Top row — spectrogram
        librosa.display.specshow(
            spec_db, sr=16000, hop_length=1024,
            x_axis='time', y_axis='mel',
            ax=axes[0][col], cmap='magma'
        )
        axes[0][col].set_title(
            f"{label}\n{Path(result['path']).name[:30]}\n{prob*100:.1f}% fake",
            fontsize=9
        )

        # Bottom row — overlay
        axes[1][col].imshow(overlay, aspect='auto', origin='lower')
        axes[1][col].set_title("Grad-CAM Overlay", fontsize=9)
        axes[1][col].set_xlabel("Time frames")
        axes[1][col].set_ylabel("Mel bands")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSummary saved → {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 4.  FREQUENCY BAND ANALYSIS
#     Which mel bands activate most for fake vs real?
# ─────────────────────────────────────────────

def analyze_frequency_bands(results, n_mels=64, sr=16000, save_path="frequency_analysis.png"):
    """
    Compute mean Grad-CAM activation per mel band for real vs fake.
    Shows which frequency regions the model relies on.
    """
    real_acts = []
    fake_acts = []

    for r in results:
        heatmap = r['heatmap']
        # Resize heatmap to (n_mels, W)
        hm_resized = tf.image.resize(
            heatmap[..., np.newaxis], [n_mels, heatmap.shape[1]]
        ).numpy()[..., 0]

        # Mean activation per mel band (across time)
        band_activation = hm_resized.mean(axis=1)   # (n_mels,)

        if r['true'] == 0:
            real_acts.append(band_activation)
        else:
            fake_acts.append(band_activation)

    if not real_acts or not fake_acts:
        print("[WARN] Need both real and fake samples for frequency analysis.")
        return

    real_mean = np.array(real_acts).mean(axis=0)
    fake_mean = np.array(fake_acts).mean(axis=0)

    # Mel band center frequencies
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grad-CAM Frequency Band Analysis: Where does the model look?",
                 fontsize=12, fontweight='bold')

    # Left — absolute activation per class
    axes[0].plot(mel_freqs, real_mean, label='Real audio', color='steelblue', linewidth=2)
    axes[0].plot(mel_freqs, fake_mean, label='Fake audio', color='crimson',   linewidth=2)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Mean Grad-CAM Activation")
    axes[0].set_title("Activation by Frequency Band")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Annotate peak fake activation band
    peak_band = np.argmax(fake_mean)
    peak_freq = mel_freqs[peak_band]
    axes[0].axvline(x=peak_freq, color='crimson', linestyle='--', alpha=0.5)
    axes[0].text(peak_freq * 1.05, fake_mean.max() * 0.95,
                 f'Peak: {peak_freq:.0f} Hz', color='crimson', fontsize=9)

    # Right — difference (fake - real) shows what's unique to fake detection
    diff = fake_mean - real_mean
    colors = ['crimson' if d > 0 else 'steelblue' for d in diff]
    axes[1].bar(range(n_mels), diff, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.8)
    axes[1].set_xlabel("Mel Band Index")
    axes[1].set_ylabel("Activation Difference (Fake − Real)")
    axes[1].set_title("Frequency Bands Uniquely Important for Fake Detection")
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add secondary x-axis with approximate Hz labels
    n_ticks = 6
    tick_indices = np.linspace(0, n_mels - 1, n_ticks, dtype=int)
    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels([f"{mel_freqs[i]:.0f} Hz" for i in tick_indices], rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Frequency analysis saved → {save_path}")
    plt.show()
    plt.close()

    # Print top-5 fake-discriminative bands
    top5 = np.argsort(diff)[::-1][:5]
    print("\nTop 5 frequency bands most activated for FAKE detection:")
    for idx in top5:
        print(f"  Mel band {idx:3d}  (~{mel_freqs[idx]:6.0f} Hz)  diff={diff[idx]:+.4f}")


# ─────────────────────────────────────────────
# 5.  MAIN RUNNER
# ─────────────────────────────────────────────

def run_gradcam(
    cnn_model_path,
    test_audio_dir,
    output_dir="gradcam_outputs",
    last_conv_layer="conv2d_1",       # second Conv2D in build_cnn()
    n_samples=10,                      # how many files to visualize
    xgb_model_path=None               # optional, only for reference
):
    """
    Load trained CNN, run Grad-CAM on n_samples from test_audio_dir.

    Parameters
    ----------
    cnn_model_path  : str   path to saved 'cnn_model.keras'
    test_audio_dir  : str   folder with subfolders 'real/' and 'fake/'
    output_dir      : str   where to save output PNGs
    last_conv_layer : str   name of last Conv2D layer in your CNN
    n_samples       : int   number of files to analyze (total, balanced)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading CNN from {cnn_model_path}...")
    cnn_model = keras.models.load_model(cnn_model_path)
    cnn_model.summary()

    # Print layer names so you can verify last_conv_layer
    print("\nConv2D layers in your model:")
    for layer in cnn_model.layers:
        if 'conv' in layer.name:
            print(f"  {layer.name}  output_shape={layer.output.shape}")

    # Collect audio files
    audio_files = []
    import pandas as pd
    import random
    
    meta_path = os.path.join(test_audio_dir, 'meta.csv')
    if not os.path.exists(meta_path):
        print(f"[WARN] Metadata not found: {meta_path}")
    else:
        df = pd.read_csv(meta_path)
        fake_files = df[df['label'] == 'spoof']['file'].tolist()
        real_files = df[df['label'] == 'bona-fide']['file'].tolist()
        
        random.shuffle(fake_files)
        random.shuffle(real_files)
        
        # append (path, true_label) where 1=fake, 0=real
        for f in fake_files[:n_samples // 2]:
            audio_files.append((os.path.join(test_audio_dir, f), 1))
        for f in real_files[:n_samples // 2]:
            audio_files.append((os.path.join(test_audio_dir, f), 0))

    print(f"\nRunning Grad-CAM on {len(audio_files)} files...")

    results = []

    for audio_path, true_label in audio_files:
        print(f"\n  Processing: {Path(audio_path).name}  (true={'fake' if true_label else 'real'})")

        tensor, spec_db = audio_to_input(audio_path)
        if tensor is None:
            continue

        heatmap, pred_prob = make_gradcam_heatmap(
            tensor, cnn_model,
            last_conv_layer_name=last_conv_layer
        )

        pred_label = "fake" if pred_prob > 0.5 else "real"
        true_str   = "fake" if true_label == 1 else "real"
        correct    = pred_label == true_str

        print(f"    Pred: {pred_label.upper()} ({pred_prob*100:.1f}% fake)  "
              f"{'✓' if correct else '✗'}")

        save_path = os.path.join(
            output_dir,
            f"{true_str}_{Path(audio_path).stem}_gradcam.png"
        )

        plot_gradcam_single(
            audio_path, spec_db, heatmap, pred_prob, true_label,
            save_path=save_path
        )

        results.append({
            'path':    audio_path,
            'true':    true_label,
            'prob':    pred_prob,
            'correct': correct,
            'heatmap': heatmap,
            'spec':    spec_db,
        })

    # Summary plot
    if results:
        plot_gradcam_summary(results, save_path=os.path.join(output_dir, "gradcam_summary.png"))
        analyze_frequency_bands(results, save_path=os.path.join(output_dir, "frequency_analysis.png"))

    # Print overall stats
    n_correct = sum(r['correct'] for r in results)
    print(f"\n{'='*50}")
    print(f"Grad-CAM run complete.")
    print(f"  Files analyzed : {len(results)}")
    print(f"  Correct        : {n_correct}/{len(results)} ({100*n_correct/max(len(results),1):.1f}%)")
    print(f"  Outputs saved  : {output_dir}/")
    print(f"{'='*50}")

    return results


# ─────────────────────────────────────────────
# 6.  USAGE — edit these paths and run
# ─────────────────────────────────────────────

if __name__ == "__main__":

    results = run_gradcam(
        cnn_model_path  = "cnn_model.keras",       # path to your saved CNN
        test_audio_dir  = "release_in_the_wild",   # folder with meta.csv and audio files
        output_dir      = "gradcam_outputs",
        last_conv_layer = "conv2d_1",              # last Conv2D in build_cnn()
        n_samples       = 20,                      # 10 real + 10 fake
    )
