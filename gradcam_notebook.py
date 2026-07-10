"""
NOTEBOOK VERSION — paste these cells directly into your soniq.ipynb
after the main() call that trains and saves your models.
"""

# ── CELL 1: Install / imports ──────────────────────────────────────────
# (tf, keras, librosa, numpy, matplotlib already imported in your notebook)
# No new pip installs needed — Grad-CAM uses only TF primitives

# ── CELL 2: Core Grad-CAM function ────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_1"):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        pred_prob = float(predictions[0][0])
        class_channel = predictions[:, 0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_prob


# ── CELL 3: Visualization helper ──────────────────────────────────────

def visualize_gradcam(audio_path, cnn_model, true_label_str="fake",
                      last_conv_layer="conv2d_1"):
    """Quick visualization for one audio file. true_label_str: 'real' or 'fake'"""
    audio, sr = preprocess_audio(audio_path)
    if audio is None:
        print("Could not load audio"); return

    spec = extract_mel_spectrogram(audio, sr)
    tensor = np.expand_dims(spec, axis=(0, -1))

    heatmap, pred_prob = make_gradcam_heatmap(tensor, cnn_model, last_conv_layer)
    pred_label = "FAKE" if pred_prob > 0.5 else "REAL"

    # Resize heatmap to spectrogram dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], [spec.shape[0], spec.shape[1]]
    ).numpy()[..., 0]

    # Build overlay
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec_rgb = np.stack([spec_norm]*3, axis=-1)
    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)[..., :3]
    overlay = np.clip(0.5 * spec_rgb + 0.5 * heatmap_colored, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"True: {true_label_str.upper()}  |  Pred: {pred_label}  "
        f"({pred_prob*100:.1f}% fake)",
        fontsize=12, fontweight='bold',
        color='green' if pred_label.lower() == true_label_str else 'red'
    )

    librosa.display.specshow(spec, sr=sr, hop_length=1024,
                              x_axis='time', y_axis='mel',
                              ax=axes[0], cmap='magma')
    axes[0].set_title("Mel-Spectrogram")

    axes[1].imshow(heatmap_resized, cmap='jet', aspect='auto', origin='lower')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].set_xlabel("Time frames"); axes[1].set_ylabel("Mel bands")

    axes[2].imshow(overlay, aspect='auto', origin='lower')
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Time frames"); axes[2].set_ylabel("Mel bands")

    plt.tight_layout()
    plt.savefig(f"gradcam_{true_label_str}_{pred_label}.png", dpi=150)
    plt.show()
    print(f"Prediction: {pred_label} ({pred_prob*100:.1f}% fake)")


# ── CELL 4: Run on a few test files ───────────────────────────────────

# After training, run these — edit paths to your actual test files

# Example: visualize one fake and one real file
import os

test_fake_dir = "processed_data/test/fake"
test_real_dir = "processed_data/test/real"

fake_files = [f for f in os.listdir(test_fake_dir) if f.endswith('.wav')][:3]
real_files = [f for f in os.listdir(test_real_dir) if f.endswith('.wav')][:3]

print("=== FAKE AUDIO GRAD-CAM ===")
for f in fake_files:
    visualize_gradcam(
        os.path.join(test_fake_dir, f),
        cnn_model,           # your trained CNN from main()
        true_label_str="fake"
    )

print("\n=== REAL AUDIO GRAD-CAM ===")
for f in real_files:
    visualize_gradcam(
        os.path.join(test_real_dir, f),
        cnn_model,
        true_label_str="real"
    )


# ── CELL 5: Frequency band analysis ───────────────────────────────────

def frequency_band_analysis(test_dir, cnn_model, n_per_class=30,
                             n_mels=64, sr=16000):
    """
    For n_per_class real and fake files, compute mean Grad-CAM
    activation per mel band. Reveals which frequencies drive detection.
    """
    real_acts, fake_acts = [], []
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr//2)

    for cls, label, store in [('real', 0, real_acts), ('fake', 1, fake_acts)]:
        folder = os.path.join(test_dir, cls)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')][:n_per_class]
        print(f"Analyzing {len(files)} {cls} files...")

        for fname in files:
            audio, sr_ = preprocess_audio(os.path.join(folder, fname))
            if audio is None: continue
            spec = extract_mel_spectrogram(audio, sr_)
            tensor = np.expand_dims(spec, axis=(0, -1))
            heatmap, _ = make_gradcam_heatmap(tensor, cnn_model)
            hm = tf.image.resize(heatmap[..., np.newaxis],
                                  [n_mels, heatmap.shape[1]]).numpy()[..., 0]
            store.append(hm.mean(axis=1))   # mean over time

    real_mean = np.array(real_acts).mean(axis=0)
    fake_mean = np.array(fake_acts).mean(axis=0)
    diff = fake_mean - real_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Which frequencies drive fake detection?",
                 fontsize=12, fontweight='bold')

    axes[0].plot(mel_freqs, real_mean, label='Real', color='steelblue', lw=2)
    axes[0].plot(mel_freqs, fake_mean, label='Fake', color='crimson',   lw=2)
    axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("Mean Grad-CAM")
    axes[0].set_title("Activation per Frequency Band")
    axes[0].set_xscale('log'); axes[0].legend(); axes[0].grid(alpha=0.3)

    colors = ['crimson' if d > 0 else 'steelblue' for d in diff]
    axes[1].bar(range(n_mels), diff, color=colors, alpha=0.7)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_xlabel("Mel Band Index")
    axes[1].set_ylabel("Fake − Real activation")
    axes[1].set_title("Bands Uniquely Important for Fake Detection")
    tick_idx = np.linspace(0, n_mels-1, 6, dtype=int)
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([f"{mel_freqs[i]:.0f}Hz" for i in tick_idx], rotation=30)
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("frequency_analysis.png", dpi=150)
    plt.show()

    top5 = np.argsort(diff)[::-1][:5]
    print("\nTop 5 frequency bands most activated for FAKE detection:")
    for i in top5:
        print(f"  Band {i:3d}  ~{mel_freqs[i]:6.0f} Hz   diff={diff[i]:+.4f}")


# Run frequency analysis
frequency_band_analysis("processed_data/test", cnn_model, n_per_class=30)
