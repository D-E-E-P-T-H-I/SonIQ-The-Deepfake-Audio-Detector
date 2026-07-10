#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
from tqdm import tqdm
import pickle
import shutil
import random
import glob

# Load and clean up audio files
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
    except:
        return None, None

# Convert audio to mel spectrogram
def extract_mel_spectrogram(audio, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, 
        n_mels=64,
        hop_length=1024
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Split our dataset into train, validation and test sets
def split_dataset(source_folder, dest_folder):
    print("\nSplitting dataset into train/val/test...")

    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(dest_folder, split, cls), exist_ok=True)

    import pandas as pd
    meta_path = os.path.join(source_folder, 'meta.csv')
    df = pd.read_csv(meta_path)
    
    fake_files = df[df['label'] == 'spoof']['file'].tolist()
    real_files = df[df['label'] == 'bona-fide']['file'].tolist()
    random.shuffle(fake_files)
    random.shuffle(real_files)

    for cls, files in [('fake', fake_files), ('real', real_files)]:
        n = len(files)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        for i, f in enumerate(files):
            if i < train_end:
                split = 'train'
            elif i < val_end:
                split = 'val'
            else:
                split = 'test'
            
            src_path = os.path.join(source_folder, f)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(dest_folder, split, cls, f))

        print(f"{cls}: Train={train_end}, Val={val_end-train_end}, Test={n-val_end}")

def augment_with_noise(audio, snr_range=(15, 40)):
    snr_db = np.random.uniform(*snr_range)
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

# Load audio files and convert them to spectrograms
def load_spectrograms(data_dir, split, augment=False):
    X, y = [], []

    for label, cls in enumerate(['real', 'fake']):
        folder = os.path.join(data_dir, split, cls)
        files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))]

        for f in tqdm(files, desc=f"Loading {split}/{cls}"):
            audio, sr = preprocess_audio(os.path.join(folder, f))
            if audio is not None:
                if augment and np.random.random() > 0.5:
                    audio = augment_with_noise(audio, snr_range=(15, 35))
                spec = extract_mel_spectrogram(audio, sr)
                spec = np.expand_dims(spec, axis=-1)
                X.append(spec)
                y.append(label)

    return np.array(X), np.array(y)

# Build our CNN model
def build_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Second convolutional block
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Extract features
    x = layers.GlobalAveragePooling2D()(x)
    embeddings = layers.Dense(128, activation='relu', name='embeddings')(x)
    x = layers.Dropout(0.5)(embeddings)

    # Final classification
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Train the CNN
def train_cnn(X_train, y_train, X_val, y_val):
    print("\nTraining CNN model...")

    model = build_cnn(X_train.shape[1:])

    # Define early stopping to prevent overtraining
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3, 
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when stuck
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return model, history

# Extract embeddings from CNN in batches to avoid memory issues
def extract_embeddings(cnn_model, X, batch_size=32):
    embedding_model = keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer('embeddings').output
    )

    embeddings = []
    num_batches = int(np.ceil(len(X) / batch_size))

    print(f"Extracting embeddings from {len(X)} samples...")
    for i in tqdm(range(0, len(X), batch_size)):
        batch = X[i:i+batch_size]
        batch_embeddings = embedding_model.predict(batch, verbose=0)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

# Train XGBoost classifier on the CNN embeddings
def train_xgboost(X_train, y_train, X_val, y_val):
    print("\nTraining XGBoost classifier...")

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        early_stopping_rounds=10
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    return model

# Evaluate our hybrid model
def evaluate_hybrid_model(cnn_model, xgb_model, X_test, y_test):
    print("\nEvaluating model on test set...")

    # Get embeddings from CNN
    embeddings = extract_embeddings(cnn_model, X_test)

    # Make predictions
    y_pred_proba = xgb_model.predict_proba(embeddings)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Print classification results
    print("\nResults:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=20)
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.yticks([0, 1], ['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

    print(f"\nFinal AUC Score: {roc_auc:.4f}")
    return roc_auc

# Main training pipeline
def main():
    print("="*60)
    print("Audio Deepfake Detection System")
    print("Hybrid CNN + XGBoost Model")
    print("="*60)

    # Dataset path
    source_folder = 'release_in_the_wild'
    dest_folder = 'processed_data'

    # Split the dataset
    split_dataset(source_folder, dest_folder)

    # Load spectrograms
    print("\nLoading spectrograms...")
    X_train, y_train = load_spectrograms(dest_folder, 'train', augment=True)
    X_val, y_val = load_spectrograms(dest_folder, 'val')
    X_test, y_test = load_spectrograms(dest_folder, 'test')

    print(f"\nDataset loaded:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Train CNN
    cnn_model, history = train_cnn(X_train, y_train, X_val, y_val)
    print("\nSaving CNN model to cnn_model.keras...")
    cnn_model.save('cnn_model.keras')

    # Extract embeddings
    print("\nExtracting embeddings...")
    train_embeddings = extract_embeddings(cnn_model, X_train)
    val_embeddings = extract_embeddings(cnn_model, X_val)
    print(f"Embedding dimension: {train_embeddings.shape}")

    # Train XGBoost
    xgb_model = train_xgboost(train_embeddings, y_train, val_embeddings, y_val)

    # Evaluate
    auc_score = evaluate_hybrid_model(cnn_model, xgb_model, X_test, y_test)

    # Save models
    print("\nSaving models...")
    cnn_model.save('cnn_model.keras')
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("Models saved successfully!")

    print("\n" + "="*60)
    print(f"Training Complete! Final AUC: {auc_score:.4f}")
    print("="*60)

    return cnn_model, xgb_model

# Predict if an audio file is real or fake
def predict_audio(audio_path, cnn_model, xgb_model):
    # Load and preprocess audio
    audio, sr = preprocess_audio(audio_path)
    if audio is None:
        return None, None

    # Convert to spectrogram
    spec = extract_mel_spectrogram(audio, sr)
    spec = np.expand_dims(spec, axis=(0, -1))

    # Get embedding from CNN
    embedding_model = keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer('embeddings').output
    )
    embedding = embedding_model.predict(spec, verbose=0)

    # Predict with XGBoost
    prob = xgb_model.predict_proba(embedding)[0, 1]

    label = 'Fake' if prob > 0.5 else 'Real'
    confidence = prob * 100 if label == 'Fake' else (1 - prob) * 100

    return label, confidence

# Run everything
if __name__ == "__main__":
    # Train the model
    cnn_model, xgb_model = main()

    # Test predictions on some sample files
    print("\n" + "="*60)
    print("Testing predictions on sample audio files")
    print("="*60)

    real_files = glob.glob('processed_data/test/real/*.wav')[:3]
    fake_files = glob.glob('processed_data/test/fake/*.wav')[:3]

    print("\nReal Audio Samples:")
    for i, audio_path in enumerate(real_files, 1):
        label, conf = predict_audio(audio_path, cnn_model, xgb_model)
        filename = os.path.basename(audio_path)
        print(f"{i}. {filename}: {label} ({conf:.1f}% confidence)")

    print("\nFake Audio Samples:")
    for i, audio_path in enumerate(fake_files, 1):
        label, conf = predict_audio(audio_path, cnn_model, xgb_model)
        filename = os.path.basename(audio_path)
        print(f"{i}. {filename}: {label} ({conf:.1f}% confidence)")

    print("\n" + "="*60)
    print("System ready for predictions!")
    print("="*60)


# In[ ]:




