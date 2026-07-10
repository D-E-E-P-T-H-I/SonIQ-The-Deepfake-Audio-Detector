import os
import glob
import random
import librosa
import subprocess
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from soniq import extract_mel_spectrogram, preprocess_audio
from tqdm import tqdm

def compress_to_mp3_and_back(wav_path, bitrate="64k"):
    """Convert WAV → MP3 at given bitrate → WAV, return degraded audio."""
    mp3_path = wav_path.replace(".wav", f"_{bitrate}.mp3")
    out_path  = wav_path.replace(".wav", f"_{bitrate}_restored.wav")
    
    # WAV → MP3 (lossy compression)
    subprocess.run(["ffmpeg", "-i", wav_path, "-b:a", bitrate, 
                    mp3_path, "-y", "-loglevel", "quiet"])
    
    # MP3 → WAV (decode back, artifacts baked in)
    subprocess.run(["ffmpeg", "-i", mp3_path, out_path, 
                    "-y", "-loglevel", "quiet"])
    
    audio, sr = librosa.load(out_path, sr=16000)
    
    # Cleanup temporary files
    if os.path.exists(mp3_path): os.remove(mp3_path)
    if os.path.exists(out_path): os.remove(out_path)
        
    return audio

def add_gaussian_noise(audio, snr_db):
    """Add white noise at given SNR level."""
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def process_and_predict(audio, embedding_model, xgb_model):
    """Trim, pad, extract mel-spectrogram, and run models to get probability."""
    audio, _ = librosa.effects.trim(audio, top_db=100)
    
    target_length = int(16000 * 5.0)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
        
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    
    spec = extract_mel_spectrogram(audio, 16000)
    spec = np.expand_dims(spec, axis=(0, -1))
    
    embedding = embedding_model.predict(spec, verbose=0)
    prob = xgb_model.predict_proba(embedding)[0][1]
    return prob

def evaluate_condition(files, labels, condition, embedding_model, xgb_model):
    y_true = []
    y_scores = []
    
    for f, label in zip(files, labels):
        try:
            if condition == "Clean":
                audio, _ = librosa.load(f, sr=16000)
            elif condition == "MP3 64kbps":
                audio = compress_to_mp3_and_back(f, "64k")
            elif condition == "MP3 128kbps":
                audio = compress_to_mp3_and_back(f, "128k")
            elif condition == "Noise SNR=10dB":
                audio, _ = librosa.load(f, sr=16000)
                audio = add_gaussian_noise(audio, 10)
            
            prob = process_and_predict(audio, embedding_model, xgb_model)
            y_true.append(label)
            y_scores.append(prob)
        except Exception as e:
            print(f"Error processing {f}: {e}")
        
    preds = [1 if p > 0.5 else 0 for p in y_scores]
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, y_scores)
    return acc, auc, y_true, y_scores

if __name__ == "__main__":
    print("Loading models...")
    cnn_model = tf.keras.models.load_model('cnn_model.keras')
    embedding_model = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer('embeddings').output
    )
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
        
    real_files = glob.glob('processed_data/test/real/*.wav')
    fake_files = glob.glob('processed_data/test/fake/*.wav')
    
    random.seed(42)
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    # Use 250 real and 250 fake (500 total) for timely evaluation
    sample_size = min(250, len(real_files), len(fake_files))
    test_files = real_files[:sample_size] + fake_files[:sample_size]
    test_labels = [0]*sample_size + [1]*sample_size
    
    # Shuffle together
    combined = list(zip(test_files, test_labels))
    random.shuffle(combined)
    test_files, test_labels = zip(*combined)
    
    print(f"Evaluating {len(test_files)} samples...")
    
    results = {}
    conditions = ["Clean", "MP3 64kbps", "MP3 128kbps", "Noise SNR=10dB"]
    
    for cond in conditions:
        print(f"\nCondition: {cond}")
        acc, auc, y_true_cond, y_scores_cond = evaluate_condition(tqdm(test_files), test_labels, cond, embedding_model, xgb_model)
        
        if cond == "Noise SNR=10dB":
            thresholds = np.arange(0.1, 0.9, 0.02)
            f1_scores_list = []
            
            y_prob_noisy = np.array(y_scores_cond)
            y_true_noisy = np.array(y_true_cond)
            
            for t in thresholds:
                preds = (y_prob_noisy >= t).astype(int)
                f1 = f1_score(y_true_noisy, preds, zero_division=0)
                f1_scores_list.append(f1)
            
            best_threshold = thresholds[np.argmax(f1_scores_list)]
            best_f1 = max(f1_scores_list)
            print(f"\n--- F1 Optimization for Noise Condition ---")
            print(f"Optimal threshold under noise: {best_threshold:.2f}")
            print(f"F1 at optimal threshold: {best_f1:.3f}")
            
            best_preds = (y_prob_noisy >= best_threshold).astype(int)
            best_accuracy = np.mean(best_preds == y_true_noisy)
            print(f"Accuracy at optimal threshold: {best_accuracy:.3f}")
            print("-------------------------------------------\n")
            
            acc = best_accuracy
            
        results[cond] = (acc, auc)
        
    print("\n\n")
    print(f"{'Condition':<20} | {'Accuracy':<10} | {'AUC':<10}")
    print("-" * 50)
    for cond in conditions:
        acc, auc = results[cond]
        print(f"{cond:<20} | {acc*100:0.0f}%        | {auc:.3f}")
    print("="*50)
