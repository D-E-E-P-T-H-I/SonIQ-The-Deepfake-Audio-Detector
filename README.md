# SonIQ: Audio Deepfake Detection

SonIQ is a hybrid audio deepfake detection system that combines deep learning and traditional machine learning to accurately distinguish between real human speech and AI-generated audio. The project emphasizes robustness, generalization, and real-world deployability rather than dataset-specific overfitting.

## Overview

Advances in speech synthesis and voice conversion have enabled highly realistic audio deepfakes, leading to risks such as fraud, impersonation, misinformation, and loss of media trust. SonIQ addresses these challenges by detecting subtle audio artifacts that are imperceptible to human listeners.

The system converts audio signals into time–frequency representations and uses a hybrid **CNN–XGBoost** architecture to deliver strong performance across diverse real-world conditions.

## Key Features

- Hybrid CNN–XGBoost architecture for improved generalization
- Robust preprocessing pipeline for noisy real-world audio
- Mel-spectrogram and MFCC-based feature extraction
- Probability-based deepfake scoring
- Lightweight and deployable design

## Dataset

- **Source**: In-the-Wild Audio Deepfake Dataset
- **Split**:
  - Training: 70%
  - Validation: 15%
  - Test: 15%
- Contains verified human speech and AI-generated counterparts with high diversity

## Methodology

### Preprocessing

- Silence and background noise removal
- Resampling to 16 kHz
- Audio duration normalization to 5 seconds
- Amplitude normalization

### Feature Extraction

- Mel-spectrograms (64 mel bands)
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Prosodic features (pitch, energy, jitter)
- Phase-based features
- Final spectrogram shape: `(64, 79, 1)`

### Model Architecture

#### CNN (Feature Extractor)

- Two convolutional layers with max pooling and dropout
- Global average pooling
- 128-dimensional embedding layer
- Approximately 150,000 trainable parameters

#### XGBoost (Classifier)

- Trained on CNN embeddings
- 100 decision trees with max depth of 6
- Early stopping to prevent overfitting
- Outputs probability score indicating likelihood of deepfake

## Performance

- **Accuracy**: 97%
- **AUC**: 0.9941
- Evaluation metrics include precision, recall, F1-score, ROC curve, and confusion matrix
- Demonstrates strong generalization on unseen real-world audio

## Strengths

- High robustness to background noise and acoustic variation
- Balanced specialization and generalization
- Scalable and modular pipeline
- Reduced overfitting compared to single-model approaches

## Limitations

- Currently trained only on English audio
- CNN training requires significant GPU resources
- Not optimized for real-time inference
- May struggle with future advanced synthesis techniques

## Applications

- Voice authentication and banking security
- Media verification and journalism
- Social media content moderation
- Fraud and misinformation prevention systems

## Future Work

- Real-time deepfake detection for live calls
- Multilingual and cross-accent support
- Integration with video deepfake detection
- Continuous updates to counter evolving synthesis methods

## Collaborators

- **Deepthi K**
- **Sucheth Katte** 
- **Devansh Shah** 
- **Rigved Patil**

## Conclusion

SonIQ demonstrates that combining deep learning feature extraction with traditional machine learning classification can produce a robust, generalizable, and practical solution for audio deepfake detection. The hybrid CNN–XGBoost architecture provides a strong foundation for future work in cybersecurity and digital forensics.
