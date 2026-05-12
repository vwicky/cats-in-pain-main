from dataclasses import dataclass, field

@dataclass
class AudioConfig:
    # Audio & Spectrogram settings
    sample_rate: int = 16000
    window_size: int = 1024
    hop_size: int = 320
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 8000

    # Path to the downloaded PANNs weights
    checkpoint_path: str = "audio_classifier_utils/pretrained_weights/Cnn14_16k_mAP=0.438.pth"
    
    # Training settings
    num_classes: int = 10
    augmentation_prob: float = 0.4

    # Target length (7 seconds). We initialize it via __post_init__ 
    target_length: int = field(init=False)

    def __post_init__(self):
        # Based on EDA: 95th percentile is 6.81s. Setting to 7 seconds.
        # This guarantees it always perfectly matches your sample_rate!
        self.target_length = 7 * self.sample_rate  # 112,000 samples