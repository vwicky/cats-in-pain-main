import torch
import librosa
import random
from torch.utils.data import Dataset
import torch.nn.functional as F

import os
import json


from .audio_config import AudioConfig

class CatEmotionDataset(Dataset):
    def __init__(self, file_list, config: AudioConfig, train: bool = True):
        self.samples = file_list
        self.config = config
        self.train = train

    def __len__(self):
        return len(self.samples)

    def augment(self, waveform):
        # Time shift: Roll the audio left or right by up to 10% of its length.
        # Added dims=-1 to ensure it only rolls across the time axis.
        shift = int(random.uniform(-0.1, 0.1) * waveform.shape[-1])
        waveform = torch.roll(waveform, shifts=shift, dims=-1)

        # Random gain: Scale the amplitude between 80% and 120%
        gain = random.uniform(0.8, 1.2)
        waveform = waveform * gain

        return waveform

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ==========================================
        # 1. AUDIO LOADING
        # ==========================================
        # NOTE: We use librosa instead of torchaudio.load. 
        # This explicitly avoids a known macOS compatibility issue between 
        # PyTorch 2.9.1, TorchCodec, and FFmpeg. 
        # Librosa handles the loading safely and mathematically resamples 
        # to our config.sample_rate (e.g., 44.1kHz -> 16000Hz) automatically.
        waveform, _ = librosa.load(path, sr=self.config.sample_rate)
        
        # Convert to tensor and temporarily add a channel dim (1, Time) for processing
        waveform = torch.tensor(waveform).unsqueeze(0) 

        # ==========================================
        # 2. LENGTH STANDARDIZATION
        # ==========================================
        target_len = self.config.target_length
        current_len = waveform.shape[1]

        if current_len > target_len:
            # Truncate: Chop off the excess if it's too long
            waveform = waveform[:, :target_len]
        elif current_len < target_len:
            # Pad: Add digital silence (zeros) to the right if it's too short
            pad_amount = target_len - current_len
            waveform = F.pad(waveform, (0, pad_amount))

        # ==========================================
        # 3. DATA AUGMENTATION
        # ==========================================
        # NOTE: Best practice is probabilistic augmentation, not 100%.
        # We only augment during training, based on the config probability (e.g., 0.4).
        if self.train and random.random() < self.config.augmentation_prob:
            waveform = self.augment(waveform)

        # ==========================================
        # 4. FINAL FORMATTING
        # ==========================================
        # NOTE: CNN14 computes Mel-spectrograms internally on the GPU, 
        # so we pass the raw waveform directly instead of extracting MELs here.
        
        # FIX: Strip away the temporary channel dimension so shape is (Time,).
        # When the DataLoader groups these, the batch shape becomes (Batch, Time).
        # The PANNs backbone will safely add the necessary internal dimensions.
        waveform = waveform.squeeze(0)

        return waveform, label
    
class CatEmotionInferenceDataset(Dataset):
    def __init__(self, file_list, metadata_path, config):
        """
        file_list: list of audio file paths
        metadata_path: metadata.json for cat_proba
        config: AudioConfig
        """
        self.files = file_list
        self.config = config

        # Build snippet_id -> cat_proba mapping
        self.snippet_to_cat = self._load_metadata(metadata_path)

    def _load_metadata(self, metadata_path):
        mapping = {}

        with open(metadata_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                for snip in entry["snippets"]:
                    mapping[snip["id"]] = snip["audio_proba"]

        return mapping

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        filename = os.path.basename(path)
        snippet_id = filename.replace(".mp3", "")

        # ======================
        # 1. AUDIO LOADING
        # ======================
        waveform, _ = librosa.load(path, sr=self.config.sample_rate)
        waveform = torch.tensor(waveform).unsqueeze(0)

        # ======================
        # 2. LENGTH STANDARDIZATION
        # ======================
        target_len = self.config.target_length
        current_len = waveform.shape[1]

        if current_len > target_len:
            waveform = waveform[:, :target_len]
        elif current_len < target_len:
            pad_amount = target_len - current_len
            waveform = F.pad(waveform, (0, pad_amount))

        waveform = waveform.squeeze(0)

        # ======================
        # 3. Metadata
        # ======================
        cat_proba = self.snippet_to_cat[snippet_id]

        return {
            "audio": waveform,
            "snippet_id": snippet_id,
            "cat_proba": cat_proba
        }