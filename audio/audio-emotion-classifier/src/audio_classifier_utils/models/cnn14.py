import torch
import torch.nn as nn
from .panns import Cnn14  # Adjust relative import as needed

# model downloaded from: https://zenodo.org/records/3987831
# file name: Cnn14_16k_mAP=0.438.pth, 358.7 MB

from ..audio_config import AudioConfig

class CatEmotionModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        
        # 1. The backbone takes raw audio and converts it to a 64-bin Mel Spectrogram internally
        self.backbone = Cnn14(
            sample_rate=config.sample_rate,
            window_size=config.window_size,
            hop_size=config.hop_size,
            mel_bins=config.mel_bins,
            fmin=config.fmin,
            fmax=config.fmax,
            classes_num=527
        )
        self.config = config
        
        # 2. LOAD THE PRE-TRAINED WEIGHTS
        print(f"Loading pre-trained weights from {config.checkpoint_path}...")
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']

        # 1. Remove the mathematical STFT/Mel keys from the loaded dictionary 
        # so they don't overwrite our custom 1024-sized mathematical matrices
        keys_to_delete = [
            'spectrogram_extractor.stft.conv_real.weight',
            'spectrogram_extractor.stft.conv_imag.weight',
            'logmel_extractor.melW'
        ]
        for key in keys_to_delete:
            if key in state_dict:
                del state_dict[key]

        # 2. Load the rest of the neural network (the actual learned CNN weights)
        self.backbone.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully!")
        
        # 3. Swap the classification head
        # We use strict=False above so it doesn't crash when we overwrite this layer
        self.backbone.fc_audioset = nn.Identity() 
        self.classifier = nn.Linear(2048, config.num_classes)

    def forward(self, x):
        # x is raw waveform: shape (Batch, 1, Time)
        
        # PANNs returns a dictionary. We want the 'embedding' feature vector.
        output_dict = self.backbone(x)
        embedding = output_dict['embedding'] # Shape: (Batch, 2048)
        
        # Pass through your new classifier
        out = self.classifier(embedding) # Shape: (Batch, 10)
        
        return out
    
    # --- FINE TUNING PHASES ---

    def freeze_for_phase1(self):
        """Phase 1 (5-10 epochs): Freeze backbone, train only the head."""
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False
            
        # Unfreeze ONLY the new classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_for_phase2(self):
        """Phase 2 (20-40 epochs): Unfreeze last 2-3 blocks."""
        # Unfreeze specific high-level blocks (CNN14 has blocks 1 through 6)
        # We will unfreeze block 5, block 6, and the fully connected layers
        blocks_to_unfreeze = [
            self.backbone.conv_block5,
            self.backbone.conv_block6,
            self.backbone.fc1, # Internal dense layer before classifier
            self.classifier    # Keep the head unfrozen!
        ]
        
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True