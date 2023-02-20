"""The core modeling part of speech to text.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
import torch
import torchaudio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ASRProcessor(ABC):
    """The base class of all ASR processors.
    """
    def __init__(self, args: Namespace, file_name: str):
        self.args = args
        self.file_name = file_name

    @abstractmethod
    def process(self):
        pass
    

class Wav2VecProcessor(ASRProcessor):
    def __init__(self, args: Namespace, file_name: str):
        super().__init__(args=args, file_name=file_name)
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(device)
        self.decoder = GreedyCTCDecoder(labels=self.bundle.get_labels())

    def process(self) -> str:
        waveform, sample_rate = torchaudio.load(self.file_name)
        waveform = waveform.to(device)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)

        with torch.inference_mode():
            emission, _ = self.model(waveform)
            transcript = self.decoder(emission[0]).replace('|', ' ').lower()
            return transcript
            

class GreedyCTCDecoder(torch.nn.Module):
    """Simplest CTC decoder. Copied from pytorch audio processing tutorial.
    """
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
