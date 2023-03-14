"""The core modeling part of speech to text.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
import torch
import torchaudio
from torchaudio.models.decoder._ctc_decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files


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

# Write a class that uses Silero STT model to do the speech to text processing. The method should name as 'process' and the constructor should accept the file_name of the audio.

class Wav2Vec2Processor(ASRProcessor):
    """Wav2Vec2: https://arxiv.org/pdf/1904.05862.pdf.
    """
    def __init__(self, args: Namespace, file_name: str):
        super().__init__(args=args, file_name=file_name)
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(device)
        if args.decoder == 'greedy':
            self.decoder = GreedyCTCDecoder(labels=self.bundle.get_labels())
        elif args.decoder == 'libri':
            pretrained_lm_files = download_pretrained_files('librispeech-4-gram')
            self.decoder = ctc_decoder(
                lexicon=pretrained_lm_files.lexicon,
                tokens=pretrained_lm_files.tokens
            )

    def process(self) -> str:
        """Load the audio from the file. Conduct resampling if needed. And then conduct inference and rescoring.
        """
        waveform, sample_rate = torchaudio.load(self.file_name)
        waveform = waveform.to(device)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)

        with torch.inference_mode():
            emission, _ = self.model(waveform)
            beam_search_result = self.decoder(emission)
            transcript = ' '.join(beam_search_result[0][0].words).strip()
            transcript = transcript.replace('|', ' ').lower()
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
