import os
import numpy as np
import pyaudio
import torch
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_from_data(data: torch.Tensor, frames_per_buffer: int = 1024, sample_rate: int = 8000):
    p = pyaudio.PyAudio()
    stream = p.open(
        format = pyaudio.paInt16,
        channels = data.shape[0],
        rate = sample_rate,
        frames_per_buffer=frames_per_buffer,
        output = True,
    )
    # Convert data to acceptable format.
    bytes_data = (data.numpy() * 32767).astype(np.int16).tobytes()
    stream.write(bytes_data)
    stream.stop_stream()
    stream.close()
    p.terminate()


class SileroTTSProcessor:
    """THe processor that leverages the Silero TTS model for text to speech processing.
    See https://habr.com/ru/post/549482/.
    """
    def __init__(self, language: str, speaker: str):
        self.model, self.symbols, self.sample_rate, _, self.apply_tts = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=speaker,
            verbose=False,
        )

    def process(self, text: str, filepath: str) -> torch.Tensor:
        audio = self.apply_tts(
            texts=[text],
            model=self.model,
            sample_rate=self.sample_rate,
            symbols=self.symbols,
            device=device,
        )
        if os.path.exists(filepath):
            os.remove(filepath)
        data = audio[0].unsqueeze(0)  # (channel, samples)
        return data
