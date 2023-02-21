import pyaudio
import torch
import torchaudio
import wave


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_wav(filepath: str):
    """Play the wave file. Copy from the pyaudio tutorial.
    """
    chunk = 1024
    wf = wave.open(filepath, 'rb')
    p = pyaudio.PyAudio()
    # Open a .Stream object to write the WAV file to a stream.
    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True,
    )
    # Play from the wave file.
    while len(data := wf.readframes(chunk)):
        stream.write(data)
    
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
        )

    def process(self, text: str):
        audio = self.apply_tts(
            texts=[text],
            model=self.model,
            sample_rate=self.sample_rate,
            symbols=self.symbols,
            device=device,
        )
        filepath = 'output_tts.wav'
        torchaudio.save(
            filepath=filepath,
            src=audio[0].unsqueeze(0),
            sample_rate=self.sample_rate,
            bits_per_sample=16,
        )
        return filepath
