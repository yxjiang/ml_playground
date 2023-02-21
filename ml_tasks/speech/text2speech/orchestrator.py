"""
The orchestrator that put all the text to speech sub-modules together.
"""
import nltk
from nltk.tokenize import sent_tokenize
import pyaudio
from threading import Thread
import wave

import processor


class TTSOrchestrator(Thread):
    """Tokenize the text, and then 
    """
    def __init__(self, text: str, sample_rate: int = 8000, language: str = 'en', read_mode: str = 'incremental'):
        super().__init__()
        self.text = text
        self.sample_rate = sample_rate
        self.language = language
        if self.language == 'en':
            self.voice = 'lj'
        elif self.language == 'ru':
            self.voice = 'natasha'
        elif self.language == 'es':
            self.voice = 'tux'
        elif self.language == 'de':
            self.voice = 'thorsten'
        elif self.language == 'fr':
            self.voice = 'gilles'
        else:
            raise ValueError(f'Unsupported language {self.language} for text to speech.')
        if sample_rate == 8000:
            self.voice += '_8khz'
        elif sample_rate == 16000:
            self.voice += '_16khz'
        else:
            raise ValueError(f'Unsupported sample rate {self.sample_rate}. Only supports 8khz and 16khz.')
        self.processor = processor.SileroTTSProcessor(language=self.language, speaker=self.voice)
        nltk.download('punkt')
        self.pyaudio = pyaudio.PyAudio()
        self.chunk = 1024
        self.stream: pyaudio.Stream = None  # initialize on first wav file load.
        self.read_mode = read_mode

    def __del__(self):
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()

    def run(self):
        """Tokenize the input text into sentences, and tts each sentence.
        """
        texts = sent_tokenize(self.text)
        if self.read_mode == 'incremental':
            for i, text in enumerate(texts):
                filepath = f'output_tts_{i}.wav'
                self.processor.process(text=text, filepath=filepath)
                processor.play_wav(filepath)
        elif self.read_mode == 'one_shot':
            wav_files = []
            for i, text in enumerate(texts):
                filepath = f'output_tts_{i}.wav'
                self.processor.process(text=text, filepath=filepath)
                wav_files.append(filepath)
            
            # Preload all files and create streams.
            data = bytearray()
            for wav_file in wav_files:
                wf = wave.open(wav_file, 'rb')
                if not self.stream:
                    self.stream = self.pyaudio.open(
                        format = self.pyaudio.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True,
                    )
                # Append all file contents to the data buffer.
                buffer = None
                while True:
                    buffer = wf.readframes(self.chunk)
                    if buffer or len(buffer):
                        data.extend(buffer)
                    else:
                        break
            self.stream.write(bytes(data))
