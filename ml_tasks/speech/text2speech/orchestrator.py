"""
The orchestrator that put all the text to speech sub-modules together.
"""
from argparse import Namespace
import nltk
from nltk.tokenize import sent_tokenize
from threading import Thread

import processor


class TTSOrchestrator(Thread):
    """Tokenize the text, and then 
    """
    def __init__(self, text: str, sample_rate: int = 8000, language: str = 'en'):
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


    def run(self):
        """Tokenize the input text into sentences, and tts each sentence.
        """
        texts = sent_tokenize(self.text)
        for text in texts:
            wav_file = self.processor.process(text)
            processor.play_wav(wav_file)


