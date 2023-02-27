"""
The orchestrator that put all the text to speech sub-modules together.
"""
import nltk
from nltk.tokenize import sent_tokenize
import os
import pyaudio
from datetime import datetime
from torch import cat
from threading import Thread


from speech.text2speech.processor import SileroTTSProcessor, play_from_data


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
        self.processor = SileroTTSProcessor(language=self.language, speaker=self.voice)
        # nltk.download('punkt')
        self.pyaudio = pyaudio.PyAudio()
        self.chunk = 1024
        self.stream: pyaudio.Stream = None  # initialize on first wav file load.
        self.read_mode = read_mode
        self.tmp_tts_folder = './tmp_tts_folder'
        if not os.path.exists(self.tmp_tts_folder):
            os.makedirs(self.tmp_tts_folder)

    def stop(self):
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()

    def process(self):
        """Tokenize the input text into sentences, and tts each sentence.
        """
        texts = sent_tokenize(self.text)
        if self.read_mode == 'incremental':
            for i, text in enumerate(texts):
                filepath = os.path.join(self.tmp_tts_folder, f'output_tts_{datetime.now()}.wav')
                if os.path.exists(filepath):
                    os.remove(filepath)
                data = self.processor.process(text=text, filepath=filepath)
                # # Wait for tmp wav file to be created.
                # while not os.path.exists(filepath):
                #     print(f'wave file [{filepath}] not available')
                #     time.sleep(0.02)
                # # Wait for tmp wav file to be stable.
                # file_size = 0
                # while (new_file_size := os.path.getsize(filepath)) != file_size:
                #     print(f'wave file [{filepath}] not ready {file_size} -> {new_file_size}')
                #     file_size = new_file_size
                #     time.sleep(0.02)
                # time.sleep(1)
                play_from_data(data)
        elif self.read_mode == 'one_shot':
            data_list = []
            wav_files = []
            for text in texts:
                filepath = os.path.join(self.tmp_tts_folder, f'output_tts_{datetime.now()}.wav')
                if os.path.exists(filepath):
                    os.remove(filepath)
                data_list.append(self.processor.process(text=text, filepath=filepath))
                wav_files.append(filepath)

            data = cat(data_list, dim=1)
            play_from_data(data)
            
            # # Preload all files and create streams.
            # data = bytearray()
            # for i, wav_file in enumerate(wav_files):
            #     # Wait for tmp wav file to be created.
            #     while not os.path.exists(wav_file):
            #         print(f'wave file [{wav_file}] not available')
            #         time.sleep(0.02)
            #     # Wait for tmp wav file to be stable.
            #     file_size = 0
            #     while (new_file_size := os.path.getsize(wav_file)) != file_size:
            #         print(f'wave file [{wav_file}] not ready {file_size} -> {new_file_size}')
            #         file_size = new_file_size
            #         time.sleep(0.02)
            #     wf = wave.open(wav_file, 'rb')
            #     if not self.stream:
            #         self.stream = self.pyaudio.open(
            #             format = self.pyaudio.get_format_from_width(wf.getsampwidth()),
            #             channels = wf.getnchannels(),
            #             rate = wf.getframerate(),
            #             output = True,
            #         )
            #     # Append all file contents to the data buffer.
            #     buffer = None
            #     while True:
            #         buffer = wf.readframes(self.chunk)
            #         if buffer and len(buffer):
            #             data.extend(buffer)
            #         else:
            #             break
            # # self.stream.write(bytes(data))
        else:
            pass
        self.stop()

    def run(self):
        self.process()
