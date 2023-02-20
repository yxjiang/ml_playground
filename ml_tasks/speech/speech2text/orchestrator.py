"""
The orchestrator that put all the speech to text sub-modules together.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from pyaudio import PyAudio, paInt16
import time
from threading import Thread
from typing import List
import wave

from processor import Wav2VecProcessor



class Listener(Thread):
    """The listener to handle the stream input.
    """

    def __init__(self, args: Namespace, frames: List[bytes]):
        """
        Args:
            sample_rate: The sample rate of the input audio.
            frames_per_buffer: Determines how many frames to read at a time in the input stream.
                It is a trade-off between throughput and latency.
        """
        super().__init__()
        self.sample_rate = args.sample_rate
        self.frames_per_buffer = args.frames_per_buffer
        self.stream = PyAudio().open(
            format=paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        self.frames = frames

    def run(self):
        """Listen forever.
        """
        while True:
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            self.frames.append(data)
            time.sleep(0.01)


class Orchestrator(ABC):
    """Abstract class for orchestrators of different tasks.
    """
    def __init__(self, args: Namespace):
        self.args = args
        self.processor = Wav2VecProcessor(args=args, file_name=self.args.file_path)

    @abstractmethod
    def run(self):
        pass


class BatchOrchestrator(Orchestrator):
    """The orchestrator that loads a file and generate the text.
    """
    def __init__(self, args: Namespace):
        super().__init__(args)


    def run(self):
        """Load the wav file and translate it into text.
        """
        transcript = self.processor.process()
        print(f'Transcript: {transcript}')




class StreamOrchestrator(Orchestrator):
    """The orchestrator that process real time speech to text.
    """
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.frames = []
        self.listener = Listener(args=args, frames=self.frames)
        self.pyaudio = PyAudio()

    def run(self):
        """Use an infinite loop to retrieve the raw input from the audio stream.
        Process the input only when there are enough frames of data.
        """
        listener = Listener(args=self.args, frames=self.frames)
        listener.start()
        print('Start to listen...')
        while True:
            if len(self.frames) < self.args.buffer_size:
                continue
            else:
                raw_input = self.frames.copy()
                self.frames.clear()
                self._process(raw_input)
            time.sleep(0.1)

    def _process(self, raw_input: List[bytes]):
        """Process the raw_input. Predict and get the result. Then convert to text.
        """
        self._save(raw_input=raw_input)
        self.processor.process()
        print('Process recognized text...')

    def _save(self, raw_input: List[bytes]):
        """Save the audio data into temp file.
        """
        obj = wave.open(self.tmp_file_name, 'wb')
        obj.setnchannels(1)
        obj.setsampwidth(self.pyaudio.get_sample_size(paInt16))
        obj.setframerate(self.args.sample_rate)
        obj.writeframes(b''.join(raw_input))


