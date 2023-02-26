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
import sys

from processor import Wav2Vec2Processor


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
        self.processor = Wav2Vec2Processor(args=args, file_name=self.args.file_path)

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
        self.listener.daemon = True  # Set listener as daemon so it can be killed when the program exits.
        self.pyaudio = PyAudio()

    def run(self, acc_transcript: str = ''):
        """Use an infinite loop to retrieve the raw input from the audio stream.
        Process the input only when there are enough frames of data.

        Args:
            acc_transcript: Used to store the accumulated transcripts. The caller can get it externally.
        """
        self.listener.start()
        print('Start to listen...')
        no_input = 0
        try:
            while True:
                if len(self.frames) < self.args.buffer_size:
                    print(f'{len(self.frames)}')
                    continue
                else:
                    raw_input = self.frames.copy()
                    self.frames.clear()
                    transcript = self._process(raw_input)
                    size = len(transcript)
                    if size > 0:
                        no_input = 0
                        acc_transcript += ' ' + transcript
                    else:
                        no_input += 1
                    # Stop if no meaningful transcript for a few times.
                    if self.args.no_input_retry != -1 and no_input == self.args.no_input_retry:
                        print(f'[No input for {self.args.no_input_retry} times, terminate.]')
                        raise SystemExit
                time.sleep(0.1)
        finally:
            print('ASR stopped')

    def _process(self, raw_input: List[bytes]) -> str:
        """Process the raw_input. Predict and get the result. Then convert to text.
        """
        self._save(raw_input=raw_input)
        transcript = self.processor.process()
        print(f'{transcript}')
        return transcript


    def _save(self, raw_input: List[bytes]):
        """Save the audio data into temp file.
        """
        obj = wave.open(self.args.file_path, 'wb')
        obj.setnchannels(1)
        obj.setsampwidth(self.pyaudio.get_sample_size(paInt16))
        obj.setframerate(self.args.sample_rate)
        obj.writeframes(b''.join(raw_input))
