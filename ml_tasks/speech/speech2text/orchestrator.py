"""
The orchestrator that put all the speech to text sub-modules together.
"""
from argparse import Namespace
import pyaudio
import time
from threading import Thread
from typing import List


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
        self.stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
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
            if self.frames:
                print(len(self.frames))
            time.sleep(0.1)


class Orchestrator:
    """Orchestrates all the sub-modules to action together.
    """

    def __init__(self, args: Namespace):
        self.args = args
        self.frames = []
        self.listener = Listener(args=args, frames=self.frames)

    def run(self):
        listener = Listener(args=self.args, frames=self.frames)
        listener.start()

        while True:
            if len(self.frames) < self.args.buffer_size:
                continue
            else:
                raw_input = self.frames.copy()
                self.frames.clear()
                self.process(raw_input)
            time.sleep(0.5)

    def process(raw_input: List[bytes]):
        """Process the raw_input. Predict and get the result. Then convert to text.
        """
        print('Process recognized text...')
