"""
The orchestrator that put all the speech to text sub-modules together.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
import json
from multiprocessing import Pipe, Process
from pyaudio import PyAudio, paInt16
import requests
import time
from threading import Thread
from typing import List
import warnings
import wave

from speech.speech2text.processor import Wav2Vec2Processor
from nlp.language_model import gpt_api
from speech.text2speech.orchestrator import TTSOrchestrator


# Suppress torch UserWarning
warnings.filterwarnings('ignore', category=UserWarning)


class Listener(Thread):
    """The listener to handle the stream input.
    """

    def __init__(self, args: Namespace, pipe: Pipe):
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
        self.pipe = pipe
        self.listen_flag = True

    def stop(self):
        self.listen_flag = False

    def run(self):
        """Listen forever.
        """
        print('[Start listening...]')
        count = 0
        while self.listen_flag:
            count += 1
            if count % 50 == 0:
                print('[Still listening...]')
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            self.pipe.send(data)
            time.sleep(0.01)
        print('[stopped]')


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
        self.consumer_pipe, self.producer_pipe = Pipe()
        self.listener = Listener(args=args, pipe=self.producer_pipe)
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
                try:
                    data = self.consumer_pipe.recv()
                    self.frames.append(data)
                except EOFError:
                    print('eof')
                if len(self.frames) < self.args.buffer_size:
                    # print(len(self.frames))
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
                time.sleep(0.01)
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


def create_listener(args: Namespace, producer_pipe: Pipe):
    l = Listener(args=args, pipe=producer_pipe)
    l.run()


def create_answer(answer: str):
    tts = TTSOrchestrator(text=answer, read_mode='one_shot')
    tts.process()


class ConversationOrchestrator(Orchestrator):
    """The orchestrator that process real time voice based converstaion.
    """
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.frames = []
        self.consumer_pipe, self.producer_pipe = Pipe()
        self.listener = Process(target=create_listener, args=(args, self.producer_pipe))
        self.pyaudio = PyAudio()
        self.communicator = gpt_api.OpenAICompleteCommunicator(is_conversation=True, model_type=self.args.model_type)

    def run(self):
        """Use an infinite loop to retrieve the raw input from the audio stream.
        Process the input only when there are enough frames of data.

        Args:
            acc_transcript: Used to store the accumulated transcripts. The caller can get it externally.
        """
        count = 0
        self.listener.start()
        no_input = 0
        acc_transcript = ''
        try:
            while True:
                count += 1
                try:
                    data = self.consumer_pipe.recv()
                    self.frames.append(data)
                except EOFError:
                    # if count % 10 == 0:
                    #     print('eof')
                    continue
                if len(self.frames) < self.args.buffer_size:
                    # Skip when there are too few frames.
                    # if count % 10 == 0:
                    #     print('skip')
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
                    
                    if self.args.no_input_retry != -1 and no_input == self.args.no_input_retry:
                        no_input = 0
                        # Answer when there is an input pause.
                        if len(acc_transcript) > 0:
                            # Stop the listener to prevent treat the answer voice as input.
                            self.listener.terminate()
                            # Conduct NUL.
                            copy_transcript = acc_transcript[:]
                            acc_transcript = ''
                            print(f'[No input for {self.args.no_input_retry} times, start to reply for {copy_transcript}.]')
                            self._update_frontend(initiator='human', text=copy_transcript)  # Send http request to update the question.

                            answer = self.communicator.send_requests(prompt=copy_transcript)
                            self._update_frontend(initiator='ai', text=answer)  # Send http request to update the answer.

                            # Conduct TTS.
                            tts = TTSOrchestrator(text=answer, read_mode='incremental')
                            tts.process()
                            # Restart a new listener after answering.
                            self.listener = Process(target=create_listener, args=(self.args, self.producer_pipe))
                            self.listener.start()
                        else:
                            print('No input, continue wait.')
                time.sleep(0.02)
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

    def _update_frontend(self, initiator: str, text: str):
        headers = {'Content-type': 'application/json'}
        request_data = {'initiator': initiator, 'text': text}
        response = requests.post('http://127.0.0.1:2222/update', data=json.dumps(request_data), headers=headers)
        if response.status_code != 200:
            print(f'Error: {response.status_code}')