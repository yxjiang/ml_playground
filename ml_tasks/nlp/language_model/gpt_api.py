from abc import ABC, abstractmethod
import os
import openai
import requests


class OpenAICommunicator(ABC):
    def __init__(self):
        localtion = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.secret_key = None
        with open(os.path.join(localtion, "openai.secrets")) as f:
            self.secret_key = f.readline()
        if not self.secret_key:
            raise ValueError(f'Secrete is null.')

    def list_model(self):
        return openai.Model.list()

    @abstractmethod
    def send_requests(self):
        pass


class OpenAICompleteCommunicator(OpenAICommunicator):

    def __init__(self, is_conversation: bool = False, max_token: int = 50, temperature: float = 0.5):
        super().__init__()
        self.is_conversation = is_conversation
        self.max_token = max_token
        self.temperature = temperature
    
    def send_requests(self, prompt: str) -> str:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.secret_key}',
        }
        prompt
        data = {
            'model': 'text-babbage-001',
            'prompt': prompt,
            'max_tokens': self.max_token,
        }
        response = requests.post(url='https://api.openai.com/v1/completions', headers=headers, json=data)
        if response.status_code != 200:
            print(response.json())
        json = response.json()
        answer = json['choices'][0]['text']
        print(answer)
        

if __name__ == "__main__":
    communicator = OpenAICompleteCommunicator(is_conversation=True)
    # communicator.send_requests(prompt='What is the diameter of the earth and sun respectively?')
    communicator.send_requests(prompt='Who is the 43th president of the US?')
