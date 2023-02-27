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

    def __init__(self, is_conversation: bool = False, max_token: int = 100, temperature: float = 0.5, model_type: str = 'text-ada-001'):
        super().__init__()
        self.is_conversation = is_conversation
        self.max_token = max_token
        self.temperature = temperature
        available_models = [
            'text-ada-001',      # $0.0004 / 1K tokens
            'text-babbage-001',  # $0.0005 / 1K tokens
            'text-curie-001',    # $0.002 / 1K tokens
            'text-davinci-003'   # $0.02 / 1K tokens
        ]
        if model_type not in available_models:
            raise ValueError(f'Only accept models in {available_models}')
        self.model_type = model_type

    
    def send_requests(self, prompt: str) -> str:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.secret_key}',
        }
        prompt
        data = {
            'model': self.model_type,
            'prompt': prompt,
            'max_tokens': self.max_token,
        }
        response = requests.post(url='https://api.openai.com/v1/completions', headers=headers, json=data)
        if response.status_code != 200:
            print(response.json())
        json = response.json()
        answer = json['choices'][0]['text']
        print(answer)
        return answer
        

if __name__ == "__main__":
    communicator = OpenAICompleteCommunicator(is_conversation=True)
    # communicator.send_requests(prompt='What is the diameter of the earth and sun respectively?')
    communicator.send_requests(prompt='Who is the 43th president of the US?')
