from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
import httpx
from langchain.schema import (
    HumanMessage
)
import openai

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
        if model_name.split('-')[0] == 'text':
            self.model_type = 'completion'
        else:
            self.model_type = 'chat'
        pass

        self.call_kwargs = kwargs
        self.call_kwargs.pop('model_name', None)
        self.call_kwargs.pop('openai_api_key', None)
        other_kwargs = self.call_kwargs.pop("model_kwargs", {})
        self.call_kwargs.update(other_kwargs)
        if 'stop' in self.call_kwargs and isinstance(self.call_kwargs['stop'], str):
            self.call_kwargs['stop'] = [self.call_kwargs['stop']]
        
        print(args, kwargs, self.call_kwargs)
        self.model = openai.OpenAI(
            base_url="http://localhost:30000/v1/",   
        )

    def __call__(self, prompt: str):
        # print(f"> prompt({self.model_type}): {prompt}")
        print(">>> prompt begin >>>")
        print(prompt)
        print("<<< prompt end <<<")
        if self.model_type == 'completion':
            result = self.model.completions.create(
                model="default",
                prompt=prompt,
                **self.call_kwargs
            )
            text = result.choices[0].text
        else:
            result = self.model.completions.create(
                model="default",
                prompt=prompt,
                **self.call_kwargs
            )
            text = result.choices[0].text   
        print(">>> response begin >>>")
        print(text)
        print("<<< response end <<<")
        return text


# class AnyOpenAILLM:
#     def __init__(self, *args, **kwargs):
#         # Determine model type from the kwargs
#         model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
#         if model_name.split('-')[0] == 'text':
#             self.model = OpenAI(*args, **kwargs)
#             self.model_type = 'completion'
#         else:
#             self.model = ChatOpenAI(*args, **kwargs)
#             self.model_type = 'chat'
    
#     def __call__(self, prompt: str):
#         if self.model_type == 'completion':
#             return self.model(prompt)
#         else:
#             return self.model(
#                 [
#                     HumanMessage(
#                         content=prompt,
#                     )
#                 ]
#             ).content
