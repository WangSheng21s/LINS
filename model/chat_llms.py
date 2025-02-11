import torch 
import ollama
import os
from openai import OpenAI
import google.generativeai as genai




class GPTs:
    def __init__(self, llm_keys: str = os.environ.get("OPEN_API_KEY"), model_name: str = "gpt-4o-mini"):
        # Initialize OpenAI client with the API key and model name

        self.client = OpenAI(api_key=llm_keys)
        self.model_name = model_name


    def chat(self, message: str, conversation_history: list = None):
        try:
            if conversation_history is None:
                conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

            # Add the user's message to the conversation history
            conversation_history.append({"role": "user", "content": message})
            # Call the OpenAI API to generate a response
            response = self.client.chat.completions.create(model=self.model_name, messages=conversation_history)

            # Get the assistant's reply from the response
            assistant_reply = response.choices[0].message.content

            # Append the assistant's reply to the conversation history
            conversation_history.append({"role": "assistant", "content": assistant_reply})

            return assistant_reply, conversation_history

        except Exception as e:
            if "context length" in str(e).lower():
                return "Error: Input exceeds model's context limit.", conversation_history
            elif "authentication" in str(e).lower():
                raise PermissionError("Invalid API key") from e
            else:
                raise e

# 定义 Gemini 类，用于调用 Gemini 模型
class Gemini:
    def __init__(self, llm_key=os.environ.get("GEMINI_KEY") , model_name: str = "gemini-1.5-pro"):
        genai.configure(api_key=llm_key)
        self.model = genai.GenerativeModel(model_name)

    def chat(self, message: str, conversation_history: list = None):
        try:
            if conversation_history is None:
                conversation_history = []
            chatone = self.model.start_chat(history=conversation_history)
            response = chatone.send_message(message)

            # Get the assistant's reply from the response
            assistant_reply = response.text

            # Append the assistant's reply to the conversation history
            conversation_history.append({"role": "model", "parts": assistant_reply})

            return assistant_reply, conversation_history

        except Exception as e:
            if "context length" in str(e).lower():
                return "Error: Input exceeds model's context limit.", conversation_history
            elif "authentication" in str(e).lower():
                raise PermissionError("Invalid API key") from e
            else:
                raise e
class chatllms:
    def __init__(self, model_name="Qwen", llm_keys=None) -> None:
        self.model_name = model_name.lower()
        self.tokenizer = None
        self.model = None

        if 'qwen' in self.model_name or 'llama' in self.model_name:
            #self._load_local_model()
            pass
        elif "gemini" in model_name:
            assert llm_keys is not None, "API key is required for Gemini."
            self.model = Gemini(llm_key=llm_keys, model_name=self.model_name)
        elif "gpt" in model_name or "o1" in model_name:
            assert llm_keys is not None, "API key is required for GPT."
            self.model = GPTs(llm_keys=llm_keys, model_name=self.model_name)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")


    @torch.no_grad()
    def chat(self, prompt="", history=None):
        if 'qwen' in self.model_name or 'llama' in self.model_name:
            return self._chat_local(prompt, history, self.model_name)
        elif "gemini" in self.model_name or "gpt" in self.model_name or "o1" in self.model_name:
            return self.model.chat(prompt, history)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _chat_local(self, prompt, history=None, model_name=None):
        try:
            if history is None:
                history = []
            history.append({"role": "user", "content": prompt})
            response = ollama.chat(model=model_name, messages=history)
            assistant_message = response['message']['content']
            history.append({"role": "assistant", "content": assistant_message})
            return assistant_message, history
        except KeyError:
            raise ValueError("Invalid response format from local model")
        except Exception as e:
            print(f"Local model error: {e}")
            raise
