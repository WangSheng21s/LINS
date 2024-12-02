from openai import OpenAI
from model.utils import run_batch_jobs
import torch
from FlagEmbedding import BGEM3FlagModel
import os

class Text_embedding_3_large:
    def __init__(self, max_thread=100, OPEN_API_KEY=os.environ.get("OPEN_API_KEY")):
        self.client = OpenAI(api_key=OPEN_API_KEY)
        self.max_thread = max_thread

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model='text-embedding-3-large').data[0].embedding
    
    def encode(self, text):
        if type(text) == str:
            return self.get_embedding(text)
        elif type(text) == list:
            text_embeddings = run_batch_jobs(run_task=self.get_embedding, tasks=text, max_thread=self.max_thread)
            return text_embeddings
        

class Text_embedding_3_small:
    def __init__(self, max_thread=100, OPEN_API_KEY=os.environ.get("OPEN_API_KEY")):
        self.client = OpenAI(api_key=OPEN_API_KEY)
        self.max_thread = max_thread

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model='text-embedding-3-small').data[0].embedding
    
    def encode(self, text):
        if type(text) == str:
            return self.get_embedding(text)
        elif type(text) == list:
            text_embeddings = run_batch_jobs(run_task=self.get_embedding, tasks=text, max_thread=self.max_thread)
            return text_embeddings


class Text_embedding_ada_002:
    def __init__(self, max_thread=100, OPEN_API_KEY=os.environ.get("OPEN_API_KEY")):
        self.client = OpenAI(api_key=OPEN_API_KEY)
        self.max_thread = max_thread

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model='text-embedding-ada-002').data[0].embedding
    
    def encode(self, text):
        assert type(text) == str or type(text) == list
        if type(text) == str:
            return self.get_embedding(text)
        else:
            text_embeddings = run_batch_jobs(run_task=self.get_embedding, tasks=text, max_thread=self.max_thread)
            return text_embeddings

class BGE:
    def __init__(self, max_thread=100, BGE_encoder_path='./model/retriever/bge/bge-m3'):
        #self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.encoder = BGEM3FlagModel(BGE_encoder_path, use_fp16=False)
        self.max_thread = max_thread

    def encode(self, text):
        assert type(text) == str or type(text) == list
        if type(text) == str:
            return self.encoder.encode([text], batch_size=1, max_length=8192)['dense_vecs'][0].tolist()
        else :
            return self.encoder.encode(text, batch_size=self.max_thread, max_length=8192)['dense_vecs'].tolist()
            

class LINS_Retriever:
    def __init__(self, retriever_name='text-embedding-3-large', max_thread=100, OPEN_API_KEY=os.environ.get("OPEN_API_KEY"), BGE_encoder_path='./model/retriever/bge/bge-m3'):
        if retriever_name=='BGE':
            self.retriever = BGE(max_thread=max_thread, BGE_encoder_path=BGE_encoder_path)
        elif retriever_name=='text-embedding-3-large':
            self.retriever = Text_embedding_3_large(max_thread=max_thread, OPEN_API_KEY=OPEN_API_KEY)
        elif retriever_name=='text-embedding-3-small':
            self.retriever = Text_embedding_3_small(max_thread=max_thread, OPEN_API_KEY=OPEN_API_KEY)
        elif retriever_name=='text-embedding-ada-2':
            self.retriever = Text_embedding_ada_002(max_thread=max_thread, OPEN_API_KEY=OPEN_API_KEY)
        else:
            print(f"retriever_name {retriever_name} not support, BGE, text-embedding-3-large, text-embedding-3-small, text-embedding-ada-2 is support")
    def encode(self, text):
        return self.retriever.encode(text)
    