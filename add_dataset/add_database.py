import json
import os
import argparse
import types
import requests
import logging
import numpy as np
from tqdm import tqdm
import os
import requests

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from model.retriever_model import LINS_Retriever

def del_None(local_data_name):#删除空行和重复行
    save_path = f"./add_dataset/{local_data_name}/{local_data_name}_embedding.json"
    save_path_2 = f"./add_dataset/{local_data_name}/{local_data_name}_embedding2.json"
    assert os.path.exists(save_path), f"File {save_path} does not exist."
    del_id = []
    
    exist_texts = []
    if os.path.exists(save_path):
        with open(save_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for id, line in enumerate(lines):
                data = json.loads(line)
                if data and data.get("text",None) and data.get("embedding", None):
                    if data['text'] not in exist_texts:
                        exist_texts.append(data['text'])
                    else:
                        del_id.append(id)
                    
                else:
                    del_id.append(id)
        with open(save_path_2, "w", encoding='utf-8') as f:
            for id, line in enumerate(lines):
                if id not in del_id:
                    f.write(line) 
    

def generate_embeddings(local_data_name, retriever):
    
    data_base_path = f"./add_dataset/{local_data_name}/{local_data_name}.json"
    if not os.path.exists(data_base_path):
        data_base_path = f"./add_dataset/{local_data_name}/{local_data_name}.txt"
    save_path = f"./add_dataset/{local_data_name}/{local_data_name}_embedding.json"

    # 确保文件存在
    assert os.path.exists(data_base_path), f"File {data_base_path} does not exist."
    
    exist_texts = []
    if os.path.exists(save_path):
        with open(save_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = json.loads(line)
                if data and data.get("text",None):
                    exist_texts.append(data["text"])
                #else:

    texts_batch = []
    with open(data_base_path, "r", encoding='utf-8') as f:
        if data_base_path.endswith(".txt"):
            texts = f.readlines()#去掉空行
            for text in texts:
                text = text.strip()
                if text and text not in exist_texts:
                    texts_batch.append(text)
        else:
            data = json.load(f)
            for item in data:
                if item and item not in exist_texts:
                    texts_batch.append(item)
    
    #批量处理，每次处理100000个
    piliang = 1000
    num_epochs = len(texts_batch) // piliang + 1
    for i in range(0, len(texts_batch), piliang):
        print(f"Processing {i/piliang} to {num_epochs}...")
        texts = texts_batch[i:i+piliang]
        texts_save = []
        for text in texts:
            if text in exist_texts:
                continue
            #elif len(text) > 300:
            #    for i in range(0, len(text), 300):
            #        texts_save.append(text[i:i+300])
            #else:
            texts_save.append(text)
        texts = texts_save
        if len(texts) == 0:
            continue
        text_embedding_numpy = retriever.encode(texts)
        text_embedding_dict = []
        for text, embedding in zip(texts, text_embedding_numpy):
            text_embedding_dict.append({"text": text, "embedding": embedding})
        #text_embedding_dict = embed(texts, model_name=model_name)

        with open(save_path, "a", encoding='utf-8') as f:
            for result in (text_embedding_dict):
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
    del_None(local_data_name=local_data_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for given data.')
    parser.add_argument('--database_name', type=str, default='oncokb', help='The name of the local data directory')
    parser.add_argument('--retriever_name', type=str, default='text-embedding-3-large', help='The name of the retriever')
    parser.add_argument('--max_thread', type=int, default=100, help='The max_thread for retriever')
    parser.add_argument('--OPEN_API_KEY', type=str, default=os.environ.get("OPEN_API_KEY"), help='OPEN_API_KEY for openai retriever, BGE dont need')
    parser.add_argument('--BGE_encoder_path', type=str, default='./model/retriever/bge/bge-m3', help='The name of the retriever')
    args = parser.parse_args()

    retriever = LINS_Retriever(retriever_name=args.retriever_name, max_thread=args.max_thread, OPEN_API_KEY=args.OPEN_API_KEY, BGE_encoder_path=args.BGE_encoder_path)
    generate_embeddings(args.database_name, retriever=retriever)
    #del_None(args.database_name)
