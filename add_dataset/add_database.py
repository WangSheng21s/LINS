import json
import os
import argparse
from FlagEmbedding import BGEM3FlagModel

def generate_embeddings(local_data_name):
    data_base_path = f"./{local_data_name}/{local_data_name}.json"
    save_path = f"./{local_data_name}/{local_data_name}_embedding.json"

    # 确保文件存在
    assert os.path.exists(data_base_path), f"File {data_base_path} does not exist."

    bge_model = BGEM3FlagModel('../model/retriever/bge/bge-m3',  
                               use_fp16=True, device="cuda") # Setting use_fp16 to True speeds up computation with a slight performance degradation
    
    texts = []
    with open(data_base_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            texts.append(item)
    
    oncokb_vector2 = bge_model.encode(texts)['dense_vecs']

    with open(save_path, "w", encoding='utf-8') as f:
        for i in range(len(oncokb_vector2)):
            result = {}
            result["text"] = texts[i]
            result["embedding"] = oncokb_vector2[i].tolist()
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for given data.')
    parser.add_argument('local_data_name', type=str, help='The name of the local data directory')
    args = parser.parse_args()

    generate_embeddings(args.local_data_name)
