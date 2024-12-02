import sys
import os
import argparse
import json
import tqdm
from wisup_e2m import PdfParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.retriever_model import LINS_Retriever
from model.utils import run_batch_jobs
from model.chat_llms import chatllms

# 加载已存在的摘要
def load_existing_abstracts(file_path="./add_dataset/guidelines/abstract2path.jsonl"):
    existing_paths = set()
    try:
        with open('file_path', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                existing_paths.add(data['path'])  # 使用相对路径
    except FileNotFoundError:
        pass  # 文件不存在
    return existing_paths

def pdf2txt(pdf_path):
    parser = PdfParser(engine="marker") # pdf 引擎: marker, unstructured, surya_layout
    pdf_data = parser.parse(pdf_path)
    return pdf_data.text

def pdf2texts(dir_path):
    #当前目录下所有文件夹
    pdf_dir_list = os.listdir(dir_path)
    pdf_dir_list = [os.path.join(dir_path, pdf_dir) for pdf_dir in pdf_dir_list if os.path.isdir(os.path.join(dir_path, pdf_dir))]
    print(pdf_dir_list)
    #pdf_dir_list = ["./心外科", "./骨科", "./肾内科", "./精神科"]

    run_task = pdf2txt
    tasks = []
    for pdf_dir in pdf_dir_list:
        for pdf_file in os.listdir(pdf_dir):
            if not pdf_file.endswith(".pdf"):
                continue
            pdf_path = os.path.join(pdf_dir, pdf_file)
            txt_path = pdf_path.replace(".pdf", ".txt")
            if os.path.exists(txt_path):
                continue
            #print(f"开始提取文件：{pdf_path}")
            tasks.append(pdf_path)

    max_thread = 2
    results = run_batch_jobs(run_task, tasks, max_thread)
    for id, result in enumerate(results):
        pdf_path = tasks[id]
        txt_path = pdf_path.replace(".pdf", ".txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Saved to {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for given data.')
    parser.add_argument('--retriever_name', type=str, default='text-embedding-3-large', help='The name of the retriever')
    parser.add_argument('--max_thread', type=int, default=100, help='The max_thread for retriever')
    parser.add_argument('--OPEN_API_KEY', type=str, default=os.environ.get("OPEN_API_KEY"), help='OPEN_API_KEY for openai retriever, BGE dont need')
    parser.add_argument('--BGE_encoder_path', type=str, default='./model/retriever/bge/bge-m3', help='The name of the retriever')
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='The name of the model')
    args = parser.parse_args()
    pdf2texts("./add_dataset/guidelines")#把所有PDF转为TXT

    retriever = LINS_Retriever(retriever_name=args.retriever_name, max_thread=args.max_thread, OPEN_API_KEY=args.OPEN_API_KEY, BGE_encoder_path=args.BGE_encoder_path)
    chat_model = chatllms(model_name=args.model_name, llm_keys=args.OPEN_API_KEY)

    # 查找需要处理的文件
    files_to_process = []
    existing_paths = load_existing_abstracts("./add_dataset/guidelines/abstract2path.jsonl")

    for root, dirs, files in os.walk('./add_dataset/guidelines/'):#遍历当前目录及子目录
        pdf_files = set()
        txt_files = set()
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.add(os.path.splitext(file)[0])
            elif file.lower().endswith('.txt'):
                txt_files.add(os.path.splitext(file)[0])
        common_files = pdf_files & txt_files
        for filename in common_files:
            pdf_path = os.path.join(root, filename + '.pdf')
            txt_path = os.path.join(root, filename + '.txt')
            # 将路径转换为相对路径
            rel_pdf_path = os.path.relpath(pdf_path)
            rel_txt_path = os.path.relpath(txt_path)
            # 检查是否已处理
            if rel_pdf_path not in existing_paths:
                files_to_process.append({'pdf_path': rel_pdf_path, 'txt_path': rel_txt_path})


    # 处理文件
    with open('./add_dataset/guidelines/abstract2path.jsonl', 'a', encoding='utf-8') as f_abstract, \
         open('./add_dataset/guidelines/guidelines_embedding.jsonl', 'a', encoding='utf-8') as f_embedding:

        for file in tqdm(files_to_process):
            txt_path = file['txt_path']
            pdf_path = file['pdf_path']
            # 读取 txt 文件
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
            # 构建要发送给 LLM 的消息
            message = f"""
    Please read the following text and extract a summary in the corresponding language of the article. The summary should meet the following criteria:

    1. **Core Content Extraction**: The summary should include the core information from the evidence-based guidelines, especially recommendations and conclusions related to diagnosis, treatment, prevention, and management. This helps to quickly understand the main points of the guidelines.

    2. **Clear Keywords**: The summary should highlight the keywords of medical issues, such as the relevant conditions, symptoms, treatment methods, indications, contraindications, etc. This will aid in more precise matching with subsequent medical queries.

    3. **Structured Information**: Whenever possible, organize the content of the summary according to the PICO (Patient/Problem, Intervention, Comparison, Outcome) framework, facilitating subsequent matching with PICO-based questions.

    4. **Concise but Comprehensive**: The summary should cover all key points concisely, avoiding lengthy background descriptions and focusing on information that is useful for evidence-based medical practice.

    5. **Retrievability**: Consider the summary's performance in information retrieval and similarity matching processes, using language that is easy to extract and match, while avoiding complex sentence structures and terminology.

    # Original Text #
    {txt_content}"""
            # 获取摘要
            # 获取摘要
            try:
                assistant_message, conversation_history, token_used = chat_model.chat(message)
            except Exception as e:
                print(f"处理文件 {txt_path} 时出错：{e}")
                continue
            # 写入 abstract2path.jsonl
            entry = {'abstract': assistant_message, 'path': pdf_path}
            f_abstract.write(json.dumps(entry, ensure_ascii=False) + '\n')
            # 对摘要进行编码
            embedding_result = retriever.encode(text=assistant_message)
            # 写入 guidelines_embedding.jsonl
            embedding_entry = {'text': assistant_message, 'embedding': embedding_result}
            f_embedding.write(json.dumps(embedding_entry, ensure_ascii=False) + '\n')
            # 刷新文件
            f_abstract.flush()
            f_embedding.flush()
