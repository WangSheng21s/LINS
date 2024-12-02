
import json
from tqdm import tqdm
import argparse
import logging
import concurrent.futures
import os
from functools import partial
import torch

def run_batch_jobs(run_task, tasks, max_thread):
    """
    Run a batch of tasks with cache.
    - run_task: the function to be called
    - tasks: the list of inputs for the function
    - max_thread: the number of threads to use
    """
    results = [None] * len(tasks)
    max_failures = 10
    observed_failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_thread) as executor, tqdm(total=len(tasks)) as pbar:
        # Map each future to its task index
        future_to_index = {executor.submit(run_task, task): idx for idx, task in enumerate(tasks)}

        for future in concurrent.futures.as_completed(future_to_index):
            pbar.update(1)
            idx = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result  # Store the result at the correct index
            except Exception as e:
                logging.exception("Error occurred during run_batch_jobs.")
                observed_failures += 1
                if observed_failures > max_failures:
                    raise
    return results


def get_retrieved_passages(question, database, retriever, top_k=5, if_split_n=False, recall_top_k=-1):
    """
    Retrieve the top-k passages from the database using the retriever.
    - question: the question to ask the retriever
    - database: the database of passages
    - retriever: the retriever model
    - top_k: the number of passages to retrieve
    """
    # Retrieve the top-k passages
    if database.database_name in ['omim', 'oncokb', 'textbooks', 'guidelines']:
        recall_top_k = top_k
        data_list = database.get_data_list(question=question, retmax=recall_top_k, if_split_n=if_split_n, retriever=retriever)
    elif database.database_name in ['pubmed', 'bing']:
        if recall_top_k == -1:
            recall_top_k = 100
        data_list = database.get_data_list(question=question, retmax=recall_top_k, if_split_n=if_split_n)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        question_embedding = torch.tensor(retriever.encode(question)).to(device)
        data_list_embedding = torch.tensor(retriever.encode(data_list['texts'])).to(device)
        scores = torch.matmul(question_embedding, data_list_embedding.T)
        topk_indices = torch.topk(scores, top_k, dim=0).indices
        data_list['scores'] = scores
        data_list['indices'] = topk_indices

        data_list['texts'] = [data_list['texts'][i] for i in topk_indices]
        data_list['titles'] = [data_list['titles'][i] for i in topk_indices]
        data_list['scores'] = [data_list['scores'][i] for i in topk_indices]
        data_list['urls'] = [data_list['urls'][i] for i in topk_indices]
    else:
        raise ValueError(f"Unsupported database name: {database.database_name}")
    
    retrieved_passages = data_list
    return retrieved_passages