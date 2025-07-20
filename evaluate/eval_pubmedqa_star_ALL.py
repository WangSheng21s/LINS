from arguments import get_medlinker_args
import json
from tqdm import tqdm
import argparse
import logging
import concurrent.futures
from functools import partial
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_LINS import LINS as model_LINS

pubmedqa_ori_pqal_path = "./evaluate_data/pubmedqa/data/ori_pqal.json"
pubmedqa_test_ground_truth = "./evaluate_data/pubmedqa/data/test_ground_truth.json"
search_path = "./search_results/pubmedqa/pubmed_gpt-4omini_keywords.jsonl"

pubmedqa_prompt = """
you are a helpful assistant specialized in single-choice. Provide only the option index ('A', 'B', 'C') as the answer to single-choice questions rather than the specific content of the options. Do not include any additional text or explanations. For example, don't say: "Here are the answer".

"""

answer_map = {"A": "yes", "B": "no", "C": "maybe"}

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
        #import pdb; pdb.set_trace()
        if tasks[0].get("context"):
            future_to_index = {executor.submit(run_task, 
                                               question=task['prompt'], 
                                               search_results=task['search_result'],
                                               retrieval_passages=task['retrieval_passage'],
                                               #embedding_model = task['embedding_model'],
                                               contexts=task['context']): idx for idx, task in enumerate(tasks)}
        else:
            future_to_index = {executor.submit(run_task, 
                                               question=task['prompt'], 
                                               search_results=task['search_result'],
                                               #embedding_model = task['embedding_model'],
                                               retrieval_passages=task['retrieval_passage']): idx for idx, task in enumerate(tasks)}

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



def main():
    medlinker_args = get_medlinker_args()
    num_batch = medlinker_args.num_batch
    model_name_list = medlinker_args.model_name_list
    task_list = medlinker_args.task_list
    local_data_name = medlinker_args.local_data_name

    result_save_path = "./evaluate_results/pubmedqa_star_ALL/"
    
    medlinker_args.llm_keys = os.environ.get("OPEN_API_KEY")
    medlinker_args.assistant_keys = os.environ.get("OPEN_API_KEY")
    
    for model_name in model_name_list:
        medlinker_args.llm_model_name = model_name
        LINS = model_LINS(LLM_name=model_name)

        for method in medlinker_args.method_list:
            for task_id, task_name in enumerate(task_list):
                assert task_name in ["pubmedqa", "pubmedqa_star"]
                task_path = pubmedqa_ori_pqal_path

                if not os.path.exists(result_save_path):
                    os.makedirs(result_save_path)
                result_save_path = result_save_path + f"{task_name}_"
                result_save_path = result_save_path + model_name.replace(".","") + f"_{method}_{local_data_name}_{medlinker_args.save_name}"
                if os.path.exists(result_save_path):
                    with open(result_save_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        num_lines = len(lines)
                        if num_lines:
                            line = json.loads(lines[-1])
                            total_number = line["total_number"]
                            correct_number = line["correct_number"]
                        else:
                            num_lines = 0
                            total_number = 0
                            correct_number = 0
                else:
                    num_lines = 0
                    total_number = 0
                    correct_number = 0

                if medlinker_args.search_results_path_list == []:
                    search_results = None
                else:
                    search_path = medlinker_args.search_results_path_list[task_id]
                    with open(search_path, "r", encoding="utf-8") as f:
                        search_results = f.readlines()
                if medlinker_args.retrieval_passages_path_list == []:
                    retrieval_passages = None
                else:
                    retrieval_path = medlinker_args.retrieval_passages_path_list[task_id]
                    with open(retrieval_path, "r", encoding="utf-8") as f:
                        retrieval_passages = f.readlines()

                batch_prompt = []
                batch_question = []
                batch_contexts = []
                batch_answer = []
                batch_answer_idx = []
                batch_search_results = []
                batch_retrieval_passages = []

                with open(pubmedqa_test_ground_truth, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    id_list = list(data.keys())
                    answer_dict = data
                with open(task_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    i=-1
                    for id, value in data.items():
                        if id not in id_list:
                            continue
                        i+=1
                        if i < num_lines:
                            continue
                        if i<medlinker_args.num_begin:
                            continue
                        if medlinker_args.num_end != -1 and i>=medlinker_args.num_end:
                            break
                        line = value
                        question = line["QUESTION"]
                        #search_results = line["search_results"]
                        prompt = f"**Question**: {question}"#\noptions:"
                        prompt += f"\n(A):yes\n(B):no\n(C):maybe"
                        prompt += f"\n**Answer**:\n"
                        contexts = line["CONTEXTS"]
                        answer = answer_dict[id]

                        batch_prompt.append(prompt)
                        batch_question.append(question)
                        batch_answer.append(answer)
                        batch_contexts.append(contexts)
                        if search_results:
                            batch_search_results.append(search_results[i])
                        if retrieval_passages:
                            batch_retrieval_passages.append(retrieval_passages[i])

                with open(result_save_path, "a", encoding="utf-8") as sf:
                    for i in range(0, len(batch_prompt), num_batch):
                        print(f"Processing {i/num_batch} to {len(batch_prompt)/num_batch}...")
                        prompts = batch_prompt[i:i+num_batch]
                        questions = batch_question[i:i+num_batch]
                        answers = batch_answer[i:i+num_batch]  
                        contexts = batch_contexts[i:i+num_batch]

                        if batch_search_results:
                            search_results = batch_search_results[i:i+num_batch]
                            search_results = [json.loads(result) for result in search_results]
                            search_questions = [result["QUESTION"] for result in search_results]
                            assert search_questions == questions
                        else:
                            search_results = None

                        if batch_retrieval_passages:
                            retrieval_passages = batch_retrieval_passages[i:i+num_batch]
                            retrieval_passages = [json.loads(result) for result in retrieval_passages]
                            retrieval_questions = [result["question"] for result in retrieval_passages]
                            retrieval_passages = [result["retrieved_passages"] for result in retrieval_passages]
                            assert retrieval_questions == questions
                        else:
                            retrieval_passages = None       

                        if method == "chat":
                            run_task = LINS.chat
                            if task_name == "pubmedqa":
                                context_strs = []
                                for context in contexts:
                                    context_str = ""
                                    for c in context:
                                        context_str += c + "\n"
                                    context_strs.append(context_str)
                                prompts = [pubmedqa_prompt + context_str + prompt for context_str, prompt in zip(context_strs, prompts)]
                            else:
                                prompts = [pubmedqa_prompt + prompt for prompt in prompts]
                        elif method == "Original_RAG":#search_results
                            run_task = partial(LINS.Original_RAG_option,topk=5,local_data_name="",yuzhi=0, if_pubmed=True, embedding_model="text-embedding-ada-002")
                        elif method == "MAIRAG":
                            run_task = partial(LINS.MAIRAG_options,topk=5,local_data_name="",yuzhi=0, if_pubmed=True)
                        else:
                            print(f"method {method} is not supported.")
                            exit()
                            
                        task_dict_list = []
                        for j, prompt in enumerate(prompts):
                            if search_results:
                                search_result = search_results[j]
                            else:
                                search_result = None
                            if retrieval_passages:
                                retrieval_passage = retrieval_passages[j]
                            else:
                                retrieval_passage = None
                            if task_name == "pubmedqa" and method != "chat":
                                context = contexts[j]
                                task_dict = {"prompt": prompt, "search_result": search_result, "retrieval_passage": retrieval_passage, "context": context}
                            else:
                                task_dict = {"prompt": prompt, "search_result": search_result, "retrieval_passage": retrieval_passage}

                            task_dict_list.append(task_dict)

                        true_correct_number = correct_number
                        true_total_number = total_number

                        for num_try in range(1):
                            correct_number = true_correct_number#错误之后要回溯
                            total_number = true_total_number
                            try:
                                results = run_batch_jobs(run_task=run_task,tasks=task_dict_list,max_thread=num_batch)
                                batch_save_results = []
                                for value in results:
                                    assert value is not None and len(value) == 3 if method == "chat" else len(value) == 5
                                for j, value in enumerate(results):
                                    if method == "chat":
                                        response, history, _ = value
                                        if "Final Answer" in response:
                                            response2 = response.split("Final Answer: ")[1]
                                        else:
                                            response2 = response
                                    else:
                                        response, urls, retrieved_passages, history, question_list = value
                                        if "Final Answer" in response:
                                            response2 = response.split("Final Answer: ")[1]
                                        else:
                                            response2 = response
                                    if response2 in ["A", "B", "C"]:
                                        response_idx = response2
                                    else:
                                        response_idx = 'A'
                                        for str in response2:
                                            if str in ["A", "B", "C"]:
                                                response_idx = str
                                                break
                                    model_pred_idx = answer_map[response_idx]
                                    question = questions[j]
                                    answer = answers[j]
                                    total_number += 1
                                    if model_pred_idx == answer:
                                        correct_number += 1
                                    acc = correct_number / total_number
                                    if method == "chat":
                                        model_results={"acc": acc, "model_pred_idx": model_pred_idx,
                                                    "response": response, 
                                                    "correct_number": correct_number, 
                                                    "total_number": total_number, 
                                                    "question": question, "history":history}
                                    else:
                                        model_results={"acc": acc, "model_pred_idx": model_pred_idx,
                                                       "response": response, 
                                                        "correct_number": correct_number, 
                                                        "total_number": total_number, 
                                                        "question_list": question_list,
                                                        "urls": urls, "retrieved_passages": retrieved_passages,
                                                        "question": question, "history": history}
                                    batch_save_results.append(model_results)
                                break
                            except:
                                print(f"Error in {num_try} try.")
                                #import pdb; pdb.set_trace()
                                #print(results)
                        if len(batch_save_results) == num_batch:
                            for save_line in batch_save_results:
                                sf.write(json.dumps(save_line, ensure_ascii=False) + "\n")
                                sf.flush()
                        else:
                            for num in range(num_batch-len(batch_save_results)):
                                model_results['acc'] = -1
                                model_results['retrieved_passages'] = []
                                sf.write(json.dumps(model_results, ensure_ascii=False) + "\n")
                                sf.flush()
                            


if __name__ == "__main__":
    main()      



        
        

