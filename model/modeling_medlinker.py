from .retriever.med_linker_search import ReferenceRetiever
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import re, os, torch, json


from Bio import Entrez, Medline
Entrez.email = "869347360@qq.com"


keywords_prompt =  """
# CONTEXT #
The task is to extract comma-separated keywords.

# OBJECTIVE #
Extract keywords in order of importance, without any additional commentary.

# STYLE #
Be concise and direct, avoiding any unnecessary explanations.

# TONE #
Professional and efficient.

# AUDIENCE #
Users seeking a quick and efficient extraction of keywords.

# RESPONSE #
Comma-separated keywords extracted from the document.

"""

num_keywords_prompt = """
# CONTEXT #
The task is to extract comma-separated keywords. The max number of keywords is **number.

# OBJECTIVE #
Extract no more than **number keywords in order of importance, without any additional commentary.

# STYLE #
Be concise and direct, avoiding any unnecessary explanations.

# TONE #
Professional and efficient.

# AUDIENCE #
Users seeking a quick and efficient extraction of keywords.

# RESPONSE #
Comma-separated keywords extracted from the document.
"""



RAG_prompt = """
# CONTEXT #
Refer to the following KNOWLEDGE along with your own understanding to answer the questions presented.

# OBJECTIVE #
Consider the given KNOWLEDGE carefully and provide an accurate response.

# STYLE #
Avoid phrases such as "the retrieved paragraph mentions" or "according to the provided KNOWLEDGE" in your response. Ensure the content is fully and clearly expressed for easy reading.

# TONE #
The response should be as detailed, professional, and objective as possible.

# AUDIENCE #
All users

# RESPONSE #
Incorporate as much content from the KNOWLEDGE as possible in your response, especially data or examples. For each sentence in the response,if the sentence includes content or data from the KNOWLEDGE, or examples, use a citation format "[n]" at the end of the sentence, where n indicates the example number. A sentence can have multiple citations such as "[1][2][3]", but the citations should always appear at the end of the sentence bef

#KNOWLEDGE#

"""

Passage_Relevance_prompt = """
# CONTEXT #
You need to determine if a paragraph of medical knowledge contains the answer to a given question.

# OBJECTIVE #
If the paragraph contains the answer to the question, output "Gold"; if it does not contain the answer, or although the paragraph is related to the question but cannot help to answer it, output "Relevant.

# STYLE #
Provide a direct and concise response without any further explanation or chat.

# TONE #
Neutral

# AUDIENCE #
Users seeking quick and clear verification of medical information.

# RESPONSE #
Output "Gold" or "Relevant" based on whether the knowledge contains the answer to the question.

"""


Question_Decomposition_prompt = """
# CONTEXT #
You need to break down a poorly answered question into more manageable sub-questions.

# OBJECTIVE #
List sub-questions that are easier to answer and retrieve information for.

# STYLE #
Concise and focused on breaking down the main question.

# TONE #
Analytical

# AUDIENCE #
Users seeking assistance in breaking down complex questions into manageable parts.

# RESPONSE #
Decomposed sub-questions in a clear and organized list form.

"""


Passage_Coherence_prompt = """
# CONTEXT #
You need to determine whether the generated sentence is consistent with the meaning expressed in the retrieved paragraph.

# OBJECTIVE #
Provide a straightforward assessment of the coherence between the sentence and the paragraph.

# STYLE #
Direct and focused response without any additional explanations.

# TONE #
Neutral

# AUDIENCE #
Users seeking quick evaluation of content consistency.

# RESPONSE #
Either "Conflict", "Coherence", or "Irrelevant" based on the relationship between the generated sentence and the retrieved paragraph.

"""

Self_knowledge_prompt = """
# CONTEXT #
You need to assess whether you can answer questions correctly.

# OBJECTIVE #
Provide an honest assessment of your ability to answer each question correctly.

# STYLE #
Be objective and truthful in your evaluation.

# TONE #
Neutral

# AUDIENCE #
Users seeking accurate self-assessment of your ability to answer questions.

# RESPONSE #
Either "CERTAIN" or "UNCERTAIN" based on your genuine assessment of your ability to answer each question correctly without any chat-based fluff and say nothing else. 

"""


class MedLinker:
    def __init__(self, medlinker_ckpt_path, retriever_ckpt_path, device="cuda:0", filter_max_batch_size=400, searcher_name="bing", filter_with_different_urls=True) -> None:
        self.device = device

        self.ref_retriever = ReferenceRetiever(retriever_ckpt_path, device, filter_max_batch_size, searcher_name)#, filter_with_different_urls=filter_with_different_urls)
        self.tokenizer = AutoTokenizer.from_pretrained(medlinker_ckpt_path, trust_remote_code=True)
        #self.model = AutoModelForCausalLM.from_pretrained(medlinker_ckpt_path, device_map="balanced_low_0" if torch.cuda.is_available() else "cpu",  trust_remote_code=True, bf16=True).eval()        
        #sequential
        self.model = AutoModelForCausalLM.from_pretrained(
            medlinker_ckpt_path,
            torch_dtype="auto",
            device_map = "auto"
        )
        #self.model.generation_config = GenerationConfig.from_pretrained(medlinker_ckpt_path, trust_remote_code=True)
        self.omim_ref_embeddings = None

    @torch.no_grad()
    def chat(self, tokenizer = None, prompt="", history=None,):
        if tokenizer is None:
            tokenizer = self.tokenizer
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if history is not None:
            messages.extend(history)
            #messages.extend([{"role": "user", "content": msg} for msg in history])
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(model_inputs.input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #\nassistant\n
        response = response.split("\nassistant\n")[-1]
        messages.append({"role": "assistant", "content": response})
        return response, messages

    @torch.no_grad()
    def PRM(self, question:str, refs:list[str], task='MedQA'):#passage relevance module
        prompt = Passage_Relevance_prompt
        if task == 'MedQA':
            prompt = MedQA_Passage_Relevance_prompt
        result = []
        for  ref in refs:
            PRM_prompt = prompt + question + "\npassage:" + ref + "\nanswer:"
            response, history = self.chat(tokenizer=self.tokenizer, prompt=PRM_prompt, history=None)
            result.append(response)
        return result
    

    @torch.no_grad()
    def SKM(self, question:str):#self knowledge module
        prompt = Self_knowledge_prompt + question
        response, history = self.chat(tokenizer=self.tokenizer, prompt = prompt, history=None)
        return [response]
    
    @torch.no_grad()
    def QDM(self, question:str):#question decomposition module
        prompt = Question_Decomposition_prompt + question + "\nanswer:"
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        question_list = []
        for i in response.split("\n"):
            if i:
                question_list.append(i)
        return question_list

    def PCM(self, sentence:str, passage:str):
        prompt = Passage_Coherence_prompt + "sentence: " + sentence + "\npassage: " + passage + "\nanswer:"
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        return response
    

    @torch.no_grad()
    def keyword_extraction(self, question, max_num_keywords=-1):
        if max_num_keywords <= 0:
            prompt = keywords_prompt + "# documents #:" + question + "\n# answer #:"
        else:
            key_prompt = num_keywords_prompt.replace("**number", str(max_num_keywords))
            prompt = key_prompt + "# documents #:" + question + "\n# answer #:"
        response = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)[0].lower()
        keyowrds = "(" + response.replace(", ", ") AND (") + ")"
        return keyowrds
    
    @torch.no_grad()
    def keyword_search(self, question, topk=50, if_short_sentences = True):
        for i in range(5):
            try:
                keyword = self.keyword_extraction(question)
                handle = Entrez.esearch(db='pubmed', term=keyword, retmax=topk, sort='relevance')
                record = Entrez.read(handle)
                pmids = record['IdList']
                while pmids == []:
                    if " AND " not in keyword:
                        break
                    #(Landolt C) AND ( Snellen e) AND ( acuity)减少一个检索词，变成(Landolt C) AND (Snellen e)
                    keyword = keyword.split(" AND ")
                    keyword.pop()
                    keyword = " AND ".join(keyword)
                    handle = Entrez.esearch(db='pubmed', term=keyword, retmax=topk, sort='relevance')
                    record = Entrez.read(handle)
                    pmids = record['IdList']
                handle = Entrez.efetch(db='pubmed', id=pmids, rettype='medline', retmode='text', retmax=topk)
                # Medline模块用来解析Entrez.efetch下载的记录。Medline.parse函数可以将其转换为一个列表。这个列表包含Bio.Medline.Record对象，就像一个字典。最常用的键是TI（标题，Title）、PMID、PG（页码，pages）、AB（摘要，Abstract）和AT（作者，Authors）
                medline_records = Medline.parse(handle)
                records = list(medline_records)
                #A U作者，TI题目，LR日期，TA杂志缩写，JT杂志全称，LID doi号
                literature_info = []
                for pmid, record in zip(pmids, records):
                    if 'TI' not in record.keys() or 'AB' not in record.keys():
                        continue
                    tex = record['AB']
                    url = "https://pubmed.ncbi.nlm.nih.gov/" + pmid
                    title = record['TI'] 
                    if if_short_sentences:
                        sentences = re.split(r'[.!?]', tex)
                        for sentence in sentences:
                            if sentence:
                                literature_info.append({'url': url,'title': title, 'text': sentence})
                    else:
                        literature_info.append({'url': url,'title': title, 'text': tex})
                break
            except:
                continue
        return literature_info

        

    @torch.no_grad()
    def Original_RAG(self, question, filter_with_different_urls=False, topk=5, if_pubmed=True, if_short_sentences=False, local_data_name=""):
        references_str = ''
        if if_pubmed:
            search_results = self.keyword_search(question, topk=5, if_short_sentences=if_short_sentences)
            if search_results != None and search_results != []:
                recall_search_results = self.ref_retriever.medlinker_query(question=question, data_list=search_results, filter_with_different_urls=filter_with_different_urls)
                rerank_search_results = self.ref_retriever.medlinker_rerank(query=question, search_results=recall_search_results)
                merge_search_results = self.ref_retriever.medlinker_merage(query=question, search_results=rerank_search_results, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=filter_with_different_urls)
                refs = merge_search_results
        else:
            refs = self.ref_retriever.medlinker_merage(query=question, filter_with_different_urls=False, topk=topk, if_pubmed=False, local_data_name=local_data_name)
        if refs != None and refs != []:
            for ix, ref in enumerate(refs["texts"]):
                references_str += "[" + str(ix+1) + "] " + ref + " \n"

        prompt = RAG_prompt + references_str + "\n#QUESTION#" + f"{question}"
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        return { "answer": response, "references": refs}



    @torch.no_grad()
    def MAIRAG(self, question, filter_with_different_urls=False, topk=5, if_pubmed=True, if_short_sentences=False, local_data_name="", itera_num=1):
        if itera_num > 3:
            return "None"
        urls = []
    
        retrieved_passages = []
        retriever_query = question
    
        if if_pubmed:
            search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
            if search_results:
                recall_search_results = self.ref_retriever.medlinker_query(
                    question=retriever_query, data_list=search_results, filter_with_different_urls=False)
                rerank_search_results = self.ref_retriever.medlinker_rerank(
                    query=retriever_query, search_results=recall_search_results)
                refs = self.ref_retriever.medlinker_merge(
                    query=retriever_query, search_results=rerank_search_results, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
        else:
            refs = self.ref_retriever.medlinker_merge(
                query=retriever_query, filter_with_different_urls=False, topk=topk, if_pubmed=False, local_data_name=local_data_name)
        if refs:
            for ref in refs:
                retrieved_passages.append(ref['texts'])
                urls.append(ref['urls'])
    
        if retrieved_passages:
            PRM_result = self.PRM(question, retrieved_passages)
            if 'Gold' in PRM_result:
                Gold_index = [i for i, result in enumerate(PRM_result) if result == "Gold"]
                references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in Gold_index)
                prompt = RAG_prompt + references_str + "\n#QUESTION#" + f"{question}"
                response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
            
                if itera_num == 1:
                    sentence = question + " answer:" + response
                    coher = self.PCM(sentence, references_str)
                    if coher == 'Conflict':
                        re_prompt = "The generated sentence is inconsistent with the retrieved paragraph. Please re-answer the question based on the retrieved paragraph but your own knowledge."
                        response, history = self.chat(tokenizer=self.tokenizer, prompt=re_prompt, history=history)
                return response, urls, retrieved_passages

        print("Retrieved knowledge did not help answer the question. Checking if the model can answer the question using its own knowledge.")
        SKM_result = self.SKM(question)
        if 'CERTAIN' in SKM_result:
            print("Model can answer the question using its own knowledge.")
            print("question:", question)
            prompt = question
            response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
            return response, [], []
        else:
            print("Model cannot answer the question using its own knowledge, further iteration needed.")
            if itera_num == 2:
                print("Iteration limit reached. No further iterations.")
                return "None"
        
            sub_questions = self.QDM(question)
            sub_questions_answers = []
            for sub_question in sub_questions:
                sub_question_answer, sub_urls, sub_texts = self.agent_iterative_query_opt(sub_question, local_data_name=local_data_name, topk=topk, if_pubmed=if_pubmed, itera_num=itera_num+1, if_short_sentences=if_short_sentences)
                sub_questions_answers.append(sub_question_answer)
                urls.extend(sub_urls)
                retrieved_passages.extend(sub_texts)
            references_str = single_choice_prompt + 'The first are some sub-questions, please refer to answering the last question.\n'
            for ix, ref in enumerate(sub_questions_answers):
                if ref != "None":
                    references_str += 'sub question: ' + sub_questions[ix] + "\n" + 'sub question answer: ' + ref + "\n"
            references_str += 'The last question is: ' + question + "\n"
            response, history = self.chat(tokenizer=self.tokenizer, prompt=references_str, history=None)
            refs = ""
            return response, urls, retrieved_passages


    

def load_model(args):
    medlinker_ckpt_path = args.medlinker_ckpt_path or os.getenv("MEDLINKER_CKPT") 
    retiever_ckpt_path = args.retriever_ckpt_path 
    if not retiever_ckpt_path:
        print('Retriever checkpoint not specified, please specify it with --retriever_ckpt_path ')
        exit(1)
    if args.serpapi_key:
        os.environ["SERPAPI_KEY"] = args.serpapi_key
    
    print('MedLinker Initializing...')
    
    medlinker = MedLinker(medlinker_ckpt_path, retiever_ckpt_path, args.device, args.filter_max_batch_size, args.searcher)
    
    print('MedLinker Loaded')
    
    return medlinker
