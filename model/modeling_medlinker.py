from .retriever.med_linker_search import ReferenceRetiever
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import re, os, torch, json


from Bio import Entrez, Medline
Entrez.email = "869347360@qq.com"

sysprompt = """

"""


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

single_choice_prompt = """

"""

MedQAprompt = """

"""

pubmedqa_prompt = """

"""
pubmedqa_prompt = """

"""


pubmed_little_shots="""

"""

MedQA_little_shots="""

"""


pubmedshots="""

"""



retrieval_prompt = """

"""

Passage_Relevance_prompt = """

"""

MedQA_Passage_Relevance_prompt = """

"""

Question_Decomposition_prompt = """

"""

oncokb_Passage_Coherence_prompt = """

"""

Passage_Coherence_prompt = """
You are an expert in determining whether there is consistency between the generated sentence and the retrieved paragraph. If there is a content conflict between the generated sentence and the retrieved paragraph, you would answer 'conflict', if the content of the two is consistent, you would answer 'coherence', and if the content of the two is not related, you would answer 'irrelevant'. Be careful to get to the point when answering, don't have any explanations or chitchat, just answer "conflict" or "coherence" or "irrelevant".

"""

Self_knowledge_prompt = """

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
            messages.extend([{"role": "user", "content": msg} for msg in history])
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
    def query(self, question, refs ="", filter_with_different_urls=True, with_omim=False, topk=5):
        references_str = ''
        if refs == "":
            refs = self.ref_retriever.medlinker_merage(query=question, filter_with_different_urls=filter_with_different_urls, topk=topk, with_omim=with_omim)
            if not refs:
                return { "references": [], "answer": "" }
            for ix, ref in enumerate(refs["texts"]):
                txt = ref
                references_str += f'[{ix+1}]: {txt}' '\\'
        else:
            for ix, ref in enumerate(refs["texts"]):
                references_str += f'[{ix+1}]: {ref}' '\\'

        #prompt = 'There is now the following question:\n ' + question + '\n The following references have been collected from the internet so far:\n ' + '[' + prompt + ']' + '\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.'
        #sub_add = "Here is an example: There have some references:\n [[1]:'Some lessons from Vietnam were clear: America fails when it attempts military action without a clear and achievable strategic goal, and when it fails to adequately understand the ideology of the enemy.', [2]:'https://www.nytimes.com/1973/04/01/archives/have-we-learned-or-only-failed-the-lessons-of-vietnam-vietnam.html', [3]:'For instance, one of the primary American history textbooks now used in the Garden Grove Unified School District includes more than 22 pages on the Vietnam War. It also asks students to respond to such questions as, “In what ways did U.S. military planners fail to understand the Vietnamese culture?” and, “The domino theory is still used by some politicians to explain global politics. Do you agree with the theory?”', [4]:'Once too controversial to be taught in American high schools, the Vietnam War has slowly slipped into the curriculum, but only in a superficial way, educators say. Although many teachers consider Vietnam crucial to understanding American foreign policy and identify it as one of the key events in U.S. history, today’s high school students learn about the Vietnam War in much the same way they study other contemporary affairs--quickly and without much depth, educators say.',[5]: '“I think it’s a shame that high school students don’t learn more about this war, because the Vietnam War is still constantly referred to in public discourse as a standard to measure proposed military intervention,” said Jerry Starr, director of the Center for Social Studies Education in Pittsburgh, Pa., who has done extensive research on how the Vietnam War is taught in American schools.']\n\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.\nQuestion：Why did the US fail in Vietnam and why is never really taught in schools?\nAnswer：The US failed in Vietnam because it attempted military action without a clear and achievable strategic goal, and failed to adequately understand the ideology of the enemy[1]. It is rarely taught in schools because the Vietnam War has slowly slipped into the curriculum, but only in a superficial way[4][5]. The primary American history textbooks now used in the Garden Grove Unified School District include some information on the Vietnam War, but it is usually not taught in depth[3]. According to Jerry Starr, director of the Center for Social Studies Education in Pittsburgh, Pa., the Vietnam War is still constantly referred to in public discourse as a standard to measure proposed military intervention, but high school students don't learn enough about the war[5].\n"
        prompt = "Answer the following question about medical as best you can. There have some references:\n" + references_str + "\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.\n" + f"Question: {question}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        pred = self.model.generate(**inputs, max_new_tokens=512, top_k=0, top_p=0.8,chat_format="raw",
                                    repetition_penalty=1.2,
                                    eos_token_id=151643,
                                    pad_token_id= 151643,
                                    stop_words_ids= [[151643]],
                                    do_sample= True,
                                    transformers_version= "4.31.0")
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        #resopnse去掉prompt
        response = response.replace(prompt, "")
        answer = response.split("\n\n")
        for i in range(len(answer)):
            if len(answer[i]) > 100:
                response = answer[i]
                break
        return { "answer": response, "references": refs}
    

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
    def keywords_query_option(self, question, refs = "", filter_with_different_urls=False, with_omim=False, topk=5, if_pubmed = True, if_merge=False, if_short_sentences = True):
        retriever_query = question.split("options")[0]
        references_str = ''
        if refs == "":
            search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
            if search_results == None or search_results == []:
                return { "references": [], "answer": "" }
            recall_search_results = self.ref_retriever.medlinker_query(question=retriever_query, data_list=search_results, filter_with_different_urls=False)
            rerank_search_results = self.ref_retriever.medlinker_rerank(query=retriever_query, search_results=recall_search_results)
            merge_search_results = self.ref_retriever.medlinker_merage(query=retriever_query, search_results=rerank_search_results, if_merge=if_merge, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
            refs = merge_search_results


            if not refs:
                return { "references": [], "answer": "" }
            
            for ix, ref in enumerate(refs["texts"]):
                txt = ref
                references_str += "[" + str(ix+1) + "] " + txt + " \n"
        else:
            for ix, ref in enumerate(refs["texts"]):
                references_str += "[" + str(ix+1) + "] " + ref + " \n"
        
        prompt = single_choice_prompt + question + retrieval_prompt + references_str
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        return { "answer": response, "references": refs}
    
    @torch.no_grad()
    def keywords_query_chat(self, question, refs = "", filter_with_different_urls=False, with_omim=False, topk=5, if_pubmed = True, if_merge=False, if_short_sentences = True):
        references_str = ''
        retriever_query = question.split("options")[0]
        if refs == "":
            search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
            if search_results == None:
                return { "references": [], "answer": "" }
            recall_search_results = self.ref_retriever.medlinker_query(question=retriever_query, data_list=search_results, filter_with_different_urls=False)
            rerank_search_results = self.ref_retriever.medlinker_rerank(query=retriever_query, search_results=recall_search_results)
            merge_search_results = self.ref_retriever.medlinker_merage(query=retriever_query, search_results=rerank_search_results, if_merge=if_merge, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
            refs = merge_search_results

            if not refs:
                return { "references": [], "answer": "" }
            for ix, ref in enumerate(refs["texts"]):
                txt = ref
                references_str += "[" + str(ix+1) + "] " + txt + " \n"
        else:
            for ix, ref in enumerate(refs["texts"]):
                references_str += "[" + str(ix+1) + "] " + ref + " \n"
        
        prompt = "MedLinker:" + "question:" + question + "\nretrieval_texts:" + references_str
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        return { "answer": response, "references": refs}

    @torch.no_grad()
    def agent_iterative_query_opt(self, question, local_data_name="", topk=5, if_pubmed=False, if_merge=False, itera_num=1, if_short_sentences=True):
        if itera_num >3:
            return "None"
        retrieved_passages = []
        retriever_query = question.split("options")[0]
        if if_pubmed:
            search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
            if search_results != None and search_results != []:
                
                recall_search_results = self.ref_retriever.medlinker_query(question=retriever_query, data_list=search_results, filter_with_different_urls=False)
                rerank_search_results = self.ref_retriever.medlinker_rerank(query=retriever_query, search_results=recall_search_results)
                merge_search_results = self.ref_retriever.medlinker_merage(query=retriever_query, search_results=rerank_search_results, if_merge=if_merge, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
                refs = merge_search_results

            
                for ix, ref in enumerate(refs["texts"]):
                    txt = ref
                    retrieved_passages.append(txt)
        else:
            refs = self.ref_retriever.medlinker_merage(query=retriever_query, filter_with_different_urls=False, topk=topk, if_pubmed=False, if_merge=False, local_data_name=local_data_name)
            if refs != None and refs != []:
                for ix, ref in enumerate(refs["texts"]):
                    txt = ref
                    retrieved_passages.append(txt)
        
        if retrieved_passages != []:
            PRM_result = self.PRM(question, retrieved_passages)#list[str]
            if 'gold' in PRM_result:
                #返回gold对应的索引
                gold_index = []
                for i in range(len(PRM_result)):
                    if PRM_result[i] == "gold":
                        gold_index.append(i)
                references_str = ''
                for ix in gold_index:
                    txt = retrieved_passages[ix]
                    references_str += "[" + str(ix+1) + "] " + txt + " \n"
                if itera_num == 1:
                    prompt = single_choice_prompt + question + retrieval_prompt + references_str
                else:
                    prompt = question + retrieval_prompt + references_str
                response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)

                #response = response.split(',')[0]
                if itera_num == 1:
                    #检查一致性
                    sentence = question + "          answer:" + response
                    coher = self.PCM(sentence, references_str)
                    if coher=='conflict':
                        re_prompt = ""
                        re_prompt += "The generated sentence is inconsistent with the retrieved paragraph. Please re-answer the question based on the retrieved paragraph but your own knowledge."
                        #print("The generated sentence is inconsistent with the retrieved paragraph. ")
                        #print("sentence:", sentence)
                        #print("passage:", references_str)
                        response, history = self.chat(tokenizer=self.tokenizer, prompt=re_prompt, history=history)
                return response
        print("检索的知识对回答问题没有帮助，下一步判断模型是否能够依靠自身知识回答问题")
        SKM_result = self.SKM(question)
        if 'CERTAIN' in SKM_result:
            print("模型自身知识可以回答问题")
            print("question:", question)
            if itera_num == 1:
                prompt = question
            else:
                prompt = single_choice_prompt + question
            response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
            refs = ""
            return response
        else:
            print("模型自身知识不能回答问题，需要进一步迭代")
            if itera_num ==2:
                print("迭代次数已经达到2次，不再迭代")
                return "None"
            sub_questions = self.QDM(question)
            sub_questions_answers = []
            for sub_question in sub_questions:
                sub_question_answer = self.agent_iterative_query_opt(sub_question, local_data_name=local_data_name, topk=topk, if_pubmed=if_pubmed, if_merge=if_merge, itera_num=itera_num+1, if_short_sentences=if_short_sentences)
                sub_questions_answers.append(sub_question_answer)
            references_str = single_choice_prompt + 'The first are some sub-questions, please refer to answering the last question.\n'
            for ix, ref in enumerate(sub_questions_answers):
                if ref != "None":
                    references_str += 'sub question: ' + sub_questions[ix] + "\n" + 'sub question answer: ' + ref + "\n"
            references_str += 'The last question is: ' + question + "\n"
            print(references_str)
            response, history = self.chat(tokenizer=self.tokenizer, prompt=references_str, history=None)
            refs = ""
            return response
        
    
    @torch.no_grad()
    def agent_iterative_query_opt_woQDM(self, question, local_data_name="", topk=5, if_pubmed=False, if_merge=False, itera_num=1, if_short_sentences=True):
        if itera_num >3:
            return "None"
        retrieved_passages = []
        retriever_query = question.split("options")[0]
        if if_pubmed:
            search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
            if search_results != None and search_results != []:
                
                recall_search_results = self.ref_retriever.medlinker_query(question=retriever_query, data_list=search_results, filter_with_different_urls=False)
                rerank_search_results = self.ref_retriever.medlinker_rerank(query=retriever_query, search_results=recall_search_results)
                merge_search_results = self.ref_retriever.medlinker_merage(query=retriever_query, search_results=rerank_search_results, if_merge=if_merge, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
                refs = merge_search_results

            
                for ix, ref in enumerate(refs["texts"]):
                    txt = ref
                    retrieved_passages.append(txt)
        else:
            refs = self.ref_retriever.medlinker_merage(query=retriever_query, filter_with_different_urls=False, topk=topk, if_pubmed=False, if_merge=False, local_data_name=local_data_name)
            if refs != None and refs != []:
                for ix, ref in enumerate(refs["texts"]):
                    txt = ref
                    retrieved_passages.append(txt)
        
        if retrieved_passages != []:
            PRM_result = self.PRM(question, retrieved_passages)#list[str]
            if 'gold' in PRM_result:
                #返回gold对应的索引
                gold_index = []
                for i in range(len(PRM_result)):
                    if PRM_result[i] == "gold":
                        gold_index.append(i)
                references_str = ''
                for ix in gold_index:
                    txt = retrieved_passages[ix]
                    references_str += "[" + str(ix+1) + "] " + txt + " \n"

                print("question:", question)
                print("passage:", references_str)
                prompt = single_choice_prompt + question + retrieval_prompt + references_str
                response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
                #response = response.split(',')[0]
                #检查一致性
                sentence = question + "          answer:" + response
                coher = self.PCM(sentence, references_str)
                if coher=='conflict':
                    re_prompt = ""
                    re_prompt += "The generated sentence is inconsistent with the retrieved paragraph. Please re-answer the question based on the retrieved paragraph but your own knowledge."
                    #print("The generated sentence is inconsistent with the retrieved paragraph. ")
                    #print("sentence:", sentence)
                    #print("passage:", references_str)
                    response, history = self.chat(tokenizer=self.tokenizer, prompt=re_prompt, history=history)
                return response
        response, history = self.chat(tokenizer=self.tokenizer, prompt=single_choice_prompt+question, history=None)
        refs = ""
        return response
        
            

    @torch.no_grad()
    def query_chat(self, question, refs ="", filter_with_different_urls=True, topk=5, if_pubmed=True, if_merge=False, if_short_sentences=False, local_data_name=""):
        references_str = ''
        if refs == "":

            if if_pubmed:
                search_results = self.keyword_search(question, topk=5, if_short_sentences=if_short_sentences)
                if search_results != None and search_results != []:

                    recall_search_results = self.ref_retriever.medlinker_query(question=question, data_list=search_results, filter_with_different_urls=False)
                    rerank_search_results = self.ref_retriever.medlinker_rerank(query=question, search_results=recall_search_results)
                    merge_search_results = self.ref_retriever.medlinker_merage(query=question, search_results=rerank_search_results, if_merge=if_merge, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
                    refs = merge_search_results


            else:
                refs = self.ref_retriever.medlinker_merage(query=question, filter_with_different_urls=False, topk=topk, if_pubmed=False, if_merge=False, local_data_name=local_data_name)
        
        if refs != None and refs != []:
            for ix, ref in enumerate(refs["texts"]):
                references_str += "[" + str(ix+1) + "] " + ref + " \n"
        #prompt = "Answer the following question about medical as best you can. There have some references:\n" + references_str + "\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.\n" + f"Question: {question}"
        #prompt = "MedLinker:" + "question:" + question + "\nretrieval_texts:" + references_str
        prompt = "Please answer the following medical question to the best of your ability. There are several references provided:\n" + references_str + "\nPlease refer to the above references and provide an effective, reasonable, comprehensive, and logical answer. The content should be as complete as possible, frequently citing data or examples from the references as evidence for the discussion. The answer should lean towards professionalism, and the corresponding reference numbers should be given in the format of [1][2] within the answer.\n" + f"Question: {question}"
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        return { "answer": response, "references": refs}

    

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
