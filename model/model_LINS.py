from model.chat_llms import chatllms
from model.retriever_model import LINS_Retriever
from model.database import LINS_Database
from model.utils import run_batch_jobs, get_retrieved_passages
from model.prompts import return_prompts
import os, torch, json
import time
import openai

prompts_dict = return_prompts()

class LINS:
    def __init__(self, LLM_name='gpt-4o', 
                 assistant_LLM_name='gpt-4o', 
                 retriever_name='text-embedding-3-large', 
                 LLM_keys=os.environ.get("OPEN_API_KEY"), 
                 Gemini_keys=os.environ.get("GEMINI_KEY"),
                 retriever_max_thread=100, 
                 retriever_api_keys=os.environ.get("OPEN_API_KEY"),
                 BGE_encoder_path='./model/retriever/bge/bge-m3',
                 database_name='pubmed',
                 local_data_path=None):
        self.model = chatllms(model_name=LLM_name, llm_keys=Gemini_keys if 'gemini' in LLM_name else LLM_keys)
        self.assistant_model = chatllms(model_name=assistant_LLM_name, llm_keys=Gemini_keys if 'gemini' in LLM_name else LLM_keys)
        self.retriever = LINS_Retriever(retriever_name, max_thread=retriever_max_thread, OPEN_API_KEY=retriever_api_keys, BGE_encoder_path=BGE_encoder_path)
        self.database = LINS_Database(database_name, retriever=self.retriever, local_data_path=local_data_path)
        self.HERD_guidelines = None
        self.HERD_pubmed = None
        self.HERD_bing = None
        
    #@torch.no_grad()
    #def chat(self, question, history=[], num_retry = 5):
    #    for i in range(num_retry):
    #        try:
    #            response, history = self.model.chat(question, history)
    #            return response, history
    #        except Exception as e:
    #            print(e)
    #            print("Retrying...")
    #    print("Failed to get response.")
    #    return None, []

    @torch.no_grad()
    def chat(self, question, history=[], num_retry=5):
        for i in range(num_retry):
            try:
                response, history = self.model.chat(question, history)
                return response, history
            except openai.APITimeoutError as e:#add error handling
                print(f"Timeout error: {e}, retrying ({i+1}/{num_retry})...")
                time.sleep(2**i)  # backoff
            except openai.AuthenticationError as e:
                print("Authentication failed. Check API key.")
                raise
            except openai.BadRequestError as e:
                print(f"Bad request: {e}")
                raise
        print("Max retries exceeded.")
        return None, []

    @torch.no_grad()
    def chat_for_evaluation(self, question="", history=None, search_results=None, retrieval_passages=None):
        for i in range(5):
            try:
                response, history = self.model.chat(question, history)
                return response, history
            except Exception as e:
                print(e)
                continue

    #@torch.no_grad()
    #def assistant_chat(self, question, history=[], num_retry = 5):
    #    for i in range(num_retry):
    #        try:
    #            response, history = self.assistant_model.chat(question, history)
    #            return response, history
    #        except Exception as e:
    #            print(e)
    #            print("Retrying...")
    #    print("Failed to get response.")
    #    return None, []

    @torch.no_grad()
    def assistant_chat(self, question, history=[], num_retry=5):
        for i in range(num_retry):
            try:
                response, history = self.assistant_model.chat(question, history)
                return response, history
            except openai.APITimeoutError as e:#add error handling
                print(f"Timeout error: {e}, retrying ({i+1}/{num_retry})...")
                time.sleep(2**i)  # backoff
            except openai.AuthenticationError as e:
                print("Authentication failed. Check API key.")
                raise
            except openai.BadRequestError as e:
                print(f"Bad request: {e}")
                raise
        print("Max retries exceeded.")
        return None, []
    
    @torch.no_grad()
    def PRM(self, question:str, refs:list[str]):#passage relevance module
        #add error handling
        if not refs:
            raise ValueError("Empty reference list provided to PRM")
        Passage_Relevance_prompt = prompts_dict["Passage_Relevance_prompt"]
        prompt = Passage_Relevance_prompt
        result = []
        for  ref in refs:
            PRM_prompt = prompt + "#QUESTION#\n" + question + "\n#PASSAGE#\n" + ref + "\n#ANSWER#\n"
            response, history = self.assistant_chat(question=PRM_prompt, history=None)
            result.append(response)
        return result

    @torch.no_grad()
    def GRM(self, question:str, refs:list[str]):#passage relevance module
        #add error handling
        if not refs:
            raise ValueError("Empty reference list provided to GRM")
        Guideline_Relevance_prompt = prompts_dict["Guideline_Relevance_prompt"]
        prompt = Guideline_Relevance_prompt
        result = []
        for  ref in refs:
            PRM_prompt = prompt + "#QUESTION#\n" + question + "\n#GUIDELINE#\n" + ref + "\n#ANSWER#\n"
            response, history = self.assistant_chat(question=PRM_prompt, history=None)
            result.append(response)
        return result
    

    @torch.no_grad()
    def SKM(self, question:str):#self knowledge module
        Self_knowledge_prompt = prompts_dict["Self_knowledge_prompt"]
        prompt = Self_knowledge_prompt + "#QUESTION#\n" + question+ "\n#ANSWER#\n"
        response, history = self.chat(question = prompt, history=None)
        return [response]
    
    @torch.no_grad()
    def QDM(self, question:str):#question decomposition module
        Question_Decomposition_prompt = prompts_dict["Question_Decomposition_prompt"]
        prompt = Question_Decomposition_prompt + "#QUESTION#\n" + question + "\n#ANSWER#\n"
        response, history = self.assistant_chat(question=prompt, history=None)
        question_list = []
        for i in response.split("\n"):
            if i:
                question_list.append(i)
        return question_list

    @torch.no_grad()
    def PCM(self, sentence:str, passage:str):
        Passage_Coherence_prompt = prompts_dict["Passage_Coherence_prompt"]
        prompt = Passage_Coherence_prompt + "#SENTENCE#\n" + sentence + "\n#PARAGRAPH#\n" + passage + "\n#PARAGRAPH#\n" + "\n#ANSWER#\n"
        response, history = self.assistant_chat(question=prompt, history=None)
        return response
    
    @torch.no_grad()
    def PICO(self, patient_information:str, clinical_question:str):
        PICO_prompt = prompts_dict["PICO_prompt"]
        prompt = PICO_prompt.format(patient_information=patient_information, clinical_question=clinical_question)
        response, history = self.assistant_chat(question=prompt, history=None)
        return response

    @torch.no_grad()
    def keyword_extraction(self, question, max_num_keywords=-1):
        keywords_prompt = prompts_dict["keywords_prompt"]
        num_keywords_prompt = prompts_dict["num_keywords_prompt"]
        if max_num_keywords <= 0:
            prompt = keywords_prompt + "# documents #:" + question + "\n# answer #:"
        else:
            key_prompt = num_keywords_prompt.replace("**number", str(max_num_keywords))
            prompt = key_prompt + "# documents #:" + question + "\n# answer #:"
        response = self.chat(question=prompt, history=None)[0].lower()
        keyowrds = "(" + response.replace(", ", ") AND (") + ")"
        return keyowrds


    @torch.no_grad()
    def KED_search(self, question, topk=50, if_split_n=False, database = None):
        for _ in range(5):
            try:
                KED_database = database if database else self.database
                data_list = KED_database.get_data_list(question=question, retmax=topk, if_split_n=if_split_n)
                if not data_list or not data_list['texts']:#先用原始问题检索，如果空了，再调用关键词提取退化算法
                    keyword = self.keyword_extraction(question)
                    #add error handling
                    print("no data found, using keyword extraction")
                    while not data_list or not data_list['texts']:
                        print("keyword:", keyword)
                        data_list = KED_database.get_data_list(question=keyword, retmax=topk, if_split_n=if_split_n)
                        #去除最后一个关键词
                        if " AND " not in keyword:
                            break
                        keyword = keyword.split(" AND ")
                        keyword.pop()
                        keyword = " AND ".join(keyword)
                    if not data_list or not data_list['texts']:
                        #add error handling
                        print("no data found after keyword fallback")
                        return None
                    else:
                        print("data found")
                        print(data_list['urls'])
                    break
                break
            except Exception as e:
                print(f"Database error: {e}")
                if "rate limit" in str(e).lower():
                    time.sleep(10)  # 处理限流
                continue
        return data_list
    
    @torch.no_grad()
    def get_passages(self, question, topk=5, if_split_n=False, recall_top_k=-1, database=None):
        pass_database = database if database else self.database
        if pass_database.database_name in ['omim', 'oncokb', 'textbooks', 'guidelines']:
            recall_top_k = topk
            data_list = self.KED_search(question=question, topk=recall_top_k, if_split_n=if_split_n, database=pass_database)
        elif pass_database.database_name in ['pubmed', 'bing']:
            if recall_top_k == -1:
                recall_top_k = 100
            data_list = self.KED_search(question=question, topk=recall_top_k, if_split_n=if_split_n, database=pass_database)
            if not data_list or not data_list['texts']:
                return None
            print("begin filtering most relevant passages")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            question_embedding = torch.tensor(self.retriever.encode(question)).to(device)
            data_list_embedding = torch.tensor(self.retriever.encode(data_list['texts'])).to(device)
            scores = torch.matmul(question_embedding, data_list_embedding.T)
            topk=min(topk, len(data_list['texts']))
            topk_indices = torch.topk(scores, topk, dim=0).indices
            data_list['scores'] = scores
            data_list['indices'] = topk_indices

            data_list['texts'] = [data_list['texts'][i] for i in topk_indices]
            data_list['titles'] = [data_list['titles'][i] for i in topk_indices]
            data_list['scores'] = [data_list['scores'][i] for i in topk_indices]
            data_list['urls'] = [data_list['urls'][i] for i in topk_indices]

            print("filtered most relevant passages")
            print(data_list['urls'])
        else:
            ValueError(f"Unsupported database name: {pass_database.database_name}")
        retrieved_passages = data_list
        return retrieved_passages


    @torch.no_grad()
    def SKA_QDA(self,
                question,
                topk=5,
                if_SKA=True,
                if_QDA=True,
                return_passages=False,
                database=None
                ):
        database = database if database else self.database
        sub_questions = []
        RAG_prompt = prompts_dict["RAG_prompt"]
        retrieved_passages = []

        if if_SKA:
            print("Self knowledge analysis...")
            SKM_result = self.SKM(question)
            if 'CERTAIN' in SKM_result:
                if return_passages:
                    return [], []
                print("Model can answer the question using its own knowledge.")
                print("question:", question)
                prompt = question
                response, history = self.chat(question=prompt, history=None)
                return response, [], [], history, []
            else:
                print("Model cannot answer the question using its own knowledge.")
        if if_QDA:
            print("Question decomposition analysis...")
            sub_questions = self.QDM(question)
            print("Sub-questions generated.")
            print(sub_questions)
            retrieved_passages = []
            urls = []
            for sub_question in sub_questions:
                print("Retrieving passages for sub-question...")
                sub_search_results = self.get_passages(question=sub_question, topk=topk, database=database)
                if sub_search_results != None and sub_search_results != []:
                    print(sub_search_results['urls'])
                    retrieved_passages.extend(sub_search_results['texts'])
                    urls.extend(sub_search_results['urls'])
        if retrieved_passages:
            if return_passages:
                return retrieved_passages, urls
            references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in range(len(retrieved_passages)))
            prompt = RAG_prompt + "\n**Retrieved information**\n" + references_str  + f"**Question**: {question}\n" + "**Answer:**"
            response, history = self.chat(question=prompt, history=None)
            return response, urls, retrieved_passages, history, sub_questions
        else:
            if return_passages:
                return []
            print("No evidence found to answer the question, use model's own knowledge.")
            prompt = question
            response, history = self.chat(question=prompt, history=None)
            return response, [], [], history, []

    @torch.no_grad()
    def MAIRAG(self, 
               question, 
               topk=5, 
               if_PRA=True, 
               if_SKA=True, 
               if_QDA=True, 
               if_PCA=False,
               recall_top_k=-1,
               if_split_n=False,
               return_passages=False,
               database=None
               ):
        database = database if database else self.database
        sub_questions = []
        RAG_prompt = prompts_dict["RAG_prompt"]
        print("Retrieving passages...")
        retrieved = self.get_passages(question, topk=topk, if_split_n=if_split_n, recall_top_k=recall_top_k, database=database)
        
        if not retrieved or not retrieved['texts']:
            print("No passage found.")
            return self.SKA_QDA(question, topk=topk, if_SKA=if_SKA, if_QDA=if_QDA, return_passages=return_passages, database=database)
            
        if if_PRA:
            print("Passage relevance analysis...")
            PRM_result = self.PRM(question, retrieved['texts'])
            if 'Gold' in PRM_result:
                print("Utility passage found.")
                Gold_index = [i for i, result in enumerate(PRM_result) if result == "Gold"]
            else:
                print("No utility passage found.")
                return self.SKA_QDA(question, topk=topk, if_SKA=if_SKA, if_QDA=if_QDA, return_passages=return_passages, database=database)
        else:
            Gold_index = [i for i in range(len(retrieved['texts']))]
        retrieved_passages = [retrieved['texts'][i] for i in Gold_index]
        urls = [retrieved['urls'][i] for i in Gold_index]
        if return_passages:
            return retrieved_passages, urls
        references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in range(len(retrieved_passages)))
        prompt = RAG_prompt + "\n**Retrieved information**\n" + references_str  + f"**Question**: {question}\n" + "**Answer:**"
        response, history = self.chat(question=prompt, history=None)
        if if_PCA:
            print("Passage coherence analysis...")
            sentence = question + " answer:" + response
            coher = self.PCM(sentence, references_str)
            if coher == 'Conflict':
                print("Conflict detected.")
                re_prompt = "The generated sentence is inconsistent with the retrieved paragraph. Please re-answer the question based on the retrieved paragraph but your own knowledge."
                print("Re-generating...")
                response, history = self.chat(question=re_prompt, history=history)
            else:
                print("Coherent.")
        return  response, urls, retrieved_passages, history, sub_questions


    @torch.no_grad()    
    def HERD_search(self, PICO_question, topk=5, if_guidelines=True, if_pubmed=True, if_bing=True):#指南最多同时用三篇
        print("HERD search...")
        retrieved_passages = []
        urls = []
        if if_guidelines:#第一级检索
            if not self.HERD_guidelines:
                self.HERD_guidelines = LINS_Database(database_name='guidelines', retriever=self.retriever)
            print("HERD_Guidelines search...")
            retrieved = self.HERD_guidelines.get_data_list(question=PICO_question, retmax=topk)
            retrieved_passages = retrieved['texts']
            urls = retrieved['urls']
            assert len(retrieved_passages) == len(urls) and len(retrieved_passages) > 0
            PRM_results = self.PRM(PICO_question, retrieved_passages)
            if "Gold" in PRM_results:
                retrieved_passages = [retrieved_passages[ix] for ix in range(len(retrieved_passages)) if PRM_results[ix] == "Gold"]
                urls = [urls[ix] for ix in range(len(urls)) if PRM_results[ix] == "Gold"]
                print("Find Gold in HERD_guidelines data.")
                print(urls)
                return retrieved_passages, urls
                #return [],[]
            else:
                print("No Gold in HERD_guidelines data.")
                retrieved_passages = []
                urls = []
                #return retrieved_passages, urls#暂时只看找到了local的情况
        if if_pubmed:  
            if not self.HERD_pubmed:
                self.HERD_pubmed = LINS_Database(database_name='pubmed', retriever=self.retriever)
            print("HERD_Pubmed search...")
            retrieved = self.get_passages(question=PICO_question, topk=topk, if_split_n=False, database=self.HERD_pubmed)
            if retrieved and retrieved['texts']:
                retrieved_passages = retrieved['texts']
                urls = retrieved['urls']
                PRM_results = self.PRM(PICO_question, retrieved_passages)
                if "Gold" in PRM_results:
                    retrieved_passages = [retrieved_passages[ix] for ix in range(len(retrieved_passages)) if PRM_results[ix] == "Gold"]
                    urls = [urls[ix] for ix in range(len(urls)) if PRM_results[ix] == "Gold"]
                    print("Find Gold in HERD_pubmed.")
                    print(urls)
                    return retrieved_passages, urls

            print("No Gold in pubmed.")
            retrieved_passages = []
            urls = []
                    #return retrieved_passages, urls
        if if_bing:
            print("HERD_Bing search...")
            if not self.HERD_bing:
                self.HERD_bing = LINS_Database(database_name='bing', retriever=self.retriever)
            retrieved = self.get_passages(question=PICO_question, topk=topk, if_split_n=False, database=self.HERD_bing)
            if retrieved and retrieved['texts']:
                retrieved_passages = retrieved['texts']
                urls = retrieved['urls']
                #PRM_results = self.PRM(PICO_question, retrieved_passages)
                #if "Gold" in PRM_results:
                #    retrieved_passages = [retrieved_passages[ix] for ix in range(len(retrieved_passages)) if PRM_results[ix] == "Gold"]
                #    urls = [urls[ix] for ix in range(len(urls)) if PRM_results[ix] == "Gold"]
                print("Find Gold in bing.")
                print(urls)
                return retrieved_passages, urls

            retrieved_passages = []
            urls = []
            print("No Gold in bing.")
        print("No Gold in HERD.")
        return retrieved_passages, urls



    @torch.no_grad()
    def AEBMP(self, patient_information, clinical_question="", PICO_question="", topk=3, if_SKM=False, if_QDA=False, if_guidelines=False):
        assert PICO_question or clinical_question
        if not PICO_question :
            print("generate PICO question...")
            PICO_question = self.PICO(patient_information, clinical_question)
            print("PICO question:", PICO_question)
        if not clinical_question:
            clinical_question = PICO_question
        retrieved_passages, urls = self.HERD_search(PICO_question=PICO_question, topk=topk, if_guidelines=if_guidelines, if_pubmed=True, if_bing=True)
        if len(retrieved_passages) > 0:
            references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in range(len(retrieved_passages)))
            AEBMP_prompt = prompts_dict["AEBMP_prompt"]
            prompt = AEBMP_prompt.format(patient_information=patient_information, clinical_question=clinical_question, PICO_question=PICO_question, retrieved_evidence=references_str)
            response, history = self.chat(question=prompt, history=None)
            return response, urls, retrieved_passages, history, PICO_question
        else:
            print("No relevant information found")
            if if_SKM:
                print("Self knowledge analysis...")
                SKM_result = self.SKM(patient_information + "\n" + clinical_question)
                if 'CERTAIN' in SKM_result:
                    print("Model can answer the question using its own knowledge.")
                    prompt = patient_information + "\n" + clinical_question
                    response, history = self.chat(question=prompt, history=None)
                    return response, [], [], history, PICO_question
                else:
                    print("Model can't answer the question using its own knowledge.")
            if if_QDA:
                print("Question decomposition analysis...")
                question_list = self.QDM(PICO_question)
                print("Original question:", PICO_question)
                print("Decomposed questions:", question_list)
                retrieved_passages = []
                urls = []
                for sub_question in question_list:
                    print("Retrieving passages for sub-question...")
                    sub_retrieved_passages, sub_urls = self.HERD_search(sub_question, topk=1, if_guidelines=if_guidelines, if_pubmed=True, if_bing=True)
                    if sub_retrieved_passages:
                        retrieved_passages.extend(sub_retrieved_passages)
                        urls.extend(sub_urls)
                if retrieved_passages:
                    references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in range(len(retrieved_passages)))
                    prompt = AEBMP_prompt.format(patient_information=patient_information, clinical_question=clinical_question, PICO_question=PICO_question, retrieved_evidence=references_str)
                    response, history = self.chat(question=prompt, history=None)
                    return response, urls, retrieved_passages, history, PICO_question
            print("No relevant information found, model help answer the question.")
            prompt = "\n#PATIENT INFORMATION#\n" + patient_information + "\n#CLINICAL QUESTION#\n" + clinical_question + "\n#ANSWER#\n"
            response, history = self.chat(question=prompt, history=None)
            return response, [], [], history, PICO_question

    

        


    @torch.no_grad()
    def Medical_Entity_Extraction(self, text:str, max_extraction_number=4):#return a dict of medical entities
        #add error handling
        try:
            Medical_Entity_Extraction_prompt = prompts_dict["Medical_Entity_Extraction_prompt"]
            prompt = Medical_Entity_Extraction_prompt.format(TEXT=text, MAX_EXTRACTION_NUMBER=max_extraction_number)
            response, history = self.assistant_chat(question=prompt, history=None)
            entities = json.loads(response)
            return entities
        except json.JSONDecodeError:
            print("Failed to parse model response as JSON")
            raise ValueError("Failed to parse model response as JSON")

    @torch.no_grad()
    def Medical_Text_Explanation(self, text:str, max_extraction_number=4, entity_list=[], database=None, topk=2, max_retries=3):
        database = database if database else self.database
        Medical_Text_Explanation_prompt = prompts_dict["Medical_Text_Explanation_prompt"]

        if entity_list==[]:
            print("Extracting medical entities...")
            #add error handling
            for _ in range(max_retries):
                try:
                    entity_dict = self.Medical_Entity_Extraction(text, max_extraction_number=max_extraction_number)
                    entity_list = entity_dict['entity_list']
                    print("Entities extracted:", entity_list)
                    break
                except Exception as e:
                    print(f"Error extracting entities: {e}")
                    print("Retrying...")
            #entity_dict = self.Medical_Entity_Extraction(text, max_extraction_number=max_extraction_number)
            #entity_list = entity_dict['entity_list']
            #print("Entities extracted:", entity_list)

        entity_urls = []
        entity_retrieved_passages = []
        entity_explanations = []
        print("Explaining medical entities...")
        for entity in entity_list:
            print(f"Searching for entity: {entity}")
            question = entity
            retrieved = self.get_passages(question, topk=topk, database=database)
            if retrieved and retrieved['texts']:
                retrieved_passages = retrieved['texts']
                urls = retrieved['urls']
                print(urls)
            else:
                retrieved_passages = []
                urls = []
                print("No relevant information found.")
            entity_retrieved_passages.append(retrieved_passages)
            entity_urls.append(urls)
            references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in range(len(retrieved_passages))) if retrieved_passages else ""
            prompt = Medical_Text_Explanation_prompt.format(TEXT=text, RETRIEVED_EVIDENCE=references_str, ENTITY=entity)
            response, history = self.chat(question=prompt, history=None)
            entity_explanations.append(response)
            print(f"Explanation for entity {entity}: {response}")
            print("Evidence sources:")
            for i in range(len(retrieved_passages)):
                print(f"[{i+1}] {urls[i]}")
        return entity_list, entity_urls, entity_retrieved_passages, entity_explanations

    @torch.no_grad()
    def Medical_Order_QA(self, patient_information:str, question:str, topk=3, database=None, if_PRA=True, if_SKA=True, if_QDA=True):
        database = database if database else self.database
        Medical_Order_Question_prompt = prompts_dict["Medical_Order_Question_prompt"]
        prompt = Medical_Order_Question_prompt.format(PATIENT_INFORMATION=patient_information, QUESTION=question)
        print("generating retrieval question...")
        retrieval_question, history = self.assistant_chat(question=prompt, history=None)
        print("retrieval question:", retrieval_question)
        retrieved_passages, urls = self.MAIRAG(question=retrieval_question, topk=topk, if_PRA=if_PRA, if_SKA=if_SKA, if_QDA=if_QDA, database=database, return_passages=True)
        
        if retrieved_passages:
            Medical_Order_QA_prompt = prompts_dict["Medical_Order_QA_prompt"]
            references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in range(len(retrieved_passages)))
            prompt = Medical_Order_QA_prompt.format(PATIENT_INFORMATION=patient_information, QUESTION=question, RETRIEVED_PASSAGES=references_str)
            response, history = self.chat(question=prompt, history=None)
            return response, retrieved_passages, urls, history
        else:
            print("No relevant information found, model help answer the question.")
            prompt = "\n#PATIENT INFORMATION#\n" + patient_information + "\n#QUESTION#\n" + question + "\n#ANSWER#\n"
            response, history = self.chat(question=prompt, history=None)
            return response, [], [], history


    
    @torch.no_grad()
    def MOEQA(self, patient_information, explain_text="", question="", max_extraction_number=4, entities=[], database=None,expla_topk=2, QA_topk=3, if_explanation=True, if_QA=True, QA_if_PRA=True, QA_if_SKA=True, QA_if_QDA=True):
        database = database if database else self.database
        assert if_explanation or if_QA
        if if_explanation:
            entity_list, entity_urls, entity_retrieved_passages, entity_explanations = self.Medical_Text_Explanation(text=explain_text, max_extraction_number=max_extraction_number, entity_list=entities, database=database, topk=expla_topk)
        if if_QA:
            QA_response, QA_retrieval_question, QA_retrieved_passages, QA_urls = self.Medical_Order_QA(patient_information, question, topk=QA_topk, database=database, if_PRA=QA_if_PRA, if_SKA=QA_if_SKA, if_QDA=QA_if_QDA)
        if not if_explanation:
            return {"QA_response":QA_response, "QA_retrieval_question":QA_retrieval_question, "QA_retrieved_passages":QA_retrieved_passages, "QA_urls":QA_urls}
        if not if_QA:
            return {"entity_list":entity_list, "entity_urls":entity_urls, "entity_retrieved_passages":entity_retrieved_passages, "entity_explanations":entity_explanations}
        return {"entity_list":entity_list, "entity_urls":entity_urls, "entity_retrieved_passages":entity_retrieved_passages, "entity_explanations":entity_explanations}, {"QA_response":QA_response, "QA_retrieval_question":QA_retrieval_question, "QA_retrieved_passages":QA_retrieved_passages, "QA_urls":QA_urls}

    @torch.no_grad()
    def MAIRAG_options(self, 
                       question, 
                       topk=5, 
                       if_pubmed=True, 
                       if_short_sentences=False, 
                       local_data_name="", 
                       itera_num=1, 
                       history = None,
                       yuzhi=0.47,
                       single_choice = True,
                       search_results = None,
                       retrieval_passages = None,
                       if_split_n = False,
                       recall_top_k = -1,
                       database = None
                       ):
        database = database or self.database
        sub_questions = []
        options_prompt = """
        You are a helpful assistant specialized in single-choice. Provide only the option index ('A', 'B', 'C', or 'D') as the answer to single-choice questions rather than the specific content of the options. Do not include any additional text or explanations. For example, don't say: "Here are the answer".
        
        """

        if history:
            response, history = self.chat(question=question, history=history)
            return response, [], [], history, sub_questions

        if itera_num > 3:
            return "None", [], [], None, sub_questions
        
        options_prompt_last = options_prompt

        
        #import pdb; pdb.set_trace()
        if retrieval_passages:
            retrieved_passages = retrieval_passages
            urls = ["None"]* len(retrieved_passages)
        elif search_results:
            retrieved_passages = search_results.get("texts", [])
            urls = search_results.get("urls", [""] * len(retrieved_passages))
        else:
            retrieved = self.get_passages(question, topk=topk, if_split_n=if_split_n, recall_top_k=recall_top_k, database=database)
            if not retrieved or not retrieved.get("texts"):
                retrieved_passages, urls = [], []
            else:
                retrieved_passages = retrieved["texts"]
                urls = retrieved["urls"]
    
        if retrieved_passages:
            PRM_result = self.PRM(question, retrieved_passages)
            if 'Gold' in PRM_result:
                Gold_index = [i for i, result in enumerate(PRM_result) if result == "Gold"]
                references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in Gold_index)
                prompt = options_prompt_last + "\n**Retrieved information**\n" + references_str + f"**Question**: {question}\n" + "**Answer:**"
                response, history = self.chat(question=prompt, history=None)
                retrieved_passages = [retrieved_passages[ix] for ix in Gold_index]
                urls = [urls[ix] for ix in Gold_index]
                #if itera_num == 1:
                #    sentence = question + " answer:" + response
                #    coher = self.PCM(sentence, references_str)
                #    if coher == 'Conflict':
                #        re_prompt = "The generated sentence is inconsistent with the retrieved paragraph. Please re-answer the question based on the retrieved paragraph but your own knowledge."
                #        response, history, token_used = self.chat(question=re_prompt, history=history)
                return response, urls, retrieved_passages, history, []

        print("Retrieved knowledge did not help answer the question. Checking if the model can answer the question using its own knowledge.")
        SKM_result = self.SKM(question)
        if 'CERTAIN' in SKM_result:
            print("Model can answer the question using its own knowledge.")
            print("question:", question)
            prompt = options_prompt_last + f"**Question**: {question}\n" + "**Answer:**"
            response, history = self.chat(question=prompt, history=None)
            return response, [], [], history, []
        else:
            print("Model cannot answer the question using its own knowledge, further iteration needed.")
            if itera_num == 2:
                print("Iteration limit reached. No further iterations.")
                return "None", [], [], None
        
            sub_questions = self.QDM(question)
            retrieved_passages = []
            urls = []
            for sub_question in sub_questions:
                #import pdb; pdb.set_trace()
                sub_refs = self.get_passages(sub_question, if_split_n=if_split_n, recall_top_k=recall_top_k, database=database)
                
                if sub_refs:
                    for i,ref in enumerate(sub_refs['texts']):
                        if sub_refs['scores'][i] > yuzhi and ref not in retrieved_passages:
                            retrieved_passages.append(ref)

                    for i,ref in enumerate(sub_refs['urls']):
                        if sub_refs['scores'][i] > yuzhi and sub_refs['texts'][i] not in retrieved_passages:
                            urls.append(ref)
            references_str = ''.join(f"[{ix+1}] {retrieved_passages[ix]} \n" for ix in range(len(retrieved_passages)))
            prompt = options_prompt_last + "\n**Retrieved information**\n" + references_str + f"**Question**: {question}\n" + "**Answer:**"
            response, history = self.chat(question=prompt, history=None)
            return response, urls, retrieved_passages, history, sub_questions
