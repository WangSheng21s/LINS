import json
import numpy as np
import os
import torch
from .searching import create_searcher
from .fetching import Fetcher
from .extracting import Extractor
from .filtering import ReferenceFilter
from typing import Optional, Union, List, Dict, Tuple, Iterable, Callable, Any

from Bio import Entrez, Medline
import re
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Optional

class RerankerForInference(nn.Module):
    def __init__(
            self,
            hf_model: Optional[PreTrainedModel] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def tokenize(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def forward(self, batch):
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

        hf_model.eval()
        return cls(hf_model, hf_tokenizer)

    def load_pretrained_model(self, pretrained_model_name_or_path, *model_args, **kwargs):
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def load_pretrained_tokenizer(self, pretrained_model_name_or_path, *inputs, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )



def pubmed_search(question:str, retmax=20) ->list:

    Entrez.email = "869347360@qq.com"
    keyword = question
    # retmax = 10
    # Entrez.esearch返返回一个采用"handle"的格式数据库标识的列表，这个列表可以用Entrez.read读取。db, term是必选参数
    handle = Entrez.esearch(db='pubmed', term=keyword, retmax=retmax, sort='relevance')
    # Entrez.read读取数据库列表，返回一个字典 record，该字典包含键"IdList"（表示配备文本查询的ID的列表），“Count”（所有ID的数目）
    record = Entrez.read(handle)
    pmids = record['IdList']
    # print(pmids)
    # Entrez.efetch用上面获得的ID列表或者单个ID作为参数，retmode表示被检索记录格式（text, HTML, XML）。rettype指显示记录的类型，这取决于访问的数据库。PubMed的rettype可以是abstract, citation或medline等。对于UniProt中rettype可以为fasta。retmax是返回的记录总数，上限1w。
    handle = Entrez.efetch(db='pubmed', id=pmids, rettype='medline', retmode='text', retmax=retmax)
    # Medline模块用来解析Entrez.efetch下载的记录。Medline.parse函数可以将其转换为一个列表。这个列表包含Bio.Medline.Record对象，就像一个字典。最常用的键是TI（标题，Title）、PMID、PG（页码，pages）、AB（摘要，Abstract）和AT（作者，Authors）
    medline_records = Medline.parse(handle)
    records = list(medline_records)
    #A U作者，TI题目，LR日期，TA杂志缩写，JT杂志全称，LID doi号

    literature_info = []
    for pmid, record in zip(pmids, records):
        url = "https://pubmed.ncbi.nlm.nih.gov/" + pmid
        literature_info.append({'url': url,
                                'title': record['TI'] if 'TI' in record.keys() else None,
                                'text': record['AB'] if 'AB' in record.keys() else None})

    return literature_info


#retriever_ckpt_path = 
device=None
filter_max_batch_size=400
searcher="bing"
searcher = create_searcher(searcher)
fetcher = Fetcher()
extractor = Extractor()
#filter = ReferenceFilter(retriever_ckpt_path, device, filter_max_batch_size)

question = "Can muscle wasting be reversed?"


class MedLinker_Searcher(object):#这里所谓的retrieval的双塔模型实际上只是一个模型，只不过二者分开编码，而不是一起编码，并不是两个模型
    def __init__(self, encoder, data_list=[], topk=10):
        self.encoder = encoder
        self.topk = topk
        self.url_list=[]
        self.title_list=[]
        self.text_list=[]
        for i in range(len(data_list)):
            self.url_list.append(data_list[i]['url'])
            self.title_list.append(data_list[i]['title'])
            self.text_list.append(data_list[i]['text'])
        
    def searcher(self, query):
        qvec = self.encoder.predict([query])
        scores = []
        embeds = []
        for i in range(len(self.text_list)):
            embeds.append(self.encoder.predict([self.text_list[i]]))
            #转制矩阵再求点积，针对qvec和embeds[i]求L2距离,而不是简单的点积
            #scores.append(np.linalg.norm(qvec - embeds[i]))
            scores.append(np.dot(qvec, embeds[i].T))  
        scores = np.array(scores)
        indices = np.argsort(scores.ravel())[::-1][:self.topk]
        return indices, scores[indices], [self.text_list[x] for x in indices], [self.url_list[x] for x in indices], [self.title_list[x] for x in indices]

    


class ReferenceRetiever():
    def __init__(self,
                 retriever_ckpt_path='./model/retriever/bge/bge-m3', 
                 device='cuda:0' if torch.cuda.is_available() else 'cpu', 
                 filter_max_batch_size=400, 
                 searcher="bing", 
                 reranker_model_path="./model/retriever/bge/bge-reranker-v2-m3",
                 #filter_with_different_urls=True
                ) -> None:
        self.searcher = create_searcher(searcher)
        self.fetcher = Fetcher()
        self.extractor = Extractor()
        self.device = device
        self.filter_max_batch_size = filter_max_batch_size
        self.retriever_ckpt_path = retriever_ckpt_path
        self.reranker_model_path = reranker_model_path

        self.filter = None
        self.rerank_model = None
        self.local_ref_embeddings = None
        #self.filter_with_different_urls = filter_with_different_urls
        
    
    def get_data_list(self, question, if_pubmed=True, retmax=50):
        if if_pubmed:
            print("[System] Searching from pubmed...")
            for i in range(5):
                try:
                    data_list = pubmed_search(question, retmax=retmax)
                    break
                except Exception as e:
                    print(e)
                    continue
            if len(data_list) == 0:
                print("[System] No available paragraphs. The pubmed references provide no useful information.")
                return None
            print("[System] Count of paragraphs: ", len(data_list))
            if len(data_list) == 0:
                print("[System] No available paragraphs. The references provide no useful information.")
                return None
            return data_list
        print("[System] Searching ...")
        search_results = self.searcher.search(question)
        if search_results == None:
            print("[System] No available search results. Please check your network connection.")
            return None
        urls = [result.url for result in search_results]
        titles = {result.url: result.title for result in search_results}
        print("[System] Count of available urls: ", len(urls))
        if len(urls) == 0:
            print("[System] No available urls. Please check your network connection.")
            return None
            
        print("[System] Fetching ...")
        fetch_results = self.fetcher.fetch(urls)
        cnt = sum([len(fetch_results[key]) for key in fetch_results])
        print("[System] Count of available fetch results: ", cnt)
        if cnt == 0:
            print("[System] No available fetch results. Please check playwright or your network.")
            return None
            
        print("[System] Extracting ...")
        data_list = []
        text_dict = {}#用于去重
        for url in fetch_results:
            extract_results = self.extractor.extract_by_html2text(fetch_results[url])
            for value in extract_results:
                if value in text_dict:#这里进行去重操作
                    continue
                else:
                    text_dict[value] = 1
                data_list.append({
                    "url": url,
                    "title": titles[url],
                    "text": value
                })
        print("[System] Count of paragraphs: ", len(data_list))
        if len(data_list) == 0:
            print("[System] No available paragraphs. The references provide no useful information.")
            return None
        return data_list

    def medlinker_query(self, question, data_list=[], filter_with_different_urls=True, if_pubmed=True) -> List[Dict[str, str]]:
        if len(data_list) == 0:
            data_list = self.get_data_list(question, if_pubmed=if_pubmed)
            if data_list == None:
                return None
        if self.filter == None:    
            self.filter = ReferenceFilter(self.retriever_ckpt_path, self.device, self.filter_max_batch_size)
        print("[System] Filtering ...")
        return self.filter.produce_medlinker_references(question, data_list, 50, filter_with_different_urls=filter_with_different_urls)
    

    
    @torch.no_grad()
    def get_filter(self, retriever_ckpt_path=None):
        if self.filter != None:
            return 
        if retriever_ckpt_path == None:
            retriever_ckpt_path = self.retriever_ckpt_path
        if self.filter == None:
            self.filter = ReferenceFilter(retriever_ckpt_path, self.device, self.filter_max_batch_size)
        


    @torch.no_grad()
    def get_reranker(self, reranker_model_path=None):
        if reranker_model_path == None:
            reranker_model_path = self.reranker_model_path
        if self.rerank_model == None:
            self.rerank_model = RerankerForInference.from_pretrained(reranker_model_path)   
            self.rerank_model.to(self.device)
        return self.rerank_model#, output_hidden_states=True, output_attentions=True, return_dict=True)
        #return model, tokenizer
    


    @torch.no_grad()
    def medlinker_rerank(self, rerank_model_path=None, query=None, search_results=None, filter_with_different_urls=True, local_data_name="", if_pubmed=True):
        #return indices, scores[indices], [self.text_list[x] for x in indices], [self.url_list[x] for x in indices], [self.title_list[x] for x in indices]
        ##self.ref_with_omim(query)  return indices, scores, [self.omim_ref_text[i] for i in indices]
        assert query != None or search_results != None
        if search_results == None and local_data_name == "":
            search_results = self.medlinker_query(query, filter_with_different_urls=filter_with_different_urls, if_pubmed=if_pubmed)
        if rerank_model_path == None:
            rerank_model_path = self.reranker_model_path
        assert rerank_model_path != None
        
        if local_data_name != "":
            omim_indices, omim_scores, omim_texts = self.refs_with_local(query, topk=50, local_data_name=local_data_name)
            index = omim_indices
            retrieval_scores = omim_scores
            texts = omim_texts
            urls = [f"https://www.{local_data_name}.org/entry/{i}" for i in omim_indices]
            titles = [f"{local_data_name}:{i}" for i in omim_indices]
            search_results=(index, retrieval_scores, texts, urls, titles)
            #return search_results #index retrieval_scores
        if search_results == None:
            return None
        index, retrieval_scores, texts, urls, titles = search_results        
        rk = self.get_reranker(rerank_model_path)
        #rk_bert_model = rk.hf_model.bert
        #rk_classifier_model = rk.hf_model.classifier

        rerank_scores = []
        rerank_embeds = []
        #sentences_pair = [[query, tex] for tex in texts]#*** NameError: name 'query' is not defined
        sentences_pair = []
        for tex in texts:
            assert tex != None
            sentences_pair.append([query, tex])
        inputs = rk.tokenizer(sentences_pair, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        results = rk.hf_model(**inputs, return_dict=True, output_hidden_states=True)
        scores = results.logits.view(-1, ).float()
        embeds = results.hidden_states
        scores = torch.sigmoid(scores).cpu()
        rerank_scores = scores.tolist()
        rerank_embeds = embeds[-1].cpu().tolist()

        search_results=(index, retrieval_scores,texts, urls, titles, rerank_scores, rerank_embeds)
        return search_results #index retrieval_scores    
    
    
    @torch.no_grad()
    def medlinker_compute_score(self, query, texts):
        if self.rerank_model == None:
            self.rerank_model = self.get_reranker(self.reranker_model_path)
        
        #先通过rerank_model计算embs
        rk = self.get_reranker(self.reranker_model_path)
        #rk_bert_model = rk.hf_model.bert
        #rk_classifier_model = rk.hf_model.classifier

        rerank_embeds = []
        #sentences_pair = [[query, tex] for tex in texts]#*** NameError: name 'query' is not defined
        sentences_pair = []
        for tex in texts:
            assert tex != None
            sentences_pair.append([query, tex])
        inputs = rk.tokenizer(sentences_pair, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        results = rk.hf_model(**inputs, return_dict=True, output_hidden_states=True)
        scores = results.logits.view(-1, ).float()
        scores = torch.sigmoid(scores).cpu()
        rerank_scores = scores.tolist()
        return rerank_scores
    
    @torch.no_grad()
    def medlinker_compute_score2(self, query:str, passage:list):
        if self.filter == None:    
            self.filter = ReferenceFilter(self.retriever_ckpt_path, self.device, self.filter_max_batch_size)
        query_embedding = self.filter.scorer.get_query_embeddings([query])
        document_embeddings = self.filter.scorer.get_embeddings(passage)
        return (query_embedding@document_embeddings.t()).cpu().numpy()[0]


    @torch.no_grad()
    def refs_with_local(self, question, local_data_name = "oncokb",topk=50):
        local_file_path = "./add_dataset/" + local_data_name +"/" + local_data_name + "_embedding.json"
        assert os.path.exists(local_file_path)
        if self.local_ref_embeddings is None:
            self.local_file = open(local_file_path, 'r')
            self.local_ref_embeddings = []
            self.local_ref_text = []
            for line in self.local_file:
                local_ref = json.loads(line)
                self.local_ref_embeddings.append(torch.tensor(local_ref["embedding"]))
                self.local_ref_text.append(local_ref["text"])
            self.local_ref_embeddings = torch.stack(self.local_ref_embeddings)
            #self.omim_file.close()
        if self.filter is None:
            self.get_filter()
        question_embedding = self.filter.scorer.get_query_embeddings([question])[0]
        scores = question_embedding @ self.local_ref_embeddings.t()
        #取最大的前5个
        scores, indices = scores.topk(min(topk, len(scores)))
        #取出对应的omim_ref中的text
        
        return indices, scores, [self.local_ref_text[i] for i in indices]


        
