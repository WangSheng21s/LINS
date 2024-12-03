from Bio import Entrez, Medline
from model.searching import create_searcher
from model.fetching import Fetcher
from model.extracting import Extractor
from model.filtering import ReferenceFilter
import os, json, torch

class Pubmed:
    def __init__(self, email="869347360@qq.com"):
        self.email = email
    def get_data_list(self, question, retmax=20, if_split_n=False, if_get_urls=False):
        Entrez.email = self.email
        keyword = question
        # retmax = 10
        # Entrez.esearch返返回一个采用"handle"的格式数据库标识的列表，这个列表可以用Entrez.read读取。db, term是必选参数
        handle = Entrez.esearch(db='pubmed', term=keyword, retmax=retmax, sort='relevance')
        # Entrez.read读取数据库列表，返回一个字典 record，该字典包含键"IdList"（表示配备文本查询的ID的列表），“Count”（所有ID的数目）
        record = Entrez.read(handle)
        pmids = record['IdList']
        if if_get_urls:
            urls = []
            for pmid in pmids:
                urls.append("https://pubmed.ncbi.nlm.nih.gov/" + pmid)
            return urls
        # print(pmids)
        # Entrez.efetch用上面获得的ID列表或者单个ID作为参数，retmode表示被检索记录格式（text, HTML, XML）。rettype指显示记录的类型，这取决于访问的数据库。PubMed的rettype可以是abstract, citation或medline等。对于UniProt中rettype可以为fasta。retmax是返回的记录总数，上限1w。
        handle = Entrez.efetch(db='pubmed', id=pmids, rettype='medline', retmode='text', retmax=retmax)
        # Medline模块用来解析Entrez.efetch下载的记录。Medline.parse函数可以将其转换为一个列表。这个列表包含Bio.Medline.Record对象，就像一个字典。最常用的键是TI（标题，Title）、PMID、PG（页码，pages）、AB（摘要，Abstract）和AT（作者，Authors）
        medline_records = Medline.parse(handle)
        records = list(medline_records)
        #A U作者，TI题目，LR日期，TA杂志缩写，JT杂志全称，LID doi号
        
        results = {"index":[], "scores":[], "texts":[], "urls":[], "titles":[]}
        #literature_info = []
        for pmid, record in zip(pmids, records):
            url = "https://pubmed.ncbi.nlm.nih.gov/" + pmid
            if record.get('AB') is None or record["AB"] == "":
                continue
            title = record['TI'] if 'TI' in record.keys() else None
            text = record['AB'] if 'AB' in record.keys() else None
            if if_split_n:
                texts = text.split('\n')
                for text in texts:
                    #literature_info.append({'url': url, 'title': title, 'text': text})
                    results["texts"].append(text)
                    results["urls"].append(url)
                    results["titles"].append(title)
            else:
                #literature_info.append({'url': url, 'title': title, 'text': text})
                results["texts"].append(text)
                results["urls"].append(url)
                results["titles"].append(title)

        return results

class Bing:
    def __init__(self):
        self.searcher = create_searcher('bing')
        self.fetcher = Fetcher()
        self.extractor = Extractor()

    def get_data_list(self, question, retmax=20, if_split_n=False, if_get_urls=False):
        #import pdb;pdb.set_trace()
        search_results = self.searcher.search(question)
        if search_results == None:
            print("[System] No available search results. Please check your network connection.")
            return None
        urls = [result.url for result in search_results]
        urls = urls[:retmax]
        titles = {result.url:result.title for result in search_results}
        if len(urls) == 0:
            print("[System] No available urls. Please check your network connection.")
            return None
        if if_get_urls:#如果只要链接
            return urls                 
        fetch_results = self.fetcher.fetch(urls)
        cnt = sum([len(fetch_results[key]) for key in fetch_results])
        if cnt == 0:
            print("[System] No available fetch results. Please check playwright or your network.")
            return None
            
        #data_list = []
        results = {"index":[], "scores":[], "texts":[], "urls":[], "titles":[]}
        text_dict = {}#用于去重
        for url in fetch_results:
            extract_results = self.extractor.extract_by_html2text(fetch_results[url], if_split_n=if_split_n)
            for value in extract_results:
                if value in text_dict or value=="":#这里进行去重操作
                    continue
                else:
                    text_dict[value] = 1
                #data_list.append({"url": url,"title": titles[url],"text": value})
                results["texts"].append(value)
                results["urls"].append(url)
                results["titles"].append(titles[url])
        if len(results['texts']) == 0:
            print("[System] No available paragraphs. The references provide no useful information.")
            return None
        return results


class OMIM:
    def __init__(self, retriever, local_path=""):
        self.retriever = retriever
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if local_path == "":
            self.local_file_path = "./add_dataset/" + "omim" +"/" + "omim_embedding.json"
        else:
            self.local_file_path = local_path + "omim_embedding.json"
        self.local_ref_embeddings = None
        self.lical_ref_text = None
    def get_data_list(self, question, question_embedding=None, retmax=50):
        if question_embedding==None:
            assert self.retriever!=None 
        local_file_path = self.local_file_path
        assert os.path.exists(local_file_path)

        if self.local_ref_embeddings is None:
            self.local_file = open(local_file_path, 'r', encoding='utf-8')
            self.local_ref_embeddings = []
            self.local_ref_text = []
            for line in self.local_file:
                local_ref = json.loads(line)
                self.local_ref_embeddings.append(torch.tensor(local_ref["embedding"]))
                self.local_ref_text.append({'text':local_ref["text"], 'url':f"local/omim", 'title':local_ref["text"][:20]})
            self.local_ref_embeddings = torch.stack(self.local_ref_embeddings)
            self.local_ref_embeddings = self.local_ref_embeddings.to(self.device)
        
        data_list_embedding = self.local_ref_embeddings
        data_list = self.local_ref_text
        if question_embedding == None:
            assert question != ""
            question_embedding = self.retriever.encode(text=question)
        assert data_list_embedding != None
        questions_embeddings = torch.tensor(question_embedding).to(self.device)
        if type(data_list_embedding) != torch.Tensor:
            passages_embeddings = torch.tensor(data_list_embedding).to(self.device)
        else:
            passages_embeddings = data_list_embedding.to(self.device)
        scores = torch.matmul(questions_embeddings, passages_embeddings.T)
        topk = retmax
        topk = min(topk, len(scores))
        topk_indices = torch.topk(scores, topk, dim=0).indices
        results = {"index":[], "scores":[], "texts":[], "urls":[], "titles":[]}
        for i in range(len(topk_indices)):
            index = topk_indices[i]
            score = scores[index]
            recall_data = data_list[index]
            results["index"].append(index)
            results["scores"].append(score)
            results["texts"].append(recall_data['text'])
            results["urls"].append(recall_data['url']+f"/entry/{index}")
            results["titles"].append(recall_data['title'])
        return results


class OncoKB:
    def __init__(self, retriever, local_path=""):
        self.retriever = retriever
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if local_path == "":
            self.local_file_path = "./add_dataset/" + "oncokb" +"/" + "oncokb_embedding.json"
        else:
            self.local_file_path = local_path + "oncokb_embedding.json"
        self.local_ref_embeddings = None
        self.lical_ref_text = None
    def get_data_list(self, question, question_embedding=None, retmax=50):
        if question_embedding==None:
            assert self.retriever!=None 
        local_file_path = self.local_file_path
        assert os.path.exists(local_file_path)

        if self.local_ref_embeddings is None:
            self.local_file = open(local_file_path, 'r', encoding='utf-8')
            self.local_ref_embeddings = []
            self.local_ref_text = []
            for line_id, line in enumerate(self.local_file):
                local_ref = json.loads(line)
                self.local_ref_embeddings.append(torch.tensor(local_ref["embedding"]))
                self.local_ref_text.append({'text':local_ref["text"], 'url':f"local/oncokb", 'title':local_ref["text"][:20]})
            self.local_ref_embeddings = torch.stack(self.local_ref_embeddings)
            self.local_ref_embeddings = self.local_ref_embeddings.to(self.device)
        
        data_list_embedding = self.local_ref_embeddings
        data_list = self.local_ref_text
        if question_embedding == None:
            assert question != ""
            question_embedding = self.retriever.encode(text=question)
        assert data_list_embedding != None
        questions_embeddings = torch.tensor(question_embedding).to(self.device)
        if type(data_list_embedding) != torch.Tensor:
            passages_embeddings = torch.tensor(data_list_embedding).to(self.device)
        else:
            passages_embeddings = data_list_embedding.to(self.device)
        scores = torch.matmul(questions_embeddings, passages_embeddings.T)
        topk = retmax
        topk = min(topk, len(scores))
        topk_indices = torch.topk(scores, topk, dim=0).indices
        results = {"index":[], "scores":[], "texts":[], "urls":[], "titles":[]}
        for i in range(len(topk_indices)):
            index = topk_indices[i]
            score = scores[index]
            recall_data = data_list[index]
            results["index"].append(index)
            results["scores"].append(score)
            results["texts"].append(recall_data['text'])
            results["urls"].append(recall_data['url']+f"/entry/{index}")
            results["titles"].append(recall_data['title'])
        return results

class Textbooks:
    def __init__(self, retriever, local_path=""):
        self.retriever = retriever
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if local_path == "":
            self.local_file_path = "./add_dataset/" + "textbooks" +"/" + "textbooks_embedding.json"
        else:
            self.local_file_path = local_path + "textbooks_embedding.json"
        self.local_ref_embeddings = None
        self.lical_ref_text = None
    def get_data_list(self, question, question_embedding=None, retmax=50):
        if question_embedding==None:
            assert self.retriever!=None 
        local_file_path = self.local_file_path
        assert os.path.exists(local_file_path)

        if self.local_ref_embeddings is None:
            self.local_file = open(local_file_path, 'r', encoding='utf-8')
            self.local_ref_embeddings = []
            self.local_ref_text = []
            for line_id, line in enumerate(self.local_file):
                local_ref = json.loads(line)
                self.local_ref_embeddings.append(torch.tensor(local_ref["embedding"]))
                self.local_ref_text.append({'text':local_ref["text"], 'url':f"local/textbooks", 'title':local_ref["text"][:20]})
            self.local_ref_embeddings = torch.stack(self.local_ref_embeddings)
            self.local_ref_embeddings = self.local_ref_embeddings.to(self.device)
        
        data_list_embedding = self.local_ref_embeddings
        data_list = self.local_ref_text
        if question_embedding == None:
            assert question != ""
            question_embedding = self.retriever.encode(text=question)
        assert data_list_embedding != None
        questions_embeddings = torch.tensor(question_embedding).to(self.device)
        if type(data_list_embedding) != torch.Tensor:
            passages_embeddings = torch.tensor(data_list_embedding).to(self.device)
        else:
            passages_embeddings = data_list_embedding.to(self.device)
        scores = torch.matmul(questions_embeddings, passages_embeddings.T)
        topk = retmax
        topk = min(topk, len(scores))
        topk_indices = torch.topk(scores, topk, dim=0).indices
        results = {"index":[], "scores":[], "texts":[], "urls":[], "titles":[]}
        for i in range(len(topk_indices)):
            index = topk_indices[i]
            score = scores[index]
            recall_data = data_list[index]
            results["index"].append(index)
            results["scores"].append(score)
            results["texts"].append(recall_data['text'])
            results["urls"].append(recall_data['url']+f"/entry/{index}")
            results["titles"].append(recall_data['title'])
        return results

class General_Local_Database:
    def __init__(self, database_name, retriever=None, local_path=""):
        self.retriever = retriever
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if not local_path :
            self.local_file_path = "./add_dataset/" + database_name +"/" + f"{database_name}_embedding.json"
        else:
            self.local_file_path = local_path + f"{database_name}_embedding.json"
        self.local_ref_embeddings = None
        self.lical_ref_text = None
        self.database_name = database_name

        if self.local_ref_embeddings is None:
            self.local_file = open(self.local_file_path, 'r', encoding='utf-8')
            self.local_ref_embeddings = []
            self.local_ref_text = []
            for line_id, line in enumerate(self.local_file):
                local_ref = json.loads(line)
                self.local_ref_embeddings.append(torch.tensor(local_ref["embedding"]))
                self.local_ref_text.append({'text':local_ref["text"], 'url':f"local/{self.database_name}", 'title':local_ref["text"][:20]})
            self.local_ref_embeddings = torch.stack(self.local_ref_embeddings)
            self.local_ref_embeddings = self.local_ref_embeddings.to(self.device)

    def get_data_list(self, question, question_embedding=None, retmax=50):
        if question_embedding==None:
            assert self.retriever!=None 
        local_file_path = self.local_file_path
        assert os.path.exists(local_file_path)
        
        data_list_embedding = self.local_ref_embeddings
        data_list = self.local_ref_text
        if not question_embedding:
            assert question != ""
            question_embedding = self.retriever.encode(text=question)
        assert data_list_embedding != None
        questions_embeddings = torch.tensor(question_embedding).to(self.device)
        if type(data_list_embedding) != torch.Tensor:
            passages_embeddings = torch.tensor(data_list_embedding).to(self.device)
        else:
            passages_embeddings = data_list_embedding.to(self.device)
        scores = torch.matmul(questions_embeddings, passages_embeddings.T)
        topk = retmax
        topk = min(topk, len(scores))
        topk_indices = torch.topk(scores, topk, dim=0).indices
        results = {"index":[], "scores":[], "texts":[], "urls":[], "titles":[]}
        for i in range(len(topk_indices)):
            index = topk_indices[i]
            score = scores[index]
            recall_data = data_list[index]
            results["index"].append(index)
            results["scores"].append(score)
            results["texts"].append(recall_data['text'])
            results["urls"].append(recall_data['url']+f"/entry/{index}")
            results["titles"].append(recall_data['title'])
        return results


class Guidelines:
    def __init__(self, retriever, local_path="",):
        self.guidelines_file_map = {}
        if not local_path:
            local_path = "./add_dataset/guidelines/"
        map_path = local_path + "abstract2path.jsonl"
        self.local_file_path = local_path + "guidelines_embedding.json"
        self.retriever = retriever
        self.local_ref_embeddings = None
        self.lical_ref_text = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        with open(map_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                self.guidelines_file_map[line['abstract']] = line['path']

        if self.local_ref_embeddings is None:
            self.local_file = open(self.local_file_path, 'r', encoding='utf-8')
            self.local_ref_embeddings = []
            self.local_ref_text = []
            for line_id, line in enumerate(self.local_file):
                local_ref = json.loads(line)
                self.local_ref_embeddings.append(torch.tensor(local_ref["embedding"]))
                self.local_ref_text.append({'text':local_ref["text"], 'url':f"local/", 'title':local_ref["text"][:20]})
            self.local_ref_embeddings = torch.stack(self.local_ref_embeddings)
            self.local_ref_embeddings = self.local_ref_embeddings.to(self.device)

    def get_data_list(self, question, question_embedding=None, retmax=5):
        if question_embedding==None:
            assert self.retriever!=None 
        data_list_embedding = self.local_ref_embeddings
        data_list = self.local_ref_text
        if not question_embedding:
            assert question != ""
            question_embedding = self.retriever.encode(text=question)
        assert data_list_embedding != None
        questions_embeddings = torch.tensor(question_embedding).to(self.device)
        if type(data_list_embedding) != torch.Tensor:
            passages_embeddings = torch.tensor(data_list_embedding).to(self.device)
        else:
            passages_embeddings = data_list_embedding.to(self.device)
        scores = torch.matmul(questions_embeddings, passages_embeddings.T)
        topk = retmax
        topk = min(topk, len(scores))
        topk_indices = torch.topk(scores, topk, dim=0).indices
        results = {"index":[], "scores":[], "texts":[], "urls":[], "titles":[]}
        for i in range(len(topk_indices)):
            index = topk_indices[i]
            score = scores[index]
            recall_data = data_list[index]
            abstract = recall_data['text']
            path = self.guidelines_file_map[abstract]
            guideline_path =  "./add_dataset/guidelines/" + path.replace(".pdf", ".txt")
            assert os.path.exists(guideline_path)
            with open(guideline_path, "r", encoding='utf-8') as f:
                guideline_text = f.read()             
            url = path
            title = url.split('/')[-1]
            results["texts"].append(guideline_text)
            results["urls"].append(url)
            results["titles"].append(title)
        return results

class Local_database:
    def __init__(self, local_database_name, retriever=None, local_path=""):
        if local_database_name=="guidelines":
            self.database = Guidelines(retriever=retriever, local_path=local_path)
        else:
            self.database = General_Local_Database(database_name=local_database_name, retriever=retriever, local_path=local_path)
    def get_data_list(self, question, question_embedding, retmax=20):
        return self.database.get_data_list(question=question, question_embedding=question_embedding, retmax=retmax)

class Online_database:
    def __init__(self, online_database_name):
        if online_database_name=='bing':
            self.database = Bing()
        elif online_database_name=='pubmed':
            self.database = Pubmed()
    def get_data_list(self, question, retmax=50, if_split_n=False, if_get_urls=False):
        return self.database.get_data_list(question=question, retmax=retmax, if_split_n=if_split_n, if_get_urls=if_get_urls)

class HRD_database:
    def __init__(self):
        pass
    def get_data_list(self, question, retmax=20):
        pass

class LINS_Database:
    def __init__(self, database_name, retriever=None, local_data_path=None):
        self.database_name = database_name
        if database_name in ['pubmed', 'bing']:
            self.database = Online_database(online_database_name=database_name)
        elif database_name in ['guidelines']:
            self.database = Guidelines(retriever=retriever, local_path=local_data_path)
        else:
            self.database = General_Local_Database(database_name=database_name, retriever=retriever, local_path=local_data_path)
    def get_data_list(self, question, retmax=20, if_split_n=False, if_get_urls=False, question_embedding=None):
        if self.database_name in ['pubmed', 'bing']:
            return self.database.get_data_list(question=question, retmax=retmax, if_split_n=if_split_n, if_get_urls=if_get_urls)
        else:
            return self.database.get_data_list(question=question, question_embedding=question_embedding, retmax=retmax)