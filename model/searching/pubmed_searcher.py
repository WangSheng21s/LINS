from Bio import Entrez, Medline
import re
import time
import threading
import concurrent.futures  

def pubmed_search(question:str, retmax = 10) ->list:
    Entrez.email = "869347360@qq.com"
    keyword = question
    # retmax = 10
    # Entrez.esearch返返回一个采用"handle"的格式数据库标识的列表，这个列表可以用Entrez.read读取。db, term是必选参数
    #start = time.time()
    handle = Entrez.esearch(db='pubmed', term=keyword, retmax=retmax, sort='relevance')
    #end = time.time()
    #print("esearch time: ", end - start)

    # Entrez.read读取数据库列表，返回一个字典 record，该字典包含键"IdList"（表示配备文本查询的ID的列表），“Count”（所有ID的数目）
    record = Entrez.read(handle)
    pmids = record['IdList']
    # print(pmids)
    # Entrez.efetch用上面获得的ID列表或者单个ID作为参数，retmode表示被检索记录格式（text, HTML, XML）。rettype指显示记录的类型，这取决于访问的数据库。PubMed的rettype可以是abstract, citation或medline等。对于UniProt中rettype可以为fasta。retmax是返回的记录总数，上限1w。
    #start = time.time()
    handle = Entrez.efetch(db='pubmed', id=pmids, rettype='medline', retmode='text', retmax=retmax)
    #end = time.time()
    #print("efetch time: ", end - start)
    # Medline模块用来解析Entrez.efetch下载的记录。Medline.parse函数可以将其转换为一个列表。这个列表包含Bio.Medline.Record对象，就像一个字典。最常用的键是TI（标题，Title）、PMID、PG（页码，pages）、AB（摘要，Abstract）和AT（作者，Authors）
    medline_records = Medline.parse(handle)
    records = list(medline_records)
    #A U作者，TI题目，LR日期，TA杂志缩写，JT杂志全称，LID doi号
    literature_info = []
    for pmid, record in zip(pmids, records):
        # # print(record)
        # if 'LID' in record.keys():
        #     text = record["LID"]
        #     start_index = text.find("[pii]") + len("[pii]")
        #     end_index = text.find("[doi]")
        #     if start_index != -1 and end_index != -1:
        #         doi = text[start_index:end_index].strip()
        #         url = "https://doi.org/" + doi
        #     elif end_index != -1:
        #         doi = text[:end_index].strip()
        #         url = "https://doi.org/" + doi
        #     else:
        #         url = None
        #     literature_info.append({'tittle': record['TI'], 'author': record['AU'], 'url': url, 'abstract': record['AB']})
        # else:
        url = "https://pubmed.ncbi.nlm.nih.gov/" + pmid
        literature_info.append({'url': url,
                                'title': record['TI'] if 'TI' in record.keys() else None,
                                'text': record['AB'] if 'AB' in record.keys() else None})
    return literature_info


def search_with_multiple_threads(question: str, num_threads=5, retmax=10) -> list:  
    # 创建一个线程池  
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:  
        # 提交多个任务到线程池  
        futures = [executor.submit(pubmed_search, question, retmax) for _ in range(num_threads)]  
          
        # 遍历futures列表，检查哪个线程先完成  
        for future in concurrent.futures.as_completed(futures):  
            # 如果线程任务被取消，future.exception()会返回CancelledError  
            if future.exception() is concurrent.futures.CancelledError:  
                continue  
              
            # 一旦有一个线程完成了任务，就取消所有其他线程  
            for other_future in futures:  
                if other_future != future:  
                    other_future.cancel()  
              
            # 返回结果  
            return future.result()  

if __name__ == "__main__":
    question = "What are the effects of combining antibiotics and immunotherapy"
    retmax = 10
    result = pubmed_search(question, retmax)
       


#main函数
#if __name__ == '__main__':
#    question_list = ["What are the effects of combining antibiotics and immunotherapy"
#    ]
#    numn_list = [10,20,100]
#    #计算每个问题的查询时间
#    for question in question_list:
#        for num in numn_list:    
#            for rank in range(10):#重复十次
#                start = time.time()
#                try:
#                    result = pubmed_search(question, retmax=num)
#                except:
#                    print("error")
#                    
#                end = time.time()
#                print("num: ", num)
#                print("rank: ", rank)
#                print("question: ", question)
#                print("time: ", end - start)
#                print("\n\n")