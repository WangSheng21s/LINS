import torch
from transformers import AutoTokenizer, AutoModel
import os

from typing import Optional, Union, List, Dict, Tuple, Iterable, Callable, Any
#from FlagEmbedding import BGEM3FlagModel

class ContrieverScorer:
    def __init__(self, retriever_ckpt_path, device=None, max_batch_size=400) -> None:
    # Load model from HuggingFace Hub
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_ckpt_path)
        self.encoder = AutoModel.from_pretrained(retriever_ckpt_path).to(self.device)
        self.encoder.eval()
    
    
    @torch.no_grad()
    def encode(self, sentences: List[str]) -> torch.Tensor:


        sentence_embeddings = torch.tensor([]).to(self.device)
        for sen in sentences:
            # Tokenize sentences
            if sen==None or sen=="":
                continue
            encoded_input = self.tokenizer(text=sen, padding=True, truncation=True, return_tensors='pt').to(self.device)
            # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
            # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            model_output = self.encoder(**encoded_input)
                # Perform pooling. In this case, cls pooling.
            embeddings = model_output[0][:, 0]
            # normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            sentence_embeddings = torch.cat((sentence_embeddings, embeddings))
        return sentence_embeddings
    
    @torch.no_grad()
    def get_query_embeddings(self, sentences: List[str]) -> torch.Tensor:
        
        return self.encode(sentences)

    @torch.no_grad()
    def score_documents_on_query(self, query: str, documents: List[str]) -> torch.Tensor:
        query_embedding = self.encode([query])
        document_embeddings = self.encode(documents)
        return (query_embedding@document_embeddings.T)[0]

    
    @torch.no_grad()
    def get_scores(self, query: str, documents: List[str]):
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        
        return self.score_documents_on_query(query, documents)

    @torch.no_grad()
    def select_topk(self, query: str, documents: List[str], k=1):
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        scores = self.get_scores(query, documents)
        scores = scores.clone().detach().to(torch.float32)
        return scores.topk(min(k, len(scores)))


class ReferenceFilter:
    def __init__(self, retriever_ckpt_path, device=None, max_batch_size=400) -> None:
        self.scorer = ContrieverScorer(retriever_ckpt_path, device, max_batch_size)

    @torch.no_grad()
    def produce_references(self, query, paragraphs: List[Dict[str, str]], topk=5) -> List[Dict[str, str]]:
        """Individually calculate scores of each sentence, and return `topk`. paragraphs should be like a list of {title, url, text}."""
        # paragraphs = self._pre_filter(paragraphs)
        texts = [item['text'] for item in paragraphs]
        topk = self.scorer.select_topk(query, texts, topk)
        indices = list(topk.indices.detach().cpu().numpy())
        return [paragraphs[idx] for idx in indices]
    
    @torch.no_grad()
    def produce_medlinker_references(self, query, paragraphs: List[Dict[str, str]], topk=5, filter_with_different_urls=True) -> List[Dict[str, str]]:
        """Individually calculate scores of each sentence, and return `topk`. paragraphs should be like a list of {title, url, text}."""
        # paragraphs = self._pre_filter(paragraphs)
        texts = [item['text'] for item in paragraphs]
        if filter_with_different_urls:
            scores = self.scorer.get_scores(query, texts)
            #得到了分数，然后排序，然后搞个url计数器，如果url出现过两次了，就不要了精良保证多url
            urls = [item['url'] for item in paragraphs]   
            urls_dict = {item['url']:0 for item in paragraphs}#标记清楚有哪些urls
            _, indices = scores.sort(descending=True)
            return_indices = torch.tensor([], dtype=torch.int8)
            return_scores = torch.tensor([], dtype=torch.float32)
            return_texts = []
            return_urls = []
            return_titles = []
            for idx in indices:
                if urls_dict[urls[idx]] < 2:
                    urls_dict[urls[idx]] += 1
                    return_indices = torch.cat((return_indices, torch.tensor([idx])))
                    return_scores = torch.cat((return_scores, torch.tensor([scores[idx]])))
                    return_texts.append(texts[idx])
                    return_urls.append(urls[idx])
                    return_titles.append(paragraphs[idx]['title'])
                    if len(return_titles) == topk:
                        break
                else:
                    continue
            return return_indices, return_scores, return_texts, return_urls, return_titles
        else:
            topk = self.scorer.select_topk(query, texts, topk)
            indices = list(topk.indices.detach().cpu().numpy())
            #print(indices)
            return topk[1], topk[0], [paragraphs[x]['text'] for x in indices], [paragraphs[x]['url'] for x in indices], [paragraphs[x]['title'] for x in indices]
            #return indices, scores[indices], [self.text_list[x] for x in indices], [self.url_list[x] for x in indices], [self.title_list[x] for x in indices]
            #return [paragraphs[idx] for idx in indices]

