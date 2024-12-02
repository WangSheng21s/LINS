import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from model.retriever.med_linker_search import ReferenceRetiever
from itertools import combinations
import re
from utils import convert_to_json
from metric.evaluator import get_evaluator

t5_11b_path = "./model/NLI/T5-11B"



def convert_to_statements(text):
    # 定义正则表达式模式来匹配引用
    pattern = re.compile(r"\[(\d+)\]")
    
    # 以句号分割段落，保留句号作为分隔符
    sentences = re.split(r'(?<=\.\s)', text.strip())
    
    statements = []
    
    for sentence in sentences:
        # 找到句子中的所有引用
        refs = pattern.findall(sentence)
        ref_numbers = {int(ref) - 1 for ref in refs}  # 将引用编号减1以符合要求
        clean_sentence = pattern.sub('', sentence).strip()  # 移除引用后的句子
        if clean_sentence:
            statements.append((clean_sentence, ref_numbers))
    
    return statements




class LinkEval:
    def __init__(self, NLI_path="./model/NLI/T5-11B", unieval_path="./model/UniEval/unieval-sum"):
        self.tokenizer = T5Tokenizer.from_pretrained(NLI_path)
        self.NLI = T5ForConditionalGeneration.from_pretrained(NLI_path)
        self.RefRetriever = ReferenceRetiever()
        self.evaluator = get_evaluator('summarization', model_path=unieval_path)

    def nli_entailment(self, premise, hypothesis):
        # Prepare the input for the T5 model
        input_text = f"nli premise: {premise} hypothesis: {hypothesis}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Generate the output
        outputs = self.NLI.generate(input_ids)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Map the output text to entailment label
        entailment_map = {"entailment": 0, "contradiction": 1, "neutral": 2}

        return entailment_map.get(output_text, 2)  # Default to "neutral" if not found
    
    def compute_correct_citations(self, statements, refs):
        correct_citations = []
        for statement, citation_set in statements:
            if not citation_set:
                continue
            # Check if citation indices are within the bounds of refs list
            if any(i >= len(refs) for i in citation_set):
                continue
            premise = " ".join([refs[i] for i in citation_set])  # Extract text from dictionary
            hypothesis = statement
            entailment_label = self.nli_entailment(premise, hypothesis)
            if entailment_label == 0:  # entailment
                correct_citations.append((statement, citation_set))
        return correct_citations
    
    def compute_correct_citation_count(self, correct_citations, refs):
        correct_count = 0
        correct_citation_details = []  # List to store details of correct citations
        for statement, citation_set in correct_citations:
            correct_set = set()
            #如果集合大小为1
            if len(citation_set) == 1:
                citation = citation_set.pop()
                correct_count += 1
                correct_set.add(citation)
            else:
                for citation in citation_set:
                    remaining_citations = citation_set - {citation}
                    if any(i >= len(refs) for i in remaining_citations):
                        continue
                    premise = " ".join([refs[i] for i in remaining_citations])  # Extract text from dictionary
                    hypothesis = statement
                    entailment_label = self.nli_entailment(premise, hypothesis)
                    if entailment_label != 0:  # not entailment
                        correct_count += 1
                        correct_set.add(citation)
            correct_citation_details.append((statement, citation_set, correct_set))
        return correct_count, correct_citation_details
    
    def compute_precision_and_recall(self, question, statements, refs, p=0.60):
        filtered_statements = [statement for statement in statements if statement[1]]
        total_citations = sum(len(citation_set) for _, citation_set in statements)
        correct_citations = self.compute_correct_citations(filtered_statements, refs)
        correct_count, correct_citation_details = self.compute_correct_citation_count(correct_citations, refs)

        #print("correct_count:",correct_count)
        #print("correct_citation_details:",correct_citation_details)
        # Compute the scores between the question and refs
        passages = refs
        scores = self.RefRetriever.medlinker_compute_score(question, passages)


        valid_refs = [i for i, score in enumerate(scores) if score > p]  # Extract score from dictionary

        # Compute the union of all valid and correct citations
        valid_correct_union = set()
        #print("correct_citation_details:",correct_citation_details)

        for _, _, correct_set in correct_citation_details:
            valid_correct_union.update(correct_set & set(valid_refs))#update()方法用于修改原集合，可以添加新的元素到集合中
        #print("valid_correct_union:",valid_correct_union)
        valid_correct_count = len(valid_correct_union)


        precision = correct_count / total_citations if total_citations > 0 else 1
        recall = valid_correct_count / len(valid_refs) if len(valid_refs) > 0 else 1
        #print(valid_correct_count)
        set_precision = len(correct_citations) / len(filtered_statements) if len(filtered_statements) > 0 else 1

        return set_precision, precision, recall
    
    def compute_statements_correctness(self, statements):
        
        texts = [statement[0] for statement in statements]
    
        # 生成所有可能的两两组合
        text_pairs = list(combinations(texts, 2))
        for premise, hypothesis in text_pairs:
            entailment_label = self.nli_entailment(premise, hypothesis)
            if entailment_label == 1:  # contradiction
                return 0
        return 1
    
    def compute_statements_fluency(self, text):
        pattern = re.compile(r"\[(\d+)\]")
        clean_text = pattern.sub('', text).strip()  # 移除引用后的句子
        output_list = [f'{clean_text}']
        data = convert_to_json(output_list=output_list, src_list = output_list, ref_list = output_list)
        eval_scores = self.evaluator.evaluate(data, print_result=False, overall=False, dims=['fluency'])
        return eval_scores


