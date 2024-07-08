from .retriever.med_linker_search import ReferenceRetiever
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import re, os, torch, json
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from Bio import Entrez, Medline
Entrez.email = "869347360@qq.com"

sysprompt = """
you are a helpful assistant specialized in single-choice. You are to the point and only give the answer in isolation without any chat-based fluff and separate them with commas. Make sure you to only return the option and say nothing else. For example, don't say: "Here are the answer".
Here are some examples:

question: 腰穿中发现血性脑脊液，鉴别是蛛网膜下腔出血还是穿刺时损伤所致，下面哪项检查最有意义？（　　）
options: A: 脑脊液蛋白质含量, B: 脑脊液氯化物含量, C: 脑脊液涂片观察红细胞形态, D: 脑脊液细胞总数, E: 脑脊液中红细胞和自细胞的比值
answer: C

question: 新生儿寒冷损伤综合征重度低温患儿，复温时间是（　　）
options: A: 立即, B: 6小时内, C: 6～12小时, D: 12～24小时, E: 24～48小时
answer: D

question: 2. The causative agent of primary peritonitis caused by hematogenous dissemination is primarily ( )
options: A: Aspergillus, B: Escherichia coli, C: Streptococcus pneumoniae, D: Pseudomonas aeruginosa, E: Anaplasma
answer: C

"""


keywords_prompt = """
You are a helpful assistant specialized in extracting comma-separated keywords.(Given in order of importance, from first to last.) You are to the point and only give the answer in isolation without any chat-based fluff and separate them with commas.Make sure you to only return the keywords and say nothing else. For example, don't say: "Here are the keywords present in the document". 

Here are some examples:
documents: The role of NK cells in tumor immunotherapy
answer: NK cells, tumor immunotherapy

documents:The relation of borderline personality disorder and non-toxic single thyroid nodules
answer:borderline personality disorder, non-toxic single thyroid nodules

documents:This bi-directional MR study supports that host response to SARS-CoV-2 viral infection plays a role in the causal association with increased risk of hypothyroidism. Long-term follow-up studies are needed to confirm the expected increased hypothyroidism risk.
answer:bi-directional MR, SARS-CoV-2, hypothyroidism

documents:What is the relationship between T53 gene mutations and breast cancer?
answer:T53, breast cancer

documents:A patient with Histiocytosis, if V600 occur at BRAF, the recommend treatment drug is?
answer: Histiocytosis, treatment drug

"""

single_choice_prompt = """
you are a helpful assistant specialized in single-choice, you will only give one option not multiple options. You are to the point and only give the answer in isolation without any chat-based fluff and separate them with commas. Make sure you to only return the option and say nothing else. For example, don't say: "Here are the answer".

"""

MedQAprompt = """
The following are multiple choice questions (with answers) about medical knowledge.Give the choice, without any other options or explanations.
"""

pubmedqa_prompt = """
You are a professional assistant who specializes in judgmental questions with three possible answers ("yes", "no", "maybe") for each question. You want to get to the point and use only one of ("yes", "no", "maybe") as the answer to the question, without any other options or explanations. 

"""
pubmedqa_prompt = """
The following are multiple choice questions (with answers) about medical knowledge.Give the choice, without any other options or explanations.
"""


pubmed_little_shots="""

**Question**: Does rural or urban residence make a difference to neonatal outcome in premature birth?
(A) yes 
(B) no
(C) maybe
**Answer:**(A yes
**Question**: Is Alveolar Macrophage Phagocytic Dysfunction in Children With Protracted Bacterial Bronchitis a Forerunner to Bronchiectasis?
(A) yes 
(B) no
(C) maybe
**Answer:**(A yes
**Question**: Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through?
(A) yes 
(B) no
(C) maybe
**Answer:**(A no
**Question**: Could Adult European Pharmacoresistant Epilepsy Patients Be Treated With Higher Doses of Zonisamide?
(A) yes 
(B) no
(C) maybe
**Answer:**(A yes
**Question**: Do elderly patients benefit from surgery in addition to radiotherapy for treatment of metastatic spinal cord compression?
(A) yes
(B) no
(C) maybe
**Answer:**(A no

"""

MedQA_little_shots="""
**Question**: 腰穿中发现血性脑脊液，鉴别是蛛网膜下腔出血还是穿刺时损伤所致，下面哪项检查最有意义？（　　）
(A) 脑脊液蛋白质含量
(B) 脑脊液氯化物含量
(C) 脑脊液涂片观察红细胞形态
(D) 脑脊液细胞总数 
(E) 脑脊液中红细胞和自细胞的比值
**Answer**:(C
**Question**: 新生儿寒冷损伤综合征重度低温患儿，复温时间是（　　）
(A) 立即
(B) 6小时内
(C) 6～12小时
(D) 12～24小时
(E) 24～48小时
**Answer**:(D
**Question**: 初产妇，23岁，规律宫缩10小时，持续观察2小时，宫口由6cm开大至7cm，胎头＋1，胎心140次/分。恰当的处置应为（　　）。
(A) 严密观察产程进展
(B) 肌注杜冷丁
(C) 静脉滴注缩宫素
(D) 立即行人工破膜
(E) 立即行剖宫手术
**Answer**:(A
**Question**: 某药的t1/2为4小时，每隔1个t1/2给药一次，要达到稳态血药浓度，需多少小时（　　）。
(A) 约10小时
(B) 约20小时
(C) 约30小时
(D) 约40小时
(E) 约50小时
**Answer**:(B
**Question**: 女，25岁，停经30天，剧烈腹痛2天，阴道不规则流血1天，今晨从阴道排出三角形膜样物质。检查：贫血貌，下腹部压痛、反跳痛明显。正确治疗应选择（　　）。
(A) 静脉滴注缩宫素
(B) 肌注麦角新碱
(C) 吸宫术终止妊娠
(D) 应用止血药
(E) 行腹腔镜手术
**Answer**:(E
"""


pubmedshots="""

**Question**: Does rural or urban residence make a difference to neonatal outcome in premature birth?
[1]Patients living in rural areas may be at a disadvantage in accessing tertiary health care.AIM: To test the hypothesis that very premature infants born to mothers residing in rural areas have poorer outcomes than those residing in urban areas in the state of New South Wales (NSW) and the Australian Capital Territory (ACT) despite a coordinated referral and transport system.
[2]\"Rural\" or \"urban\" status was based on the location of maternal residence. Perinatal characteristics, major morbidity and case mix adjusted mortality were compared between 1879 rural and 6775 urban infants<32 weeks gestational age, born in 1992-2002 and admitted to all 10 neonatal intensive care units in NSW and ACT.
[3]Rural mothers were more likely to be teenaged, indigenous, and to have had a previous premature birth, prolonged ruptured membrane, and antenatal corticosteroid. Urban mothers were more likely to have had assisted conception and a caesarean section. More urban (93% v 83%) infants were born in a tertiary obstetric hospital. Infants of rural residence had a higher mortality (adjusted odds ratio (OR) 1.26, 95% confidence interval (CI) 1.07 to 1.48, p = 0.005). This trend was consistently seen in all subgroups and significantly for the tertiary hospital born population and the 30-31 weeks gestation subgroup. Regional birth data in this gestational age range also showed a higher stillbirth rate among rural infants (OR 1.20, 95% CI 1.09 to 1.32, p<0.001).  
(A) yes 
(B) no
(C) maybe
**Answer:**(A yes
**Question**: Is Alveolar Macrophage Phagocytic Dysfunction in Children With Protracted Bacterial Bronchitis a Forerunner to Bronchiectasis?
[1]Children with recurrent protracted bacterial bronchitis (PBB) and bronchiectasis share common features, and PBB is likely a forerunner to bronchiectasis. Both diseases are associated with neutrophilic inflammation and frequent isolation of potentially pathogenic microorganisms, including nontypeable Haemophilus influenzae (NTHi), from the lower airway. Defective alveolar macrophage phagocytosis of apoptotic bronchial epithelial cells (efferocytosis), as found in other chronic lung diseases, may also contribute to tissue damage and neutrophil persistence. Thus, in children with bronchiectasis or PBB and in control subjects, we quantified the phagocytosis of airway apoptotic cells and NTHi by alveolar macrophages and related the phagocytic capacity to clinical and airway inflammation.
[2]Children with bronchiectasis (n = 55) or PBB (n = 13) and control subjects (n = 13) were recruited. Alveolar macrophage phagocytosis, efferocytosis, and expression of phagocytic scavenger receptors were assessed by flow cytometry. Bronchoalveolar lavage fluid interleukin (IL) 1\u03b2 was measured by enzyme-linked immunosorbent assay.
[3]For children with PBB or bronchiectasis, macrophage phagocytic capacity was significantly lower than for control subjects (P = .003 and P<.001 for efferocytosis and P = .041 and P = .004 for phagocytosis of NTHi; PBB and bronchiectasis, respectively); median phagocytosis of NTHi for the groups was as follows: bronchiectasis, 13.7% (interquartile range [IQR], 11%-16%); PBB, 16% (IQR, 11%-16%); control subjects, 19.0% (IQR, 13%-21%); and median efferocytosis for the groups was as follows: bronchiectasis, 14.1% (IQR, 10%-16%); PBB, 16.2% (IQR, 14%-17%); control subjects, 18.1% (IQR, 16%-21%). Mannose receptor expression was significantly reduced in the bronchiectasis group (P = .019), and IL-1\u03b2 increased in both bronchiectasis and PBB groups vs control subjects.
(A) yes 
(B) no
(C) maybe
**Answer:**(A yes
**Question**: Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through?
[1]The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment of Hirschsprung disease (HD), but overstretching of the anal sphincters remains a critical issue that may impact the continence. This study examined the long-term outcome of TERPT versus conventional transabdominal (ABD) pull-through for HD.
[2]Records of 41 patients more than 3 years old who underwent a pull-through for HD (TERPT, n = 20; ABD, n = 21) were reviewed, and their families were thoroughly interviewed and scored via a 15-item post-pull-through long-term outcome questionnaire. Patients were operated on between the years 1995 and 2003. During this time, our group transitioned from the ABD to the TERPT technique. Total scoring ranged from 0 to 40: 0 to 10, excellent; 11 to 20 good; 21 to 30 fair; 31 to 40 poor. A 2-tailed Student t test, analysis of covariance, as well as logistic and linear regression were used to analyze the collected data with confidence interval higher than 95%.
[3]Overall scores were similar. However, continence score was significantly better in the ABD group, and the stool pattern score was better in the TERPT group. A significant difference in age at interview between the 2 groups was noted; we therefore reanalyzed the data controlling for age, and this showed that age did not significantly affect the long-term scoring outcome between groups.
(A) yes 
(B) no
(C) maybe
**Answer:**(A no
**Question**: Could Adult European Pharmacoresistant Epilepsy Patients Be Treated With Higher Doses of Zonisamide?
[1]To examine the clinical effect (efficacy and tolerability) of high doses of zonisamide (ZNS) (>500 mg/d) in adult patients with pharmacoresistant epilepsy.
[2]Between 2006 and 2013, all epileptic outpatients treated with high doses of ZNS were selected. Safety and efficacy were assessed based on patient and caregiver reports. Serum levels of ZNS and other concomitant antiepileptic drugs were evaluated if available.
[3]Nine patients (5 female): 8 focal/1 generalized pharmacoresistant epilepsy. Mean age: 34 years. Most frequent seizure type: complex partial seizures; other seizure types: generalized tonic-clonic, tonic, myoclonia. Zonisamide in polytherapy in all (100%), administered in tritherapy in 3 (33%) of 9 patients; mean dose: 633 (600-700) mg/d; efficacy (>50% seizure reduction) was observed in 5 (55%) of 9 patients. Five of 9 patients are still taking high doses of ZNS (more than 1 year). Adverse events were observed in 3 (37%) of 8 patients. Good tolerance to high doses of other antiepileptic drugs had been observed in 6 (66%) of 9 patients. Plasma levels of ZNS were only available in 2 patients; both were in the therapeutic range (34.95, 30.91) (10-40 mg/L)
(A) yes 
(B) no
(C) maybe
**Answer:**(A yes
**Question**: Do elderly patients benefit from surgery in addition to radiotherapy for treatment of metastatic spinal cord compression?
[1]Treatment of elderly cancer patients has gained importance. One question regarding the treatment of metastatic spinal cord compression (MSCC) is whether elderly patients benefit from surgery in addition to radiotherapy? In attempting to answer this question, we performed a matched-pair analysis comparing surgery followed by radiotherapy to radiotherapy alone.
[2]Data from 42 elderly (age>\u200965 years) patients receiving surgery plus radiotherapy (S\u2009+\u2009RT) were matched to 84 patients (1:2) receiving radiotherapy alone (RT). Groups were matched for ten potential prognostic factors and compared regarding motor function, local control, and survival. Additional matched-pair analyses were performed for the subgroups of patients receiving direct decompressive surgery plus stabilization of involved vertebrae (DDSS, n\u2009=\u200981) and receiving laminectomy (LE, n\u2009=\u200945).
[3]Improvement of motor function occurred in 21% after S\u2009+\u2009RT and 24% after RT (p\u2009=\u20090.39). The 1-year local control rates were 81% and 91% (p\u2009=\u20090.44), while the 1-year survival rates were 46% and 39% (p\u2009=\u20090.71). In the matched-pair analysis of patients receiving DDSS, improvement of motor function occurred in 22% after DDSS\u2009+\u2009RT and 24% after RT alone (p\u2009=\u20090.92). The 1-year local control rates were 95% and 89% (p\u2009=\u20090.62), and the 1-year survival rates were 54% and 43% (p\u2009=\u20090.30). In the matched-pair analysis of patients receiving LE, improvement of motor function occurred in 20% after LE\u2009+\u2009RT and 23% after RT alone (p\u2009=\u20090.06). The 1-year local control rates were 50% and 92% (p\u2009=\u20090.33). The 1-year survival rates were 32% and 32% (p\u2009=\u20090.55)
(A) yes
(B) no
(C) maybe
**Answer:**(A no

"""



retrieval_prompt = """

The following is some retrieved knowledge for reference.

retrieval_texts:
"""

Passage_Relevance_prompt = """
You're an expert in the field of mutated cancer. Given a multiple-choice question and a paragraph of knowledge, you need to determine if the knowledge contains the answer to the multiple-choice question. Output "gold" if it does, "irrelevant" if it doesn't. Only consider the knowledge containing the answer to the multiple-choice question if:

<1> The disease name, gene, or mutation site mentioned in the multiple-choice question appears in the knowledge.
<2> The disease name, gene, or mutation site that appears must match exactly with the multiple-choice question, even minor differences like V600E and V600C are not acceptable.
<3> The knowledge must contain one or more drugs from the multiple-choice question options. If no drugs appear or the drugs that appear are not in the options of the multiple-choice question, it is not acceptable.

Just answer with "gold" or "irrelevant" directly, without any further explanation or chat.

question:
"""

MedQA_Passage_Relevance_prompt = """
Given a question about medical knowledge and a paragraph of knowledge, you need to determine if the knowledge contains the answer to the question. Output "gold" if it does, "relevant" if it doesn't. Just answer with "gold" or "relevant" directly, without any further explanation or chat.
question:胃癌最主要的转移途径是（　　）
passage:胃癌是为最常见的恶性肿瘤，其发病原因目前认为与幽门螺杆菌感染以及胃部慢性炎性疾病，比如慢性萎缩性胃炎，或者长期过多地进食腌制、烟熏类的食物。
answer:relevant
question:经调查证实出现医院感染流行时，医院应报告当地卫生行政部门的时间是（　　）
passage:医院调查证实出现医院感染流行或暴发时，应于24小时之内报告当地卫生行政部门。
answer:gold
question:8岁男孩，右肘关节外伤，当地摄X线片诊断肱骨髁上骨折，经两次手法复位未成功，来院时为伤后48小时。查体：右肘关节半屈位，肿胀较重，压痛明显，手指活动障碍，桡动脉搏动弱，手指凉、麻木。应诊断为肱骨髁上骨折合并。options:A: 主要静脉损伤, B: 广泛软组织挫伤, C: 肱动脉损伤, D: 肌肉断裂伤, E: 正中、尺、桡神经损伤
passage:右肘关节外伤，当地摄X线片诊断肱骨髁上骨折，经两次手法复位未成功，来院时为伤后48小时。查体：右肘关节半屈位，肿胀较重，压痛明显，手指活动障碍，桡动脉搏动弱，手指凉、麻木。应诊断为肱骨髁上骨折合并。
answer:relevant
question:8岁男孩，右肘关节外伤，当地摄X线片诊断肱骨髁上骨折，经两次手法复位未成功，来院时为伤后48小时。查体：右肘关节半屈位，肿胀较重，压痛明显，手指活动障碍，桡动脉搏动弱，手指凉、麻木。应诊断为肱骨髁上骨折合并。options:A: 主要静脉损伤, B: 广泛软组织挫伤, C: 肱动脉损伤, D: 肌肉断裂伤, E: 正中、尺、桡神经损伤
passage:问题 8岁男孩，右肘关节外伤，当地拍X线片诊断肱骨髁上骨折，经两次手法复位未成功，来院时为伤后48小时，查体右肘关节半屈位，肿胀较重，压痛明显，手指活动障碍，桡动脉搏动弱，手指凉，麻木，应诊断为肱骨髁上骨折合并  选项 A.广泛软组织挫伤 B.主要静脉损伤 C.肱动脉损伤 D.肌肉断裂伤 E.正中、尺、桡神经损伤  答案 C
answer:gold
question:
"""

Question_Decomposition_prompt = """
You are an expert at breaking down questions, now there is a question that is not very well answered and I can't retrieve information about it online, please help me break this question down into a few sub-questions that are more conducive to answering and retrieving. Note that the question is generated in an open-ended way, without any other explanatory or chatty words, and outputs all the decomposed sub-questions in list form.

Here are some example:

question: A patient with Glioma, if Amplification occur at EGFR, the recommended drug is? options: A: Lapatinib;  B: Midostaurin + High Dose Chemotherapy;  C: Chemotherapy + Panitumumab, Panitumumab, Chemotherapy + Cetuximab, Cetuximab;  D: RLY-2608 

answer: 
1. What gene mutation diseases is Lapatinib generally used to treat?
2. What mutations are typically treated with Midostaurin + High Dose Chemotherapy?
3. Chemotherapy + Panitumumab, Panitumumab, Chemotherapy + Cetuximab, Cetuximab is generally used to treat what gene mutation disease?
4. What mutation is RLY-2608 generally used to treat?
5. A patient with Glioma, if Amplification occurs at EGFR, the recommended drug is?


question: What's the effect of combined antibiotic and immunotherapy treatments?
1. Have there been any clinical trials or studies examining the efficacy of combined antibiotic and immunotherapy treatments?
2. How do the mechanisms of action of antibiotics and immunotherapy intersect or complement each other in the context of treatment?

question: 
"""

oncokb_Passage_Coherence_prompt = """
You are an expert in determining whether there is consistency between the generated sentence and the retrieved paragraph. If there is a content conflict between the generated sentence and the retrieved paragraph, you would answer 'conflict', if the content of the two is consistent, you would answer 'coherence', and if the content of the two is not related, you would answer 'irrelevant'. Be careful to get to the point when answering, don't have any explanations or chitchat, just answer "conflict" or "coherence" or "irrelevant".
Here are some example:

sentence: A patient with All Solid Tumors, if G469A occur at BRAF, the recommend treatment drug is?       options:A: Nilotinib;  B: PLX8394 ;  C: Temsirolimus, Everolimus ;  D: Entrectinib           answer:B
passage:  A patient with All Solid Tumors, if G469A occur at BRAF, the recommended drug is PLX8394
answer: coherence

sentence: question: A patient with Erdheim-Chester Disease, if Oncogenic Mutations occur at ARAF, the recommended drug is?   options: A: Cobimetinib, Trametinib;  B: Niraparib + Prednisone + Abiraterone Acetate;  C: Sorafenib;  D: Binimetinib + Ribociclib                 answer: C
passage:  A patient with Erdheim-Chester Disease, if Oncogenic Mutations occur at ARAF, the recommended drug is Cobimetinib, Trametinib
answer: conflict

sentence: question: A patient with Erdheim-Chester Disease, if Oncogenic Mutations occur at ARAF, the recommended drug is?   options: A: Cobimetinib, Trametinib;  B: Niraparib + Prednisone + Abiraterone Acetate;  C: Sorafenib;  D: Binimetinib + Ribociclib                 answer: A
passage:  A patient with Erdheim-Chester Disease, if Oncogenic Mutations occur at ARAF, the recommended drug is Cobimetinib, Trametinib
answer: coherence


sentence: question: What gene mutations are associated with breast cancer?                 answer: Breast cancer can be associated with mutations in a number of genes. The most well known are BRCA1 and BRCA2 mutations, which significantly increase the risk of breast and ovarian cancer. In addition, other genes such as PTEN, TP53, CDH1, STK11, ATM, CHEK2, PALB2 and BRIP1 have also been implicated in breast cancer.
passage: In conclusion, despite decades of medical research, the causative gene mutation has been identified in less than 30% of cases with a personal and/or family history of hereditary breast cancer. The vast majority of these cases are due to mutations in one of the highly penetrant breast cancer genes (BRCA1, BRCA2, PTEN, TP53, CDH1, and STK11), and there are guidelines available that provide specific guidance for the treatment of these patients.
answer: coherence

sentence: question: What genes are associated with breast cancer?                 answer: Breast cancer can be associated with mutations in a number of genes. The most well known are mutations in the BRCA1 and BRCA2 genes.
passage: A small number of cases with a personal and/or family history of hereditary breast cancer are due to mutations in moderate genetic risk genes (CHEK2, ATM, BRIP1 and PALB2).
answer: irrelevant


"""

Passage_Coherence_prompt = """
You are an expert in determining whether there is consistency between the generated sentence and the retrieved paragraph. If there is a content conflict between the generated sentence and the retrieved paragraph, you would answer 'conflict', if the content of the two is consistent, you would answer 'coherence', and if the content of the two is not related, you would answer 'irrelevant'. Be careful to get to the point when answering, don't have any explanations or chitchat, just answer "conflict" or "coherence" or "irrelevant".

"""

Self_knowledge_prompt = """
You are an expert with a clear self-knowledge of what questions you can answer correctly and what questions you are not so sure about. Here are some questions for you, if you are 90% sure that you answered them correctly, you answer 'CERTAIN', if you are not 90% sure that you answered them correctly, you answer 'UNCERTAIN'. Later I will check if your answers are wrong but you answered 'CERTAIN', You get right to the point and give only 'CERTAIN' or 'UNCERTAIN' without any chat-based fluff and say nothing else. 

question:
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
    def keyword_extraction(self, question, max_num_keywords=0):
        if max_num_keywords <= 0:
            prompt = keywords_prompt + "documents:" + question + "\nanswer:"
        else:
            sp = "(Given in order of importance, from first to last."
            prompt = keywords_prompt.split(sp)[0] + sp + f" The number of keywords is limited to {max_num_keywords}.)" + keywords_prompt.split(sp)[1] + "documents:" + question + "\nanswer:"
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
    
    @torch.no_grad()
    def opt_MedQA_chat(self, question, refs ="", filter_with_different_urls=False, local_data_name="", topk=5, if_pubmed=False, if_merge=False, if_agents=False, if_short_sentences=False):
        references_str = ''
        retriever_query = question.split("options:")[0]
        opts_str = question.split("options:")[1]#':A: 2小时, B: 4小时内, C: 8小时内, D: 12小时内, E: 24小时内'
        #转化为MedQA的格式
        opts_str = opts_str.replace("A:", "(A)").replace("B:", "(B)").replace("C:", "(C)").replace("D:", "(D)").replace("E:", "(E)").replace(", ", "\n")

        #print("retriever_query:", retriever_query)
        if topk  == 0:
            references_str = ""
        elif refs == "":

            if if_pubmed:
                search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
                if search_results != None and search_results != []:

                    recall_search_results = self.ref_retriever.medlinker_query(question=retriever_query, data_list=search_results, filter_with_different_urls=False)
                    rerank_search_results = self.ref_retriever.medlinker_rerank(query=retriever_query, search_results=recall_search_results)
                    merge_search_results = self.ref_retriever.medlinker_merage(query=retriever_query, search_results=rerank_search_results, if_merge=if_merge, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
                    refs = merge_search_results

            else:
                refs = self.ref_retriever.medlinker_merage(query=retriever_query, filter_with_different_urls=False, topk=topk, if_pubmed=False, if_merge=False, local_data_name=local_data_name)
            
            if refs == None or refs == [] or type(refs) == str:
                topk = 0
            else:
                if if_agents:
                    PRM_result = self.PRM(question, refs['texts'], task='MedQA')
                    for id, res in enumerate(PRM_result):
                        if res != 'gold':
                            #删除索引为id的元素
                            del refs['texts'][id]
                for ix, ref in enumerate(refs["texts"]):
                    if ix+1 > topk:
                        break
                    txt = ref
                    references_str += "[" + str(ix+1) + "] " + txt + " \n"
        else:
            if if_agents:
                PRM_result = self.PRM(question, refs['texts'], task='MedQA')
                #for id, res in enumerate(PRM_result):
                #应该倒着枚举，防止出现数组越界
                for id, res in enumerate(reversed(PRM_result)):    
                    if res != 'gold':
                        #删除索引为id的元素
                        del refs['texts'][len(PRM_result) - 1 - id]
            for ix, ref in enumerate(refs["texts"]):
                if ix+1 > topk:
                    break
                references_str += "[" + str(ix+1) + "] " + ref + " \n"
        #prompt = "Answer the following questions as best you can. There have some references:\n" + references_str + "\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.\n" + f"Question: {question}"
        
        #prompt = sysprompt + question + "\nanswer:"
        #question = "question: " + question
        #prompt = single_choice_prompt + question #+ retrieval_prompt + references_str
        #prompt = single_choice_prompt + "**Question**: " + retriever_query + opts_str + "\n**Answer**:"
        #prompt = single_choice_prompt + "**Question**: " + retriever_query +  "(A)\n(B)\n(C)\n(D)\n**Answer**:"
        #prompt = single_choice_prompt + MedQA_little_shots + "**Question**: " + retriever_query + opts_str + "\n**Answer**:"
        if topk ==0:
            prompt = single_choice_prompt + MedQA_little_shots + "**Question**: " + retriever_query + opts_str + "\n**Answer**:"
        else:
            prompt = single_choice_prompt + MedQA_little_shots + references_str + "**Question**: " + retriever_query + opts_str + "\n**Answer**:"
        #prompt = prompt + "\n\nThe following is some retrieved knowledge for reference, but the knowledge below is likely not directly related to the answer to the question, only refer to the knowledge below if you find it informative, otherwise please ignore it." + "\n\nretrieval_texts:" + references_str
        print(prompt)
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        #if if_agents:
        #    sentence = question + "          answer:" + response
        #    coher = self.PCM(sentence, references_str)
        #    if coher=='conflict':
        #        re_prompt = ""
        #        re_prompt += "The generated sentence is inconsistent with the retrieved paragraph. Please re-answer the question based on the retrieved paragraph but your own knowledge."
        #        #print("The generated sentence is inconsistent with the retrieved paragraph. ")
        #        #print("sentence:", sentence)
        #        #print("passage:", references_str)
        #        print(f"answer:{response}和参考内容不一致，需要重新回答")
        #        response, history = self.chat(tokenizer=self.tokenizer, prompt=re_prompt, history=history)
        #        print(f"重新回答的答案为:{response}")
        return { "answer": response, "references": refs}
    
    @torch.no_grad()
    def pubmedQA_opt(self, question, refs ="", filter_with_different_urls=False, local_data_name="", topk=5, if_pubmed=False, if_merge=False, if_short_sentences = False, if_agent=False):
        references_str = ""#topk为0不进行检索，当refs不为空也不检索，通过refs['texts']进行回答
        retriever_query = question
        #print("retriever_query:", retriever_query)
        if topk  == 0:
            references_str = ""
        elif topk == 100 and refs != "":#既加检索，又加gold contexts
            #refs = {}
            #refs['texts']=[]
            texts = refs["texts"]
            for ix, ref in enumerate(refs["texts"]):
                references_str += "[" + str(ix+1) + "] " + ref + " \n"
            tex_id = ix + 2
            search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
            if search_results != None and search_results != []:
                
                recall_search_results = self.ref_retriever.medlinker_query(question=retriever_query, data_list=search_results, filter_with_different_urls=False)
                rerank_search_results = self.ref_retriever.medlinker_rerank(query=retriever_query, search_results=recall_search_results)
                merge_search_results = self.ref_retriever.medlinker_merage(query=retriever_query, search_results=rerank_search_results, if_merge=if_merge, topk=1, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
                refs = merge_search_results
            
                if if_agent:
                    PRM_result = self.PRM(question, refs['texts'])#list[str]
                    for id, res in enumerate(PRM_result):
                        if res != 'gold':
                            #删除索引为id的元素
                            del refs['texts'][id]

            if refs['texts'] != None:
                for ix, ref in enumerate(refs["texts"]):
                    txt = ref
                    #如果tex长度大于2000，截断
                    if len(txt) > 3000:
                        txt = txt[:3000]
                    brk = True
                    for i, tex in enumerate(texts):
                        if tex in ref:
                            brk = False
                            break
                    if brk:
                        references_str += f"[{tex_id}] " + txt + " \n"
        elif refs == "":
            #refs = {}
            #refs['texts']=[]
            search_results = self.keyword_search(retriever_query, topk=5, if_short_sentences=if_short_sentences)
            if search_results != None and search_results != []:
                
                recall_search_results = self.ref_retriever.medlinker_query(question=retriever_query, data_list=search_results, filter_with_different_urls=False)
                rerank_search_results = self.ref_retriever.medlinker_rerank(query=retriever_query, search_results=recall_search_results)
                merge_search_results = self.ref_retriever.medlinker_merage(query=retriever_query, search_results=rerank_search_results, if_merge=if_merge, topk=topk, if_pubmed=if_pubmed, local_data_name="", filter_with_different_urls=False)
                refs = merge_search_results

            
                #for ix, ref in enumerate(refs["texts"]):
                #    txt = ref
                #    refs['texts'].append(txt)

            if not refs:
                return { "references": [], "answer": "" }
            for ix, ref in enumerate(refs["texts"]):
                txt = ref
                #如果tex长度大于3000，截断
                if len(txt) > 3000:
                    txt = txt[:3000]
                references_str += "[" + str(ix+1) + "] " + txt + " \n"
        else:
            for ix, ref in enumerate(refs["texts"]):
                references_str += "[" + str(ix+1) + "] " + ref + " \n"
        #prompt = "Answer the following questions as best you can. There have some references:\n" + references_str + "\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.\n" + f"Question: {question}"
        
        #prompt = sysprompt + question + "\nanswer:"
        question = "**Question**: " + question + "\n"
        if topk==0 and refs == "":
            prompt = pubmedqa_prompt + pubmedshots + question
        else:
            #prompt = pubmedqa_prompt + question + pubmed_little_shots + references_str#question first
            #prompt = pubmedqa_prompt + pubmedshots + question + references_str#references first
            #prompt = pubmedqa_prompt + question + references_str#question first
            #prompt = pubmedqa_prompt + pubmed_little_shots + question + references_str 67.8
            #prompt = pubmedqa_prompt + pubmedshots + references_str + question 72.4
            prompt = pubmedqa_prompt  + references_str + question #73.8
            #prompt = pubmedqa_prompt + question + pubmed_little_shots + references_str#67.4
        prompt += "\n(A) yes\n(B) no\n(C) maybe\n**Answer:**("
        #print(prompt)
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        #print(response)
        #变成小写
        response = response.lower()
        return response, refs
    
#    @torch.no_grad()
#    def chat(self, question, history = None):
#        response, history = self.chat(tokenizer=self.tokenizer, prompt=question, history=history)
#        return response, history
    
    @torch.no_grad()
    def opt_chat(self, question, history = None):
        prompt = single_choice_prompt + question
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=history)
        return response
    

    @torch.no_grad()
    def query_opt_chat(self, question, refs ="", filter_with_different_urls=False, local_data_name="", topk=5):
        references_str = ''
        if refs == "":
            refs = self.ref_retriever.medlinker_merage(query=question, filter_with_different_urls=filter_with_different_urls, topk=topk, local_data_name=local_data_name, if_pubmed=False)
            if not refs:
                return { "references": [], "answer": "" }
            for ix, ref in enumerate(refs["texts"]):
                txt = ref
                references_str += "[" + str(ix+1) + "] " + txt + " \n"
        else:
            for ix, ref in enumerate(refs["texts"]):
                references_str += "[" + str(ix+1) + "] " + ref + " \n"
        #prompt = "Answer the following questions as best you can. There have some references:\n" + references_str + "\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.\n" + f"Question: {question}"
        prompt = "MedLinker_opt:" + "question:" + question + "\nretrieval_texts:" + references_str
        response, history = self.chat(tokenizer=self.tokenizer, prompt=prompt, history=None)
        return { "answer": response, "references": refs}
    

    @torch.no_grad()
    def stream_query(self, question):
        refs = self.ref_retriever.query(question)
        if not refs:
            yield { "references": [], "answer": "" }
            return
        yield { "references": refs }
        references_str = ''
        for ix, ref in enumerate(refs):
            txt = ref["text"]
            references_str += f'[{ix+1}]: {txt}' '\\'

        #prompt = 'There is now the following question:\n ' + question + '\n The following references have been collected from the internet so far:\n ' + '[' + prompt + ']' + '\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.'
        prompt = "Answer the following questions as best you can. There have some references:\n" + references_str + "\n Please refer to the above references and give a valid, reasonable, complete and logical answer, with the requirement to give the appropriate citations.\n" + f"Question: {question}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        yield { "answer": response }


    



def load_model(args):
    medlinker_ckpt_path = args.medlinker_ckpt_path or os.getenv("MEDLINKER_CKPT") #or 'THUDM/WebGLM'
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
