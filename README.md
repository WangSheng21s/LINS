<h1> LINS: A Multi-Agent Retrieval-Augmented Framework for Enhancing the Quality and Credibility of LLMs’ Medical Responses</h1>

<div align="justify">
We developed LINS, a multi-agent retrieval-augmented framework seamlessly adaptable to any medical vertical without additional training or fine-tuning. Additionally, LINS introduces innovative algorithms, including Multi-Agent Iterative Retrieval Augmented Generation (MAIRAG) algorithm and Keyword Extraction Degradation (KED), aiming to generate high-quality Citation-Based Generative Text (CBGT). Furthermore, we proposed the Link-Eval automated evaluation system to assess CBGT quality.LINS achieved state-of-the-art (SOTA) performance in both subjective and human evaluations on specialized medical datasets. Additionally, we showcased the promising potential of LINS in real clinical scenarios, including assisting physicians in evidence-based medical practice and helping patients with medical order explanation, yielding encouraging results. In conclusion, LINS serves as a general medical question-answering framework that helps LLMs overcome limitations, effectively improving the quality and credibility of medical responses. Our study demonstrates that retrieving high-quality evidence enables LLMs to generate superior medical responses, and that providing evidence-traceable formats enhances credibility. This approach helps overcome user trust barriers toward LLMs, thereby increasing their applicability and value in the medical field.
</div>

<br>

![paper](./assets/LINS.png)

# Documentation 

LINS comes with different functionalities.

- [**Keyword Extraction**](#keyword-extraction)
- [**Retrieve Evidence**](#retrieve-evidence)
- [**Direct Multi-Round Q&A**](#direct-multi-round-qa)
- [**Original Retrieval-Augmented Generation**](#original-retrieval-augmented-generation)
- [**Multi-Agent Iterative Retrieval-Augmented Generation**](#multi-agent-iterative-retrieval-augmented-generation)
- [**Integrate Local Knowledge Base for Answering**](#integrate-local-knowledge-base-for-answering)
- [**Link-Eval Computation**](#link-eval-computation)

## Environmental Preparation

Clone this repo, and install python requirements.

```bash
conda create --name LINS python=3.11.6
conda activate LINS
pip install -r requirements.txt
conda install -c pytorch faiss-gpu
```

Install torch
```bash
pip install torch==2.1.0+cu118 torchvision torchaudio -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

Install Nodejs.

```bash
apt install nodejs # If you use Ubuntu
```

Then, set the environment variable `OPEN_API_KEY`/`GEMINI_KEY` to your key if you want to use Openai/Gemini API.

```bash
export OPEN_API_KEY_KEY=YOUR_KEY
export GEMINI_KEY=YOUR_KEY
```


# Usage 
We provide examples of the main use cases of LINS, utilizing the GPT-4o LLM, the `text-embedding-3-large` retriever, and the PubMed retrieval database. Please ensure you set the environment variable beforehand: 
```bash
export OPENAI_API_KEY=YOUR_KEY
```
```bash
from model.model_LINS import LINS

lins = LINS() # Initialization

#Direct Multi-Round Q&A
response, history = lins.chat(question="hello") 

#Generating Citation-Based Generative Text using  the MAIRAG algorithm
response, urls, retrieved_passages, history, sub_questions = lins.MAIRAG(question="For Parkinson's disease, whether prasinezumab showed greater benefits on motor signs progression in prespecified subgroups with faster motor progression?")

#Generating Evidence-Based Recommendations for Evidence-Based Medicine Practice
response, urls, retrieved_passages, history, PICO_question = lins.AEBMP(patient_information="A 76-year-old female patient was admitted to the hospital due to "numbness in the left lower limb for 1 year and involuntary tremors in the right lower limb for more than 3 months." The patient reported experiencing numbness in the left lower limb a year ago without any apparent trigger, for which no specific treatment was administered. Three months ago, she began experiencing involuntary tremors in the right lower limb without any apparent cause. The tremors intensified during moments of mental tension or emotional excitement and eased during sleep. Tremors were also observed in the right upper limb when holding objects, accompanied by difficulty initiating walking, feelings of fatigue, and memory decline, primarily affecting recent memory. She reported no additional symptoms, such as decreased sense of smell, shortness of breath, chest tightness, frequent nightmares, suspiciousness, or limb numbness. The patient sought medical attention at a local hospital, where she was diagnosed with "Parkinson's disease" and prescribed "Tasud 50 mg, three times daily." Two months ago, she experienced a coma after taking the medication, with no response to external stimuli, and was urgently taken to the local hospital, where her blood glucose level was measured at 1.4 mmol/L. Her condition improved after receiving appropriate symptomatic treatment. She is currently taking "Madopar 125 mg, three times daily" regularly, is able to perform fine motor tasks adequately, and can manage daily activities independently. Since the onset of her illness, she has had a generally stable mental state, with a normal appetite, good sleep, bowel movements every 2-3 days, normal urination, and no significant changes in body weight.", PICO_question="For Parkinson's disease, whether prasinezumab showed greater benefits on motor signs progression in prespecified subgroups with faster motor progression?", if_guidelines=False)

```

## Direct Multi-Round Q&A

```bash
from arguments import get_medlinker_args
from model.modeling_medlinker import load_model

args = get_medlinker_args()
args.medlinker_ckpt_path = "./model/generator/Qwen1.5-110B-Chat"

medlinker = load_model(args)

response, history = medlinker.chat(prompt="hello", history=None)

"""response
'Hello there! How can I assist you today?'
"""

response, history = medlinker.chat(prompt="do models using demographic shortcuts have unfair predictions across subpopulations?", history=history)

"""response
Models that use demographic shortcuts, or proxies for sensitive attributes like age, gender, race, etc., can indeed lead to unfair predictions across subpopulations. This happens because relying on demographic characteristics can inadvertently bake in societal biases and historical inequalities into the model's decision-making process. When a model uses such shortcuts, it may:\n\n1. **Sterotyping**: It may make assumptions based on group characteristics rather than individual qualities or circumstances, leading to unfair treatment of certain groups. For example, assuming that all individuals from a particular demographic have lower creditworthiness without considering their personal financial history.\n\n2. **Disparate Impact**: Certain subpopulations might be disproportionately affected by the model's decisions. If a model uses a demographic feature that is not directly related to the outcome but is correlated with it due to societal factors, it can lead to unequal outcomes. For instance, using zip codes as a proxy for credit risk could inadvertently discriminate against certain ethnic groups if those areas have historically been subject to redlining.\n\n3. **Oversimplification**: Demographic variables can oversimplify complex social dynamics, neglecting other important factors that might contribute to different outcomes within those groups.\n\nTo mitigate these issues, developers of predictive models often employ techniques such as:\n\n- **Fairness-aware machine learning**: This involves adjusting algorithms to ensure they treat all groups fairly according to specific fairness definitions (e.g., demographic parity, equalized odds).\n  \n- **Feature selection**: Careful consideration is given to which features to include, avoiding those that could lead to unfair biases.\n  \n- **Blinding**: Removing sensitive attributes from the training data to prevent the model from directly learning from them.\n  \n- **Adaptive or contextual models**: These models might consider additional context or use more nuanced approaches to capture individual circumstances rather than relying on broad demographic categories.\n\nIt's important to note that achieving fairness in machine learning is a complex task, often involving trade-offs between different fairness definitions and accuracy. Context and the specific application domain play a crucial role in determining what constitutes a fair model.
"""
```


## Original Retrieval-Augmented Generation

```bash
from arguments import get_medlinker_args
from model.modeling_medlinker import load_model

args = get_medlinker_args()
args.medlinker_ckpt_path = "./model/generator/Qwen1.5-110B-Chat"

medlinker = load_model(args)

question = "What are the effects of combining antibiotics and immunotherapy?"
results = medlinker.Original_RAG(question)

"""results
answer: Combining antibiotics and immunotherapy has not been specifically detailed in the provided knowledge regarding gastric MALT lymphoma. However, H. pylori eradication, typically achieved with antibiotics, is the first-line treatment leading to high remission rates, while immunotherapy like rituximab is reserved for non-responding cases. The synergistic effects or specific combinations in the context of MALT lymphoma treatment are not mentioned. In general, antibiotic therapy targets the bacterial infection, whereas immunotherapy aims to enhance or restore the patient's immune response to cancer cells. Further research would be necessary to discuss the combined effects in detail.[1]

refs: [1] https://pubmed.ncbi.nlm.nih.gov/24363507
"""
```

## Multi-Agent Iterative Retrieval-Augmented Generation

```bash
from arguments import get_medlinker_args
from model.modeling_medlinker import load_model

args = get_medlinker_args()
args.medlinker_ckpt_path = "./model/generator/Qwen1.5-110B-Chat"

medlinker = load_model(args)

question = "What are the effects of combining antibiotics and immunotherapy?"
response, urls, retrieved_passages, history = medlinker.MAIRAG(question)
#response, urls, retrieved_passages, history = medlinker.MAIRAG(question, history = history)#continue chat

"""results
answer: Combining antibiotics with immunotherapy has demonstrated enhanced treatment efficacy against bacterial infections, particularly in combating drug-resistant pathogens. For instance, the coadministration of Clofazimine (CFZ) and Rapamycin (RAPA) effectively eliminates both multiple and extensively drug-resistant (MDR and XDR) strains of Mycobacterium tuberculosis in a mouse model by boosting T-cell memory and polyfunctional TCM responses, while also reducing latency-associated gene expression in human macrophages [2]. This approach not only improves bacterial clearance but also holds promise for addressing the issue of drug resistance and disease recurrence in tuberculosis. Similarly, N-formylated peptides have shown adjunctive therapeutic effects when combined with anti-tuberculosis drugs (ATDs), conferring additional therapeutic benefits in mouse models of TB by enhancing neutrophil function and reducing bacterial load [3]. These findings highlight the potential of combining antimicrobial and immunomodulatory agents to achieve improved outcomes in bacterial infection treatment.

refs: [1] https://pubmed.ncbi.nlm.nih.gov/37481650
      [2] https://pubmed.ncbi.nlm.nih.gov/37290049
      [3] https://pubmed.ncbi.nlm.nih.gov/31046503
      [4] https://pubmed.ncbi.nlm.nih.gov/33552074
      [5] https://pubmed.ncbi.nlm.nih.gov/11244046
"""
```

## Integrate Local Knowledge Base for Answering

![LRAG](./assets/LocalRAG.png)

Assume you have a local knowledge base that you want to connect to LINS for RAG (using OncoKB as an example). Ensure you have prepared the local knowledge base JSON file in the add_dataset folder as shown below (the format can refer to oncokb.json).

```bash
├── add_database.py
├── __init__.py
└── oncokb
    └── oncokb.json
```

Follow the steps below to get the local vector knowledge base.
```bash
pip install FlagEmbedding
cd add_dataset
python add_database.py oncokb
```
Then you get the vector knowledge base at the end of embedding.json

```bash
├── add_database.py
├── __init__.py
└── oncokb
    └── oncokb.json
    └── oncokb_embedding.json
```
The above operations only need to be performed the first time a new knowledge base is accessed, and subsequently LINS can be invoked to retrieve the local knowledge base for RAG.

```bash
from arguments import get_medlinker_args
from model.modeling_medlinker import load_model

args = get_medlinker_args()
args.medlinker_ckpt_path = "./model/generator/Qwen1.5-110B-Chat"

medlinker = load_model(args)

question = "What medications should be used to treat breast cancer?"

#(sub_question_answer, sub_urls, sub_texts) = medlinker.Original_RAG(question=question, loacal_data_name='oncokb', if_pubmed=False)
(sub_question_answer, sub_urls, sub_texts，history) = medlinker.MAIRAG(question=question, loacal_data_name='oncokb', if_pubmed=False)

"""results
The treatment for breast cancer varies depending on the specific genetic alterations present in the tumor. If there is an amplification at the ERBB2 gene, several treatment options are recommended. These include:\n\n1. Chemotherapy combined with Pertuzumab and Trastuzumab [1].\n2. Chemotherapy with Trastuzumab, either alone or in combination [2].\n3. Tucatinib, Capecitabine, and Trastuzumab as a therapeutic regimen [3].\n4. Alternatively, for cases with ERBB2 amplification, Lapatinib with Letrozole or in combination with Capecitabine can also be considered [5].\n\nFor patients with a specific mutation (H1047R) in the PIK3CA gene, the recommended medications expand to include LOXO-783 in various combinations, such as with Fulvestrant, LY3484356, Abemaciclib, Paclitaxel, or Aromatase Inhibition Therapy [4].\n\nIt's important to note that these recommendations are tailored to specific genetic profiles and should be prescribed by an oncologist based on comprehensive analysis and clinical judgment.

refs: 'https://www.oncokb.org/entry/1873'
      'https://www.oncokb.org/entry/5556'
      'https://www.oncokb.org/entry/4709'
      'https://www.oncokb.org/entry/4697'
      'https://www.oncokb.org/entry/7735'
      'https://www.oncokb.org/entry/4697'
      'https://www.oncokb.org/entry/3910'
      'https://www.oncokb.org/entry/2371'
      'https://www.oncokb.org/entry/287'
      'https://www.oncokb.org/entry/6001'
"""
```

## Link-Eval Computation

![Link-Eval](./assets/Link-Eval1.png)
If you want to use Link-Eval to evaluate the quality of citation-based generative text (CBGT), please follow the steps below to set up the environment and encapsulate the data.

First, set up the environment:
```bash
pip install prettytable
pip install sentencepiece
```
Next, download the model from [unieval-sum](https://huggingface.co/MingZhong/unieval-sum) and store it in the following directory:
```bash
├──model
│   ├──retriever
│   ├──generator
│   ├──NLI
│   ├──UniEval
│   │   ├──unieval-sum
```
Finally, use the following code to calculate the Link-Eval score:
```bash
from Link_Eval import LinkEval, convert_to_statements

linkeval = LinkEval(NLI_path="./model/MLI/T5-11B", unieval_path="./model/UniEval/unieval-sum")

question = "What are the effects of combining antibiotics and immunotherapy?"

answer = "Combining antibiotics with immunotherapy has demonstrated enhanced treatment efficacy against bacterial infections, particularly in combating drug-resistant pathogens. For instance, the coadministration of Clofazimine (CFZ) and Rapamycin (RAPA) effectively eliminates both multiple and extensively drug-resistant (MDR and XDR) strains of Mycobacterium tuberculosis in a mouse model by boosting T-cell memory and polyfunctional TCM responses, while also reducing latency-associated gene expression in human macrophages [2]. This approach not only improves bacterial clearance but also holds promise for addressing the issue of drug resistance and disease recurrence in tuberculosis. Similarly, N-formylated peptides have shown adjunctive therapeutic effects when combined with anti-tuberculosis drugs (ATDs), conferring additional therapeutic benefits in mouse models of TB by enhancing neutrophil function and reducing bacterial load [3]. These findings highlight the potential of combining antimicrobial and immunomodulatory agents to achieve improved outcomes in bacterial infection treatment."

refs = ["The advent of drug-resistant pathogens results in the occurrence of stubborn bacterial infections that cannot be treated with traditional antibiotics. Antibacterial immunotherapy by reviving or activating the body's immune system to eliminate pathogenic bacteria has confirmed promising therapeutic strategies in controlling bacterial infections. Subsequent studies found that antimicrobial immunotherapy has its own benefits and limitations, such as avoiding recurrence of infection and autoimmunity-induced side effects. Current studies indicate that the various antibacterial therapeutic strategies inducing immune regulation can achieve superior therapeutic efficacy compared with monotherapy alone. Therefore, summarizing the recent advances in nanomedicine with immunomodulatory functions for combating bacterial infections is necessary. Herein, we briefly introduce the crisis caused by drug-resistant bacteria and the opportunity for antibacterial immunotherapy. Then, immune-involved multimodal antibacterial therapy for the treatment of infectious diseases was systematically summarized. Finally, the prospects and challenges of immune-involved combinational therapy are discussed.", "Mycobacterium tuberculosis, the causative agent of tuberculosis, is acquiring drug resistance at a faster rate than the discovery of new antibiotics. Therefore, alternate therapies that can limit the drug resistance and disease recurrence are urgently needed. Emerging evidence indicates that combined treatment with antibiotics and an immunomodulator provides superior treatment efficacy. Clofazimine (CFZ) enhances the generation of T central memory (TCM) cells by blocking the Kv1.3+ potassium channels. Rapamycin (RAPA) facilitates M. tuberculosis clearance by inducing autophagy. In this study, we observed that cotreatment with CFZ and RAPA potently eliminates both multiple and extensively drug-resistant (MDR and XDR) clinical isolates of M. tuberculosis in a mouse model by inducing robust T-cell memory and polyfunctional TCM responses. Furthermore, cotreatment reduces the expression of latency-associated genes of M. tuberculosis in human macrophages. Therefore, CFZ and RAPA cotherapy holds promise for treating patients infected with MDR and XDR strains of M. tuberculosis.", "Objective: The current therapeutic regimens for tuberculosis (TB) are complex and involve the prolonged use of multiple antibiotics with diverse side effects that lead to therapeutic failure and bacterial resistance. The standard appliance of immunotherapy may aid as a powerful tool to combat the ensuing threat of TB. We have earlier reported the immunotherapeutic potential of N-formylated peptides of two secretory proteins of Mycobacterium tuberculosis H37Rv. Here, we investigated the immunotherapeutic effect of an N-formylated peptide from Listeria monocytogenes in experimental TB. Methods: The N-terminally formylated listerial peptide with amino acid sequence 'f-MIGWII' was tested for its adjunctive therapeutic efficacy in combination with anti-tuberculosis drugs (ATDs) in the mouse model of TB. In addition, its potential to generate reactive oxygen species (ROS) in murine neutrophils was also evaluated. Results: The LemA peptide (f-MIGWII) induced a significant increase in the intracellular ROS levels of mouse neutrophils (p ≤ .05). The ATD treatment reduced the colony forming units (CFU) in lungs and spleen of infected mice by 2.39 and 1.67 log10 units, respectively (p < .001). Treatment of the infected mice with combination of ATDs and LemA peptide elicited higher therapeutic efficacy over ATDs alone. The histopathological changes in the lungs of infected mice also correlated well with the CFU data. Conclusions: Our results clearly indicate that LemA peptide conferred an additional therapeutic effect when given in combination with the ATDss (p < .01) and hence can be used as adjunct to the conventional chemotherapy against TB.", "Recurrent urinary tract infections (RUTIs) and recurrent vulvovaginal candidiasis (RVVCs) represent major healthcare problems with high socio-economic impact worldwide. Antibiotic and antifungal prophylaxis remain the gold standard treatments for RUTIs and RVVCs, contributing to the massive rise of antimicrobial resistance, microbiota alterations and co-infections. Therefore, the development of novel vaccine strategies for these infections are sorely needed. The sublingual heat-inactivated polyvalent bacterial vaccine MV140 shows clinical efficacy for the prevention of RUTIs and promotes Th1/Th17 and IL-10 immune responses. V132 is a sublingual preparation of heat-inactivated Candida albicans developed against RVVCs. A vaccine formulation combining both MV140 and V132 might well represent a suitable approach for concomitant genitourinary tract infections (GUTIs), but detailed mechanistic preclinical studies are still needed. Herein, we showed that the combination of MV140 and V132 imprints human dendritic cells (DCs) with the capacity to polarize potent IFN-γ- and IL-17A-producing T cells and FOXP3+ regulatory T (Treg) cells. MV140/V132 activates mitogen-activated protein kinases (MAPK)-, nuclear factor-κB (NF-κB)- and mammalian target of rapamycin (mTOR)-mediated signaling pathways in human DCs. MV140/V132 also promotes metabolic and epigenetic reprogramming in human DCs, which are key molecular mechanisms involved in the induction of innate trained immunity. Splenocytes from mice sublingually immunized with MV140/V132 display enhanced proliferative responses of CD4+ T cells not only upon in vitro stimulation with the related antigens contained in the vaccine formulation but also upon stimulation with phytohaemagglutinin. Additionally, in vivo sublingual immunization with MV140/V132 induces the generation of IgG and IgA antibodies against all the components contained in the vaccine formulation. We uncover immunological mechanisms underlying the potential mode of action of a combination of MV140 and V132 as a novel promising trained immunity-based vaccine (TIbV) for GUTIs.", "Helicobacter pylori is a gram negative, spiral, microaerophylic bacterium that infects the stomach of more than 50% of the human population worldwide. It is mostly acquired during childhood and, if not treated, persists chronically, causing chronic gastritis, peptic ulcer disease, and in some individuals, gastric adenocarcinoma and gastric B cell lymphoma. The current therapy, based on the use of a proton-pump inhibitor and antibiotics, is efficacious but faces problems such as patient compliance, antibiotic resistance, and possible recurrence of infection. The development of an efficacious vaccine against H. pylori would thus offer several advantages. Various approaches have been followed in the development of vaccines against H. pylori, most of which have been based on the use of selected antigens known to be involved in the pathogenesis of the infection, such as urease, the vacuolating cytotoxin (VacA), the cytotoxin-associated antigen (CagA), the neutrophil-activating protein (NAP), and others, and intended to confer protection prophylactically and/or therapeutically in animal models of infection. However, very little is known of the natural history of H. pylori infection and of the kinetics of the induced immune responses. Several lines of evidence suggest that H. pylori infection is accompanied by a pronounced Th1-type CD4(+) T cell response. It appears, however, that after immunization, the antigen-specific response is predominantly polarized toward a Th2-type response, with production of cytokines that can inhibit the activation of Th1 cells and of macrophages, and the production of proinflammatory cytokines. The exact effector mechanisms of protection induced after immunization are still poorly understood. The next couple of years will be crucial for the development of vaccines against H. pylori. Several trials are foreseen in humans, and expectations are that most of the questions being asked now on the host-microbe interactions will be answered."]
    
statements = convert_to_statements(answer)

citation_set_precision, citation_precision, citation_recall = linkeval.compute_precision_and_recall(question, statements, refs)

statement_correctness = linkeval.compute_statements_correctness(statements)

stanement_fluency = linkeval.compute_statements_fluency(answer)

"""results
+------------------------------+----------+
|          Dimensions          |  Score   |
+------------------------------+----------+
|    citation_set_precision    |  1.00    |
|      citation_precision      |  1.00    |
|       citation_recall        |  0.50    |
|    statement_correctness     |  1.00    |
|      stanement_fluency       |  0.95    |
+------------------------------+----------+
"""
```

# Data available

All the data used in LINS are publicly accessible. The PubMedQA dataset can be obtained from https://pubmedqa.github.io, and the MedQA dataset is available at https://drie.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view. The HealthSearchQA data can be accessed from https://www.nature.com/articles/s41586-023-06291-2#Sec59. Additionally, datasets such as LocalTest, PubMedQA*, and GeneMedQA are available at https://figshare.com/projects/LINS/212636.


## Model Preparation

The Qwen model can be found at [Qwen1.5-110B on Hugging Face](https://huggingface.co/Qwen/Qwen1.5-110B), the recall model for the retriever at [bge-m3 on Hugging Face](https://huggingface.co/BAAI/bge-m3), and the ranking model at [bge-reranker-v2-m3 on Hugging Face](https://huggingface.co/BAAI/bge-reranker-v2-m3). Finally, the T5 model (used for Link-Eval) is accessible at [t5-11b on Hugging Face](https://huggingface.co/google-t5/t5-11b).




These models are placed in the following positions：
```bash
├──model
│   ├──retriever
│   │   ├──bge
│   │   │   ├──bge-m3
│   │   │   ├──bge-reranker-v2-m3
│   ├──generator
│   │   ├──Qwen1.5-110B-Chat
│   ├──NLI
│   │   ├──T5-11B
```

