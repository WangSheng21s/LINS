We will soon update detailed tutorials.

# Documentation

LINS comes with different functionalities.
- [**LLM Selection**](#llm-selection)
- [**Retriever Selection**](#retriever-selection)
- [**Database Selection**](#database-selection)
- [**Local Retrieval Database Construction**](#local-retrieval-database-construction)
- [**Keyword Extraction**](#keyword-extraction)
- [**Retrieve Evidence Using KED**](#retrieve-evidence-using-ked)
- [**Direct Multi-Round Q&A**](#direct-multi-round-qa)
- [**Original Retrieval-Augmented Generation**](#original-retrieval-augmented-generation)
- [**Multi-Agent Iterative Retrieval-Augmented Generation**](#multi-agent-iterative-retrieval-augmented-generation)
- [**Link-Eval Computation**](#link-eval-computation)

## LLM-Selection
LINS allows users to select the appropriate LLM based on their specific needs. Currently, it supports LLMs from four families: GPT, Gemini, Qwen, and Llama, with plans to continuously update and integrate new LLMs in the future.

Before selecting a specific LLM, here are some necessary conditions:
```bash
# For GPT families  
export OPEN_API_KEY=your_key

# For Gemini families  
export GEMINI_KEY=your_key

#For Qwen families, make sure you have installed Ollama and are able to successfully run the following command:
ollama run qwen2.5-72b

#For Llama families, make sure you have installed Ollama and are able to successfully run the following command:
ollama run llama3.1-70b
```
After that, you can select a specific LLM through the following parameters:
```bash
from model.model_LINS import LINS

#Currently, LINS supports all LLMs from the GPT and Gemini families, as well as Qwen2.5-72b and Llama3.1-70b.
lins = LINS(LLM_name='gpt-4o-mini') 

#LINS also supports selecting `assistant_LLM_name`. The `assistant_LLM` handles some of the multi-agent functions. When users choose high-cost models like `o1-preview` as the main LLM, they can opt to replace the `assistant_LLM` with a more affordable LLM, which will help save some costs.
lins = LINS(LLM_name='o1-preview', assistant_LLM_name='gpt-4o') 
```


## Retriever-Selection
LINS allows users to select the appropriate retriever based on their specific needs. Currently, it supports retrievers from both OpenAI and open-source models: text-embedding-ada-2, text-embedding-3-small, text-embedding-3-large, and BGE (text-embedding- retrievers need OPEN_API_KEY). We plan to continuously update and integrate new retrievers in the future.

You can select a specific retriever through the following parameters:
```bash
from model.model_LINS import LINS

#'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-2', 'BGE' is available.
#The first time you use the BGE model, it will download the corresponding model from Hugging Face. Please ensure that your network connection is stable.
lins = LINS(LLM_name='gpt-4o', retriever_name='text-embedding-3-large')
```

## Database-Selection
LINS allows users to select the appropriate retrieval database based on their specific needs. Currently, it supports both local retrieval databases and online retrieval databases, including PubMed, Bing, and local retrieval databases. We plan to continuously update and integrate new retrieval databases in the future. For detailed instructions on constructing a local retrieval database, please refer to our guide on [**Local Retrieval Database Construction**](#local-retrieval-database-construction).

You can select a specific retrieval database through the following parameters:
```bash
from model.model_LINS import LINS

#'pubmed', 'bing' or {your local database name} is available.
lins = LINS(LLM_name='gpt-4o', retriever_name='text-embedding-3-large', database_name='pubmed')
```

## Local-Retrieval-Database-Construction
LINS supports users in building personalized local retrieval databases. Below, we demonstrate how to construct your own local retrieval database using OncoKB as an example.

### Steps to Build a Local Retrieval Database

1. **Create the OncoKB Folder**:  
   Navigate to the `add_dataset` directory, create an `oncokb` folder, and upload the `oncokb.json` file (ensure the file name remains consistent).  
   We have provided an example `oncokb` folder and `oncokb.json` file in the `add_dataset` directory for reference.

2. **Generate Embeddings**:  
   Run the following command in the `LINS` folder to generate embeddings for the OncoKB data:

   ```bash
   python ./add_dataset/add_database.py --database_name oncokb --retriever_name text-embedding-3-large
   ```

   After successful execution, the `oncokb.embedding.json` file will be created in the `oncokb` folder to store the embeddings for OncoKB.

3. **Integrate OncoKB for Retrieval-Enhanced Q&A**:  
   Once the embeddings are generated, you can integrate the OncoKB local retrieval database into LINS for retrieval-enhanced question answering. Use the following code snippet:

   ```python
   from model.model_LINS import LINS
   lins = LINS(LLM_name='gpt-4o', 'retriever_name'='text-embedding-3-large', database_name='oncokb')
   response, urls, retrieved_passages, history, sub_questions = lins.MAIRAG(question="What is BCR-ABL1?")
   ```

This allows you to leverage the OncoKB local database for advanced, retrieval-augmented AI capabilities in your application.

## Direct Multi-Round Q&A


## Multi-Agent Iterative Retrieval-Augmented Generation

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
## Data available
