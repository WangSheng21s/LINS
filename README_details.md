# Documentation

LINS comes with different functionalities.
- [**LLM Selection**](#llm-selection)
- [**Retriever Selection**](#retriever-selection)
- [**Database Selection**](#database-selection)
- [**Local Retrieval Database Construction**](#local-retrieval-database-construction)
- [**Usage Examples**](#usage-examples)
- [**Privacy Protection Settings**](#privacy-protection-settings)
- [**Multi-Agent Selection**](#multi-agent-selection)
- [**Reproducibility**](#Reproducibility
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
   lins = LINS(LLM_name='gpt-4o', retriever_name='text-embedding-3-large', database_name='oncokb')
   response, urls, retrieved_passages, history, sub_questions = lins.MAIRAG(question="What is BCR-ABL1?")
   ```

This allows you to leverage the OncoKB local database for advanced, retrieval-augmented AI capabilities in your application.


### Explanation:
- **LLM_name**: Specifies the local LLM to use (in this case, `qwen2.5-72b`).
- **retriever_name**: Sets the local retrieval model (here, `BGE`).
- **database_name**: Defines the local database used for retrieval (in this case, `oncokb`).

This setup ensures that all operations are performed locally, without any data leaving the user’s environment, offering a higher level of privacy protection.
## Usage-Examples
We provide examples of the main use cases of LINS, utilizing the GPT-4o LLM, the `text-embedding-3-large` retriever, and the PubMed retrieval database. Please ensure you set the environment variable beforehand: 
```bash
export OPEN_API_KEY=YOUR_KEY
```
```bash
from model.model_LINS import LINS

lins = LINS() # Initialization

#Direct Multi-Round Q&A
response, history = lins.chat(question="hello") 

#Generating Citation-Based Generative Text using  the MAIRAG algorithm
response, urls, retrieved_passages, history, sub_questions = lins.MAIRAG(question="For Parkinson's disease, whether prasinezumab showed greater benefits on motor signs progression in prespecified subgroups with faster motor progression?")

#Generating Evidence-Based Recommendation for Evidence-Based Medicine Practice
response, urls, retrieved_passages, history, PICO_question = lins.AEBMP(PICO_question="For Parkinson's disease, whether prasinezumab showed greater benefits on motor signs progression in prespecified subgroups with faster motor progression?", if_guidelines=False, patient_information="A 76-year-old female patient was admitted to the hospital due to "numbness in the left lower limb for 1 year and involuntary tremors in the right lower limb for more than 3 months." The patient reported experiencing numbness in the left lower limb a year ago without any apparent trigger, for which no specific treatment was administered. Three months ago, she began experiencing involuntary tremors in the right lower limb without any apparent cause. The tremors intensified during moments of mental tension or emotional excitement and eased during sleep. Tremors were also observed in the right upper limb when holding objects, accompanied by difficulty initiating walking, feelings of fatigue, and memory decline, primarily affecting recent memory. She reported no additional symptoms, such as decreased sense of smell, shortness of breath, chest tightness, frequent nightmares, suspiciousness, or limb numbness. The patient sought medical attention at a local hospital, where she was diagnosed with "Parkinson's disease" and prescribed "Tasud 50 mg, three times daily." Two months ago, she experienced a coma after taking the medication, with no response to external stimuli, and was urgently taken to the local hospital, where her blood glucose level was measured at 1.4 mmol/L. Her condition improved after receiving appropriate symptomatic treatment. She is currently taking "Madopar 125 mg, three times daily" regularly, is able to perform fine motor tasks adequately, and can manage daily activities independently. Since the onset of her illness, she has had a generally stable mental state, with a normal appetite, good sleep, bowel movements every 2-3 days, normal urination, and no significant changes in body weight.")

#Generating Evidence-Based Answers for Medical Order Explanation to Patients
medical_term_explanations, clinical_answer = lins.MOEQA(if_QA=True, if_explanation=True, question="Why do I have ischemic bowel disease?", explain_text="Preliminary Diagnosis: Ischemic Bowel Disease.\nManagement: Instructed patient to rest in bed, avoid stress, keep nil by mouth, provide continuous oxygen inhalation, fluid replacement to maintain water and electrolyte balance, use papaverine hydrochloride to relieve spasms and pain, dilate blood vessels to maintain blood flow, and observe symptoms the next day.", patient_information="Gender: Female, Age: 53 years\nChief Complaint: Admitted for \"recurrent abdominal pain and bloating for over 2 years.\"\nCurrent Illness History: The patient experienced abdominal pain 2 years ago, especially under the xiphoid process, presenting as intermittent dull pain and discomfort, with episodes lasting variable durations, aggravated after a full meal, accompanied by bloating, bitter mouth, fatigue, without cough or sputum, chills, or fever. Local hospital's gastroscopy diagnosed chronic gastritis, treated with oral Zhi Shu Kuang Zhong Capsules, Domperidone Tablets, etc. with symptoms improving occasionally but easily recurring. Four days ago, a broad-based polyp about 0.6 cm in diameter was found in the hepatic flexure and removed with endoscopic clipping, with no abnormalities observed in the rest of the colon and rectum; on the first postoperative day, the patient experienced abdominal cramps and frequent bloody stools.\nPast History: No history of hypertension, diabetes, coronary artery disease; no drug or food allergies, no history of ulcerative colitis or Crohn's disease, no history of hematological diseases.\nPhysical Examination: Pulse 71/min, Respiration 20/min, Blood pressure 120/80 mmHg (1 mmHg=0.133 kPa). Abdomen flat, no gastrointestinal shape or peristaltic wave observed, no abdominal wall varicosities, whole abdomen soft, tenderness under xiphoid and around navel, no rebound tenderness or muscle tension, liver and spleen not palpable below ribs. Murphy sign negative. Whole abdomen without palpable mass, shifting dullness negative, no knocking pain in liver and kidney areas, bowel sounds 4/min.\nAuxiliary Examination: No abnormalities in routine blood tests and coagulation function tests. Colonoscopy: diffuse dark red and purplish-red changes in descending colon and sigmoid colon mucosa, significant swelling with multiple patchy erosions and irregular shallow ulcers, bruising; observation of post-polypectomy site revealed a clip device in place, no bleeding points found; Abdominal enhanced CT: swelling of the descending and sigmoid colon with multiple small blood vessels showing around normally contrasting bowel segments, abdominal vascular CTA showed clear mesenteric artery and major branches, no thrombosis or significant stenosis noted.\nPreliminary Diagnosis: Ischemic Bowel Disease.\nManagement: Instructed patient to rest in bed, avoid stress, keep nil by mouth, provide continuous oxygen inhalation, fluid replacement to maintain water and electrolyte balance, use papaverine hydrochloride to relieve spasms and pain, dilate blood vessels to maintain blood flow, and observe symptoms the next day.")
```

## Privacy-Protection-Settings
If you have very high privacy protection requirements, LINS supports building fully localized configurations to ensure that your data remains entirely protected on-premise, without being uploaded to the internet or exposed to potential data breaches. Below is an example of how to set up a fully localized configuration:

### Fully Localized Configuration Example:

```python
from model.model_LINS import LINS

# Initialize LINS with a local configuration
lins = LINS(LLM_name="qwen2.5-72b", retriever_name='BGE', database_name='oncokb')

# Perform a retrieval-augmented question-answering task
lins.MAIRAG(question="What is BCR-ABL1?")
```

## Multi-Agent-Selection
The **MAIRAG** algorithm provides higher quality answers, but it comes with higher token and time consumption compared to the original RAG algorithm. If users prefer faster responses with lower costs, LINS allows users to disable the multi-agent components, effectively reverting the MAIRAG algorithm to the original RAG algorithm.

Here is how you can configure LINS to disable the multi-agent modules and use the original RAG algorithm for faster and more cost-efficient service:

### Disable Multi-Agent Modules and Use Original RAG:

```python
from model.model_LINS import LINS

# Initialize LINS with specific settings
lins = LINS(LLM_name='gpt-4o', retriever_name='text-embedding-3-large', database_name='pubmed')

# Use MAIRAG with disabled multi-agent features to revert to RAG
response, urls, retrieved_passages, history, sub_questions = lins.MAIRAG(question="What is BCR-ABL1?", if_PRA=False, if_SKA=False, if_QDA=False, if_PCA=False)
```

### Explanation:
- **if_PRA=False**: Disables the Passage Relevant Agent.
- **if_SKA=False**: Disables the Self-Knowledge Agent.
- **if_QDA=False**: Disables the Question Decomposition Agent.
- **if_PCA=False**: Disables the Passage Coherence Agent.

By setting these options to `False`, you will use the original RAG algorithm, resulting in faster responses with reduced token usage and lower overall consumption.


## Reproducibility

We provide example commands for reproducing the evaluation results in the `evaluate` section. Due to potential variability introduced by large language model versions, knowledge updates on the web, and inherent model randomness, exact replication of experimental results may show reasonable fluctuations.

To minimize such variability, we include the retrieval results generated by the `gpt-4o-mini` model on the `PubMedQA*` dataset at the time of our experiments. These are located in `./evaluate/search_results/pubmedqa`.

You may use the following command to attempt reproduction of our results:

```bash
cd evaluate
python -m pdb eval_pubmedqastar_ALL.py \
    --model_name_list=gpt-4o-mini \
    --num_batch=1 \
    --task_list=pubmedqa_star \
    --method_list=MAIRAG \
    --retrieval_passages_path_list=./search_results/pubmedqa/pubmedqa_star_gpt-4o-mini_MAIRAG__results.jsonl
```

This setup allows the evaluation pipeline to directly use the provided retrieval outputs, reducing fluctuations caused by re-running the retrieval process.

We also provide the original experimental results in `./evaluate/evaluate_results/pubmedqa_star_gpt-4o-mini_MAIRAG__results.jsonl`.

