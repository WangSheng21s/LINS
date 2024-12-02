<h1> LINS: A Multi-Agent Retrieval-Augmented Framework for Enhancing the Quality and Credibility of LLMs’ Medical Responses</h1>

<div align="justify">
We developed LINS, a multi-agent retrieval-augmented framework seamlessly adaptable to any medical vertical without additional training or fine-tuning. Additionally, LINS introduces innovative algorithms, including Multi-Agent Iterative Retrieval Augmented Generation (MAIRAG) algorithm and Keyword Extraction Degradation (KED), aiming to generate high-quality Citation-Based Generative Text (CBGT). Furthermore, we proposed the Link-Eval automated evaluation system to assess CBGT quality.LINS achieved state-of-the-art (SOTA) performance in both subjective and human evaluations on specialized medical datasets. Additionally, we showcased the promising potential of LINS in real clinical scenarios, including assisting physicians in evidence-based medical practice and helping patients with medical order explanation, yielding encouraging results. In conclusion, LINS serves as a general medical question-answering framework that helps LLMs overcome limitations, effectively improving the quality and credibility of medical responses. Our study demonstrates that retrieving high-quality evidence enables LLMs to generate superior medical responses, and that providing evidence-traceable formats enhances credibility. This approach helps overcome user trust barriers toward LLMs, thereby increasing their applicability and value in the medical field.
</div>

<br>

![paper](./assets/LINS.png)


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

#Generating Evidence-Based Recommendation for Evidence-Based Medicine Practice
response, urls, retrieved_passages, history, PICO_question = lins.AEBMP(PICO_question="For Parkinson's disease, whether prasinezumab showed greater benefits on motor signs progression in prespecified subgroups with faster motor progression?", if_guidelines=False, patient_information="A 76-year-old female patient was admitted to the hospital due to "numbness in the left lower limb for 1 year and involuntary tremors in the right lower limb for more than 3 months." The patient reported experiencing numbness in the left lower limb a year ago without any apparent trigger, for which no specific treatment was administered. Three months ago, she began experiencing involuntary tremors in the right lower limb without any apparent cause. The tremors intensified during moments of mental tension or emotional excitement and eased during sleep. Tremors were also observed in the right upper limb when holding objects, accompanied by difficulty initiating walking, feelings of fatigue, and memory decline, primarily affecting recent memory. She reported no additional symptoms, such as decreased sense of smell, shortness of breath, chest tightness, frequent nightmares, suspiciousness, or limb numbness. The patient sought medical attention at a local hospital, where she was diagnosed with "Parkinson's disease" and prescribed "Tasud 50 mg, three times daily." Two months ago, she experienced a coma after taking the medication, with no response to external stimuli, and was urgently taken to the local hospital, where her blood glucose level was measured at 1.4 mmol/L. Her condition improved after receiving appropriate symptomatic treatment. She is currently taking "Madopar 125 mg, three times daily" regularly, is able to perform fine motor tasks adequately, and can manage daily activities independently. Since the onset of her illness, she has had a generally stable mental state, with a normal appetite, good sleep, bowel movements every 2-3 days, normal urination, and no significant changes in body weight.")

#Generating Evidence-Based Answers for Medical Order Explanation to Patients
medical_term_explanations, clinical_answer = lins.MOEQA(if_QA=True, if_explanation=True, question="Why do I have ischemic bowel disease?", explain_text="Preliminary Diagnosis: Ischemic Bowel Disease.\nManagement: Instructed patient to rest in bed, avoid stress, keep nil by mouth, provide continuous oxygen inhalation, fluid replacement to maintain water and electrolyte balance, use papaverine hydrochloride to relieve spasms and pain, dilate blood vessels to maintain blood flow, and observe symptoms the next day.", patient_information="Gender: Female, Age: 53 years\nChief Complaint: Admitted for \"recurrent abdominal pain and bloating for over 2 years.\"\nCurrent Illness History: The patient experienced abdominal pain 2 years ago, especially under the xiphoid process, presenting as intermittent dull pain and discomfort, with episodes lasting variable durations, aggravated after a full meal, accompanied by bloating, bitter mouth, fatigue, without cough or sputum, chills, or fever. Local hospital's gastroscopy diagnosed chronic gastritis, treated with oral Zhi Shu Kuang Zhong Capsules, Domperidone Tablets, etc. with symptoms improving occasionally but easily recurring. Four days ago, a broad-based polyp about 0.6 cm in diameter was found in the hepatic flexure and removed with endoscopic clipping, with no abnormalities observed in the rest of the colon and rectum; on the first postoperative day, the patient experienced abdominal cramps and frequent bloody stools.\nPast History: No history of hypertension, diabetes, coronary artery disease; no drug or food allergies, no history of ulcerative colitis or Crohn's disease, no history of hematological diseases.\nPhysical Examination: Pulse 71/min, Respiration 20/min, Blood pressure 120/80 mmHg (1 mmHg=0.133 kPa). Abdomen flat, no gastrointestinal shape or peristaltic wave observed, no abdominal wall varicosities, whole abdomen soft, tenderness under xiphoid and around navel, no rebound tenderness or muscle tension, liver and spleen not palpable below ribs. Murphy sign negative. Whole abdomen without palpable mass, shifting dullness negative, no knocking pain in liver and kidney areas, bowel sounds 4/min.\nAuxiliary Examination: No abnormalities in routine blood tests and coagulation function tests. Colonoscopy: diffuse dark red and purplish-red changes in descending colon and sigmoid colon mucosa, significant swelling with multiple patchy erosions and irregular shallow ulcers, bruising; observation of post-polypectomy site revealed a clip device in place, no bleeding points found; Abdominal enhanced CT: swelling of the descending and sigmoid colon with multiple small blood vessels showing around normally contrasting bowel segments, abdominal vascular CTA showed clear mesenteric artery and major branches, no thrombosis or significant stenosis noted.\nPreliminary Diagnosis: Ischemic Bowel Disease.\nManagement: Instructed patient to rest in bed, avoid stress, keep nil by mouth, provide continuous oxygen inhalation, fluid replacement to maintain water and electrolyte balance, use papaverine hydrochloride to relieve spasms and pain, dilate blood vessels to maintain blood flow, and observe symptoms the next day.")
```
LINS allows users to flexibly choose the LLM, retriever, and retrieval database based on their specific needs. It also supports the creation of personalized local retrieval databases. Detailed tutorials are provided in ![**README_details**](./README_details.md).
