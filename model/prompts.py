
keywords_prompt =  """
# CONTEXT #
The task is to extract comma-separated keywords.

# OBJECTIVE #
Extract keywords in order of importance, without any additional commentary.

# STYLE #
Be concise and direct, avoiding any unnecessary explanations.

# TONE #
Professional and efficient.

# AUDIENCE #
Users seeking a quick and efficient extraction of keywords.

# RESPONSE #
Comma-separated keywords extracted from the document.

"""

num_keywords_prompt = """
# CONTEXT #
The task is to extract comma-separated keywords. The max number of keywords is **number.

# OBJECTIVE #
Extract no more than **number keywords in order of importance, without any additional commentary.

# STYLE #
Be concise and direct, avoiding any unnecessary explanations.

# TONE #
Professional and efficient.

# AUDIENCE #
Users seeking a quick and efficient extraction of keywords.

# RESPONSE #
Comma-separated keywords extracted from the document.
"""


Passage_Relevance_prompt = """
# CONTEXT #
You need to determine if a paragraph of medical knowledge contains the answer to a given question.

# OBJECTIVE #
If the paragraph contains the answer to the question, output "Gold"; if it does not contain the answer, or although the paragraph is related to the question but cannot help to answer it, output "Relevant.

# STYLE #
Provide a direct and concise response without any further explanation or chat.

# TONE #
Neutral

# AUDIENCE #
Users seeking quick and clear verification of medical information.

# RESPONSE #
Output "Gold" or "Relevant" based on whether the knowledge contains the answer to the question.

"""

Guideline_Relevance_prompt = """
# Background #
You will receive a summary of a medical guideline and a medical question. You need to determine whether the guideline corresponding to the summary is relevant to the question.

# Objective #
If the guideline corresponding to the summary may contain the answer to the question, please output "Gold"; if the guideline does not contain the answer, or although related to the question but cannot help answer it, please output "Relevant".

# Style #
Provide a direct and concise response without further explanation or small talk.

# Tone #
Neutral

# Audience #
Users seeking quick and clear verification of medical information.

# Response #
Based on whether the guideline is relevant to the medical question, output "Gold" or "Relevant".
"""

Question_Decomposition_prompt = """
# CONTEXT #
The current question is difficult to answer directly and lacks a clear focus for retrieval.

# OBJECTIVE #
Generate up to four sub-questions that can help gather information to answer the original question. The sub-questions should remain independent from the original question and not rely on information from it.

# STYLE #
Concise and focused on breaking down the main question. The sub-questions should be in the same language as the original question.

# RESPONSE #
List the decomposed sub-questions in a clear and organized manner. Directly list each sub-question without adding any redundant dialogue.

"""


Passage_Coherence_prompt = """
# CONTEXT #
You need to determine whether the generated sentence is consistent with the meaning expressed in the retrieved paragraph.

# OBJECTIVE #
Provide a straightforward assessment of the coherence between the sentence and the paragraph.

# STYLE #
Direct and focused response without any additional explanations.

# TONE #
Neutral

# AUDIENCE #
Users seeking quick evaluation of content consistency.

# RESPONSE #
Either "Conflict", "Coherence", or "Irrelevant" based on the relationship between the generated sentence and the retrieved paragraph.

"""

Self_knowledge_prompt = """
# CONTEXT #
You need to assess whether you can answer questions correctly.

# OBJECTIVE #
Provide an honest assessment of your ability to answer each question correctly.

# STYLE #
Be objective and truthful in your evaluation.

# TONE #
Neutral

# AUDIENCE #
Users seeking accurate self-assessment of your ability to answer questions.

# RESPONSE #
Either "CERTAIN" or "UNCERTAIN" based on your genuine assessment of your ability to answer each question correctly without any chat-based fluff and say nothing else. 

"""


RAG_prompt = """
# CONTEXT #
Refer to the following KNOWLEDGE along with your own understanding to answer the questions presented.

# OBJECTIVE #
Consider the given KNOWLEDGE carefully and provide an accurate response.

# STYLE #
Avoid phrases such as "the retrieved paragraph mentions" or "according to the provided KNOWLEDGE" in your response. Ensure the content is fully and clearly expressed for easy reading.

# TONE #
The response should be as detailed, professional, and objective as possible.

# AUDIENCE #
All users

# RESPONSE #
Incorporate as much content from the KNOWLEDGE as possible in your response, especially data or examples. For each sentence in the response,if the sentence includes content or data from the KNOWLEDGE, or examples, use a citation format "[n]" at the end of the sentence, where n indicates the example number. A sentence can have multiple citations such as "[1][2][3]", but the citations should always appear at the end of the sentence bef

#KNOWLEDGE#

"""


AEBMP_prompt = """
# OBJECTIVE #
You are a trusted assistant to the doctor, and your task is to assist in evidence-based medical practice. Below is the patient's information, the clinical questions posed by the doctor, the corresponding PICO questions, and some evidence retrieved based on the PICO questions. Please combine the information provided and answer the PICO questions to assist the doctor in making the next decision.

# REQUIREMENTS #
Evidence-based medicine requires answers to be based on high-quality evidence and to consider the patient's specific information. Where possible, the answer should include quantitative results and risk assessments. For example, the effectiveness of a treatment intervention can be described through statistical data, such as reducing mortality, decreasing the incidence of complications, etc.

# ANSWER FORMAT #
Based on high-quality evidence, the answer should clearly indicate references to the evidence to allow users to trace the sources. The citation format should be as follows: statement 1[n]. statement 2[m][p]. Where [m][n][p] refer to the evidence numbers.

# PATIENT INFORMATION #
{patient_information}

# CLINICAL QUESTION #
{clinical_question}

# PICO QUESTION #
{PICO_question}

# RETRIEVED EVIDENCE #
{retrieved_evidence}

# ANSWER #
"""

AEBMP_GUIDELINE_prompt = """
# OBJECTIVE #
You are a trusted assistant to the doctor, and your task is to assist in evidence-based medical practice. Below is the patient's information, the clinical questions posed by the doctor, the corresponding PICO questions, and several guidelines retrieved based on the PICO questions. Please combine the provided information and answer the PICO questions to assist the doctor in making the next decision.

# REQUIREMENTS #
Evidence-based medicine requires answers to be based on high-quality evidence and to consider the patient's specific information. Where possible, the answer should include quantitative results and risk assessments. For example, the effectiveness of a treatment intervention can be described through statistical data, such as reducing mortality, decreasing the incidence of complications, etc.
Answer should be the same language as the PICO question.

# ANSWER FORMAT #
The answer should include both recommendations and citation content. The citations should be sourced from the guidelines, and the recommendations should indicate which citations are used in which parts of the text with [n]. The specific answer format (use josn) is as follows:
{{"Recommendation": statement1[1]. statement2[2][3]. statement3. # No restriction on sentence or citation count; adjust as needed
"Citations": ["[1]: citation1", "[2]: citation2", "[3]: citation3"],
"Citation_source_id": [0,1,1] # Indicates which guideline each citation is from}}


# PATIENT INFORMATION #
{patient_information}

# CLINICAL QUESTION #
{clinical_question}

# PICO QUESTION #
{PICO_question}

# GUIDELINES #
{GUIDELINES}

# ANSWER #
"""


PICO_prompt = """
# OBJECTIVE #
You are a reliable assistant to the doctor. Your task is to assist the doctor in generating PICO questions for evidence-based practice. The following provides the patient's information and the clinical question posed by the doctor. Please combine the above information to formulate the corresponding PICO question for evidence-based retrieval.

# REQUIREMENTS #
1. The generated PICO question should be in the same language as the clinical question.
2. Directly generate the PICO question without outputting any other chat content.

# Example clinical question and corresponding PICO question #
1. Clinical question: Should glucocorticoids be used for patients with tuberculous pericarditis?
   PICO question: Can glucocorticoids reduce the mortality risk in adult patients with tuberculous pericarditis?

2. Clinical question: What is the significance of ascitic ADA in diagnosing tuberculous peritonitis?
   PICO question: What is the sensitivity and specificity of ascitic ADA in diagnosing tuberculous peritonitis?

3. Clinical question: What is the risk of tuberculosis in vegetarians?
   PICO question: Is the risk of tuberculosis higher in vegetarians compared to those on a normal diet?

4. Clinical question: Will patients with tuberculous pericarditis develop constrictive pericarditis?
   PICO question: What is the likelihood of future constriction in patients with tuberculous pericarditis?

# PATIENT INFORMATION #
{patient_information}

# CLINICAL QUESTION #
{clinical_question}

# PICO question #
"""




Medical_Entity_Extraction_prompt = """
# Goal #
You need to extract some professional medical entities from the content below labeled as TEXT, to explain them to patients in subsequent tasks.

# Requirements #
1. Extract no more than {MAX_EXTRACTION_NUMBER} entities. If there are many, select up to {MAX_EXTRACTION_NUMBER} important ones.
2. Extract entities in order.
3. The extracted entities should be output in the form of a dict: {{"entity_list":[""]}}.
4. Output only the extracted entities; do not output any other conversation content.
5. Output exactly as they appear in the original text without modifying the expressions or language.


# TEXT #
{TEXT}

# ANSWER #
"""

Medical_Text_Explanation_prompt = """
# Task #
The TEXT contains a part of the patient's medical record, and ENTITY includes the relevant medical terms. Please appropriately combine the content from RETRIEVED EVIDENCE to explain the corresponding ENTITY clearly to the patient.

# Requirements #
1、Provide a scientific explanation in one paragraph. Do not greet or chat with the patient.
2、Use the same language as the patient's medical record.

# TEXT #
{TEXT}

# RETRIEVED EVIDENCE #
{RETRIEVED_EVIDENCE}

# ENTITY #
{ENTITY}

# ANSWER #
"""


Medical_Order_Question_prompt = """
# Task #
PATIENT INFORMATION contains part of the patient's medical record, and QUESTION includes the clinical question raised by the patient. Please combine both to transform the QUESTION into a clear and retrievable query for the next step in evidence retrieval.
Directly output the transformed question.

# Requirements #
1. Clear and concise: Use straightforward language to directly describe the core of the question, avoiding lengthy or complex sentences.
2. Highlight keywords: Include keywords such as disease names or treatment methods in the question.
3. Avoid vague terms: The question should be independent of the electronic medical record. Avoid questions with unclear references like “What is the impact of surgery on future fertility?”, “How heritable is the condition?”, or “What are the long-term side effects of these medications?” Instead, specify exactly which surgery, condition, or medication is being discussed.
4. The language used in the TRANSFORMED QUESTION should be consistent with that in QUESTION.

# PATIENT INFORMATION #
{PATIENT_INFORMATION}

# QUESTION #
{QUESTION}

# TRANSFORMED QUESTION #
"""


Medical_Order_QA_prompt = """
# Task # 
You are a valuable assistant to the patient. PATIENT INFORMATION contains the patient's medical record information, QUESTION includes the clinical question raised by the patient, and RETRIEVED PASSAGES contains some retrieved information. Please combine the patient's medical information and the retrieved information to answer the patient's question.

# Answer Format #
The answer should clearly indicate references to the evidence to allow users to trace the sources. The citation format should be as follows: statement 1[n]. statement 2[m][p], where [m][n][p] refer to the evidence numbers.(There is no need to list specific references under the statements, as the next step in the task will handle that separately.)

# Requirements #
Cite only from the content in RETRIEVED PASSAGES. If RETRIEVED PASSAGES is empty, do not include any citations.

# PATIENT INFORMATION #
{PATIENT_INFORMATION}

# RETRIEVED PASSAGES #
{RETRIEVED_PASSAGES}

# QUESTION #
{QUESTION}

# Answer #
"""

def return_prompts():
    return {
        "keywords_prompt": keywords_prompt,
        "num_keywords_prompt": num_keywords_prompt,
        "Passage_Relevance_prompt": Passage_Relevance_prompt,
        "Guideline_Relevance_prompt": Guideline_Relevance_prompt,
        "Question_Decomposition_prompt": Question_Decomposition_prompt,
        "Passage_Coherence_prompt": Passage_Coherence_prompt,
        "Self_knowledge_prompt": Self_knowledge_prompt,
        "AEBMP_prompt": AEBMP_prompt,
        "AEBMP_GUIDELINE_prompt": AEBMP_GUIDELINE_prompt,
        "PICO_prompt": PICO_prompt,
        "Medical_Entity_Extraction_prompt": Medical_Entity_Extraction_prompt,
        "Medical_Text_Explanation_prompt": Medical_Text_Explanation_prompt,
        "Medical_Order_Question_prompt": Medical_Order_Question_prompt,
        "Medical_Order_QA_prompt": Medical_Order_QA_prompt,
        "RAG_prompt": RAG_prompt
    }