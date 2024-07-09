<h1> LINS: A Professional Medical Q&A Framework for Enhancing Knowledge Privacy and Timeliness</h1>

<div align="justify">
We developed LINS, a general medical Q&A framework that can seamlessly adapt to any medical field without additional training or fine-tuning. We introduced the Multi-Agent Iterative Retrieval Augmented Generation (MAIRAG) algorithm and the Keyword Extraction Degradation (KED) algorithm to help LINS generate Citation-Based Generative Text (CBGT). LINS achieved state-of-the-art (SOTA) performance in both subjective and objective evaluations on specialized medical datasets. Additionally, LINS supports keyword extraction, retrieval of the latest knowledge, and assists in evidence-based medical practice. It can also easily integrate with local knowledge bases without additional training or fine-tuning. In summary, LINS is a multifunctional, highly professional, privacy-protecting, and up-to-date medical Q&A framework with broad application value in the medical field. It is expected to promote the application and development of large language models in medicine, thereby improving the efficiency of related professionals. 
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

Prepare SerpAPI Key

If you wants to use SerpAPI to get search results. You need to get a SerpAPI key from [here](https://serpapi.com/).

Then, set the environment variable `SERPAPI_KEY` to your key.

```bash
export SERPAPI_KEY="YOUR KEY"
```

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

# Functional implementation

## Keyword Extraction

```bash
from arguments import get_medlinker_args
from model.modeling_medlinker import load_model

args = get_medlinker_args()
args.medlinker_ckpt_path = "./model/generator/Qwen1.5-110B-Chat"

medlinker = load_model(args)

sentence = "As artificial intelligence (AI) rapidly approaches human-level performance in medical imaging, it is crucial that it does not exacerbate or propagate healthcare disparities. Previous research established AI’s capacity to infer demographic data from chest X-rays, leading to a key concern: do models using demographic shortcuts have unfair predictions across subpopulations? In this study, we conducted a thorough investigation into the extent to which medical AI uses demographic encodings, focusing on potential fairness discrepancies within both in-distribution training sets and external test sets. Our analysis covers three key medical imaging disciplines—radiology, dermatology and ophthalmology—and incorporates data from six global chest X-ray datasets. We confirm that medical imaging AI leverages demographic shortcuts in disease classification. Although correcting shortcuts algorithmically effectively addresses fairness gaps to create ‘locally optimal’ models within the original data distribution, this optimality is not true in new test settings. Surprisingly, we found that models with less encoding of demographic attributes are often most ‘globally optimal’, exhibiting better fairness during model evaluation in new test environments. Our work establishes best practices for medical imaging models that maintain their performance and fairness in deployments beyond their initial training contexts, underscoring critical considerations for AI clinical deployments across populations and sites."

max_num_keywords = -1  #There is no limit on the maximum number of keywords when max_num_keywords <= 0.

keywords = medlinker.keyword_extraction(sentence, max_num_keywords)
```

## Retrieve Evidence

![KED](./assets/KED.png)
Content for retrieve evidence...

## Direct Multi-Round Q&A

Content for direct multi-round Q&A...

## Original Retrieval-Augmented Generation

Content for original retrieval-augmented generation...

## Multi-Agent Iterative Retrieval-Augmented Generation

Content for multi-agent iterative retrieval-augmented generation...

## Integrate Local Knowledge Base for Answering

Content for integrate local knowledge base for answering...

