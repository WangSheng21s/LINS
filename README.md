<h1> LINS: A Professional Medical Q&A Framework for Enhancing Knowledge Privacy and Timeliness</h1>

We developed LINS, a general medical Q&A framework that can seamlessly adapt to any medical field without additional training or fine-tuning. We introduced the Multi-Agent Iterative Retrieval Augmented Generation (MAIRAG) algorithm and the Keyword Extraction Degradation (KED) algorithm to help LINS generate Citation-Based Generative Text (CBGT). LINS achieved state-of-the-art (SOTA) performance in both subjective and objective evaluations on specialized medical datasets. Additionally, LINS supports keyword extraction, retrieval of the latest knowledge, and assists in evidence-based medical practice. It can also easily integrate with local knowledge bases without additional training or fine-tuning. In summary, LINS is a multifunctional, highly professional, privacy-protecting, and up-to-date medical Q&A framework with broad application value in the medical field. It is expected to promote the application and development of large language models in medicine, thereby improving the efficiency of related professionals.

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
pip install -r requirements.txt
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

The Qwen model can be found at [https://huggingface.co/Qwen/Qwen1.5-110B](https://huggingface.co/Qwen/Qwen1.5-110B), the recall model for the retriever at [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3),and the ranking model at [https://huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3). Finally, the T5 model is accessible at [https://huggingface.co/google-t5/t5-11b](https://huggingface.co/google-t5/t5-11b).

## Keyword Extraction

Content for keyword extraction...

## Retrieve Evidence

Content for retrieve evidence...

## Direct Multi-Round Q&A

Content for direct multi-round Q&A...

## Original Retrieval-Augmented Generation

Content for original retrieval-augmented generation...

## Multi-Agent Iterative Retrieval-Augmented Generation

Content for multi-agent iterative retrieval-augmented generation...

## Integrate Local Knowledge Base for Answering

Content for integrate local knowledge base for answering...
