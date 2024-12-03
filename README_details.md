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

## Database-Selection

## Local-Retrieval-Database-Construction

## Keyword Extraction

## Retrieve Evidence using KED

## Direct Multi-Round Q&A

## Original Retrieval-Augmented Generation

## Multi-Agent Iterative Retrieval-Augmented Generation

## Link-Eval Computation

## Data available
