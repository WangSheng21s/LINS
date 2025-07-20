import argparse


def add_model_medlinker_config_args(parser):
    """Model arguments"""
    parser.add_argument("-r", "--retriever_ckpt_path", type=str, default='./model/retriever/bge/bge-m3', help="path to the retriever checkpoint, default to $WEBGLM_RETRIEVER_CKPT")
    
    parser.add_argument("-d", "--device", type=str, default="cuda:1", help="device to run the model, default to cuda")
    
    #parser.add_argument("-b", "--filter_max_batch_size", type=int, default=50, help="max batch size for the retriever, default to 50")
    
    parser.add_argument("-s", "--serpapi_key", type=str, default=None, help="serpapi key for the searcher, default to $SERPAPI_KEY")
    parser.add_argument("--searcher", type=str, default="bing", help="searcher to use (serpapi or bing), default to bing")
    parser.add_argument("-k", "--llm_keys", type=str, default=None, help="api_key for the LLM")
    parser.add_argument("-n", "--llm_model_name", type=str, default="gpt-4o-mini", help="the llm name")

    ##added
    parser.add_argument('--num_batch', type=int, default=1, help='Number of batches to process at a time.')
    parser.add_argument('--local_data_name', type=str, default='', help='Name of the local data.')
    parser.add_argument('--model_name_list', nargs='+', default=['gpt-4o-mini'], help='List of model names.')
    parser.add_argument('--task_list', nargs='+', default=["medqa_ch", "medqa_us", "medqa_tw"], help='List of task names.')
    parser.add_argument('--save_name', type=str, default='results.jsonl', help='Name of the save file.')
    parser.add_argument('--method_list', nargs='+', default=['chat', 'Original_RAG', 'MAIRAG'], help='Methods to use (e.g., chat, MAIRAG, MAIRAG_options).')
    parser.add_argument('--num_begin', type=int, default=0, help='begin to test')
    parser.add_argument('--num_end', type=int, default=-1, help='end to test')
    parser.add_argument('--retrieval_passages_path_list', nargs='+', default=[], help='path to retrieval passages')
    parser.add_argument('--search_results_path_list', nargs='+', default=[], help='path to search results')
    parser.add_argument('--assistant_model_name', type=str, default='gpt-4o-mini', help='assistant model name')
    parser.add_argument('--assistant_keys', type=str, default=None, help='assistant api key')
    
    return parser



def get_medlinker_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description='medlinker')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    args, unknown = parser.parse_known_args()
    parser = add_model_medlinker_config_args(parser)
    #parser = add_evaluation_args(parser)
    
    return parser.parse_args()
