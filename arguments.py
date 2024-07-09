import argparse


def add_model_medlinker_config_args(parser):
    """Model arguments"""
    parser.add_argument("-w", "--medlinker_ckpt_path", type=str, default='./model/generator/Qwen1.5-110B-Chat', help="path to the webqwen checkpoint")
    parser.add_argument("-r", "--retriever_ckpt_path", type=str, default='./model/retriever/bge/bge-m3', help="path to the retriever checkpoint, default to $WEBGLM_RETRIEVER_CKPT")
    
    parser.add_argument("-d", "--device", type=str, default="cuda:1", help="device to run the model, default to cuda")
    
    parser.add_argument("-b", "--filter_max_batch_size", type=int, default=50, help="max batch size for the retriever, default to 50")
    
    parser.add_argument("-s", "--serpapi_key", type=str, default=None, help="serpapi key for the searcher, default to $SERPAPI_KEY")
    parser.add_argument("--searcher", type=str, default="bing", help="searcher to use (serpapi or bing), default to bing")
    
    return parser



def get_medlinker_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description='medlinker')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    
    parser = add_model_medlinker_config_args(parser)
    #parser = add_evaluation_args(parser)
    
    return parser.parse_args()
