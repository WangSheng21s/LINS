from .extracting_by_bs4 import extracting as bs4
from .html2text import html2text

from typing import List, Dict
import re

class Extractor:
    def __init__(self) -> None:
        pass
    
    def _pre_filter(self, paragraphs):
        # sorted_paragraphs = sorted(paragraphs, key=lambda x: len(x))
        # if len(sorted_paragraphs[-1]) < 10:
        #     return []
        ret = []
        for item in paragraphs:
            #print(item)
            item = item.strip()
            item = re.sub(r"\[\d+\]", "", item) 
            #首先判断是中文还是以英文
            if re.search(r"[\u4e00-\u9fa5]", item) == None:#如果是英文
                #如果是英文
                if len(item) < 200:
                    continue
                
            elif re.search(r"[a-zA-Z]", item) == None:#如果是中文
                if len(item) < 30:
                    continue
            
            if len(item) > 2000:
                item = item[:2000] + "..."
            ret.append(item)
        return ret
    
    def extract_by_bs4(self, html) -> List[str]:
        return self._pre_filter(bs4(html))
    
    def extract_by_html2text(self, html) -> List[str]:
        return self._pre_filter(html2text(html).split("\n"))