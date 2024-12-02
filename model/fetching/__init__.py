from .playwright_based_crawl_new import get_raw_pages
from .import playwright_based_crawl_new

import asyncio
    
from typing import List, Dict

from playwright.sync_api import sync_playwright


class Fetcher:
    def __init__(self) -> None:
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def get_rendered_html(self, url: str) -> str:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url)
                page.wait_for_load_state('networkidle')
                rendered_html = page.content()
                page.close()  # 关闭页面
                browser.close()  # 关闭浏览器
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
        return rendered_html

    def fetch(self, urls: List[str]) -> Dict[str, str]:
        """抓取并返回每个URL的渲染后HTML"""
        self.loop.run_until_complete(get_raw_pages(urls, close_browser=True))
        responses = {}
        results = [playwright_based_crawl_new.results[url] for url in urls]
        for url, result in zip(urls, results):
            if ".pdf" in url:
                continue
            if result[1] is not None and "JavaScript enabled" in result[1]:
                text = self.get_rendered_html(url)
                if text is None:
                    continue
                else:
                    responses[url] = text
            else:
                if result[1] is not None:
                    responses[url] = result[1]
        return responses


#import asyncio
#from typing import List, Dict
#from playwright.sync_api import sync_playwright
#from .playwright_based_crawl_new import get_raw_pages
#from . import playwright_based_crawl_new
#from threading import Lock
#
#class Fetcher:
#    def __init__(self) -> None:
#        pass
#
#    def get_rendered_html(self, url: str) -> str:
#        try:
#            with sync_playwright() as p:
#                browser = p.chromium.launch(headless=True)
#                page = browser.new_page()
#                page.goto(url)
#                page.wait_for_load_state('networkidle')
#                rendered_html = page.content()
#                page.close()
#                browser.close()
#        except Exception as e:
#            print(f"Error fetching {url}: {e}")
#            return None
#
#        return rendered_html
#
#    def fetch(self, urls: List[str]) -> Dict[str, str]:
#        """抓取并返回每个URL的渲染后HTML"""
#        # 初始化 results 字典
#        playwright_based_crawl_new.results = {}
#
#        # 在单线程中运行异步任务
#        try:
#            asyncio.run(get_raw_pages(urls, close_browser=True))
#        except Exception as e:
#            print(f"Error running get_raw_pages: {e}")
#            return {}
#
#        responses = {}
#        for url in urls:
#            result = playwright_based_crawl_new.results.get(url)
#            if result is None:
#                print(f"No result for {url}")
#                continue
#
#            if ".pdf" in url:
#                continue
#            if result[1] is not None and "JavaScript enabled" in result[1]:
#                text = self.get_rendered_html(url)
#                if text is None:
#                    continue
#                else:
#                    responses[url] = text
#            else:
#                if result[1] is not None:
#                    responses[url] = result[1]
#        return responses
