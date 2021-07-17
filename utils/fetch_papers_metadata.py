import os
import re
import json
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

"""
Get metadata of papers of Scicite using Semantic Scholar API
"""

if __name__ == '__main__':
    # papers urls set
    with open("data/scicite_paper_urls", "r") as fin:
        paper_urls = [line.strip() for line in fin]

    with open("papers_metadata.jsonl", "w") as fout, open("failed.jsonl", "w") as fout_error:
        num_of_data = len(paper_urls)
        with tqdm(total=num_of_data) as pbar:
            idx = 0
            while idx < num_of_data:
                url = paper_urls[idx]
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"}
                    resp = requests.get(url, headers=headers)

                    if resp.status_code == 200:
                        fout.write(json.dumps(json.loads(resp.text)))
                        fout.write("\n")
                        idx += 1
                        pbar.update(1)
                    else:
                        print(
                            f"HTTP code {resp.status_code} {url} Not success!")
                        fout_error.write(json.dumps({
                            "status_code": resp.status_code,
                            "url": url
                        }))
                        fout_error.write("\n")
                        if resp.status_code in [429, 403]:
                            time.sleep(300)
                except:
                    print(f"{url} failed")
                    fout_error.write(json.dumps({
                        "status_code": "",
                        "url": url
                    }))
                    fout_error.write("\n")
                    idx += 1
                    pbar.update(1)
