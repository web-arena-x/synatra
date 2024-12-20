"""Extract the html and get the accessibility tree from WARC file"""
import argparse
import json
import random
import tempfile

import numpy as np
import tiktoken
import tqdm
from browser_env.processors import (
    ObservationHandler,  # webarena handler
)
from playwright._impl._api_types import (
    TimeoutError as playwrightTimeoutError,
)
from playwright.sync_api import expect, sync_playwright

random.seed(87)
np.random.seed(124)

import gzip


def semicolon_to_dict(metadata_list: list[str]):
    """convert meta data to a dictionary"""
    metadata_dict = {}
    for element in metadata_list:
        try:
            index = element.index(":")
            key, value = element[:index], element[index + 2 :]  # skip ': '
            metadata_dict[key] = value
        except:
            continue

    return metadata_dict


def html_to_axtree(html: str) -> str:
    # dump the html to a file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".html"
    ) as f:
        f.write(html)
        html_file = f.name

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            try:
                page.goto(f"file://{html_file}")
            except playwrightTimeoutError:
                browser.close()
                return ""
            cdp_client = page.context.new_cdp_session(page)
            obs_handler = ObservationHandler(
                "text",
                "accessibility_tree",
                "",
                False,
                {"width": 1280, "height": 1080},
            )
            cdp_client.send("Accessibility.enable", {})
            obs = obs_handler.get_observation(page, cdp_client)["text"]
            browser.close()

    return obs


def remove_tabs(chunk: list[str], keep_n: int = 1) -> list[str]:
    # find the number of \t in each line
    num_tabs = [line.count("\t") for line in chunk]
    # find the minimum number of tabs
    # concatenate with the header requires keep at least one tab
    min_tabs = min(num_tabs) - keep_n
    if min_tabs < 0:
        return chunk
    # remove the tabs
    chunk = [line[min_tabs:] for line in chunk]
    return chunk


def sample_subtree(
    ax_tree: str, max_tokens: int = 4096, k: int = 1
) -> list[str]:
    """
    Sample the tree by number of lines per chunk and then randomly sample
    """
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    mean, std = 110, 50
    chunk_size = max(int(np.random.normal(mean, std)), 50)
    lines = ax_tree.split("\n")
    chunks = [
        lines[i : i + chunk_size] for i in range(3, len(lines), chunk_size)
    ]
    if len(chunks) == 0:
        return ""
    # sample a chunk
    chunk_str_list = []
    for _ in range(min(k, len(chunks))):
        chunk = random.choice(chunks)
        chunks.remove(chunk)
        clean_chunk = remove_tabs(chunk)
        clean_chunk_str = "\n".join(lines[:3]) + "\n" + "\n".join(clean_chunk)

        tok_chunk_str = tokenizer.encode(clean_chunk_str)
        chunk_str = tokenizer.decode(tok_chunk_str[:max_tokens])
        chunk_str_list.append(chunk_str)
    return chunk_str_list


def extract_data(warc_file: str, st: int, ed: int) -> None:
    def goto_start(fin: gzip.GzipFile) -> None:
        cur = ""
        cur_st = 0
        while True:
            line = fin.readline()
            cur += line.decode()
            if cur.endswith("\r\n\r\n\r\n"):
                cur_st += 1
                cur = ""
                if cur_st == st:
                    print("Found start")
                    return

    # check how many lines are already extracted
    with open(warc_file.replace(".warc.gz", f".{st}.{ed}.jsonl"), "r") as f:
        _st = st
        st = st + len(f.readlines())
        print(f"Start from {st}th item")

    with gzip.open(warc_file, "r") as fin, open(
        warc_file.replace(".warc.gz", f".{_st}.{ed}.jsonl"), "a"
    ) as fout:
        goto_start(fin)

        cur = ""
        pbar = tqdm.tqdm(total=ed - st)
        cur_ed = st

        for line in fin:
            cur += line.decode()
            if cur.endswith("\r\n\r\n\r\n"):
                metadata = cur.strip().split("\r\n\r\n")[0]
                metadata = semicolon_to_dict(metadata.split("\r\n"))
                html = "\r\n\r\n".join(cur.strip().split("\r\n\r\n")[1:])
                ax_tree = html_to_axtree(html)
                # sample a subtree
                subtree = sample_subtree(ax_tree, k=1)[0]

                if subtree.strip():
                    fout.write(
                        json.dumps(
                            {
                                "metadata": metadata,
                                "html": html,
                                "ax_tree": ax_tree,
                                "subtree": subtree,
                            }
                        )
                        + "\n"
                    )
                cur = ""

                pbar.update(1)
                cur_ed += 1
                if cur_ed == ed:
                    break

        pbar.close()

    print(f"Extracted {ed - st} pages from {warc_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--st", type=int, default=1)
    parser.add_argument("--ed", type=int, default=2)
    args = parser.parse_args()

    extract_data(
        "CLUEWEB_FOLDER_PATH/clueweb/en0000-01.warc.gz", args.st, args.ed
    )
