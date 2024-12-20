"""Extract wikihow methods and steps from raw HTMLs"""
import argparse
import glob
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tqdm
from bs4 import BeautifulSoup
from wikihow import Article, ParseError

# Global lock for updating shared variables
lock = threading.Lock()
success_failure = [0, 0]  # [success_count, failure_count]
parsed_articles_batch = []
batch_save = 500


def parse_html(html_path: str) -> dict[str, Any]:
    try:
        article = Article(html_path=html_path)
    except ParseError:
        return {}
    parsed = {}
    parsed["title"] = article.title
    parsed["url"] = article.url
    parsed["intro"] = article.intro
    parsed["methods"] = []
    for method in article.methods:
        method_dict = {"title": None, "steps": []}
        if method.title == "Steps":
            method_dict["title"] = article.title
        else:
            method_dict["title"] = method.title
        for step in method.steps:
            method_dict["steps"].append(f"{step.title} {step.description}")
        parsed["methods"].append(method_dict)

    return parsed


def threaded_parse(html_path):
    parsed_article = parse_html(html_path)
    with lock:
        if not parsed_article:
            success_failure[1] += 1
        else:
            success_failure[0] += 1
            parsed_articles_batch.append(parsed_article)
            if len(parsed_articles_batch) >= batch_save:
                with open(f"{args.output_path}", "a+") as f:
                    for example in parsed_articles_batch:
                        f.write(json.dumps(example) + "\n")
                parsed_articles_batch.clear()


def parallel_main(args):
    paths_to_parse = []
    for entry in os.scandir(args.data_dir):
        html_path = (
            entry.path
            if entry.is_file()
            else next(os.scandir(entry.path)).path
        )
        paths_to_parse.append(html_path)
    print(f"Total number of files: {len(paths_to_parse)}")

    with ThreadPoolExecutor(max_workers=16) as executor:
        # Create a tqdm progress bar
        pbar = tqdm.tqdm(total=len(paths_to_parse))

        # A callback to update the progress bar when a future completes
        def update_pbar(future):
            pbar.update(1)

        # Submit tasks and add done callback
        futures = [
            executor.submit(threaded_parse, path) for path in paths_to_parse
        ]
        for future in futures:
            future.add_done_callback(update_pbar)

        # Ensure all tasks are done before exiting context
        for future in futures:
            future.result()

        pbar.close()

    # Dump any remaining parsed articles
    if parsed_articles_batch:
        with open(args.output_path, "a+") as f:
            for example in parsed_articles_batch:
                f.write(json.dumps(example) + "\n")

    print(f"Success: {success_failure[0]}, Failed: {success_failure[1]}")


def main(args):
    success = 0
    failed = 0
    parsed_articles = []

    paths_to_parse = []
    for entry in os.scandir(args.data_dir):
        html_path = (
            entry.path
            if entry.is_file()
            else next(os.scandir(entry.path)).path
        )
        paths_to_parse.append(html_path)

    print(f"Total number of files: {len(paths_to_parse)}")

    for idx, html_path in enumerate(tqdm.tqdm(paths_to_parse)):
        parsed_artcle = parse_html(html_path)

        if not parsed_artcle:
            failed += 1
            continue
        success += 1

        parsed_articles.append(parsed_artcle)
        if len(parsed_articles) >= batch_save:
            with open(args.output_path, "a+") as f:
                for example in parsed_articles:
                    f.write(json.dumps(example) + "\n")
            parsed_articles.clear()

    with open(args.output_path, "a+") as f:
        for example in parsed_articles:
            f.write(json.dumps(example) + "\n")

    print(f"Success: {success}, Failed: {failed}")


def save_parts_article(data_dir):
    parts_articles = []
    for entry in os.scandir(data_dir):
        html_path = (
            entry.path
            if entry.is_file()
            else next(os.scandir(entry.path)).path
        )
        with open(html_path, "r") as f:
            read_content = f.read()
            soup = BeautifulSoup(read_content, "html.parser")
            methods_html = soup.findAll(
                "div",
                {
                    "class": [
                        "section steps steps_first sticky",
                        "section steps sticky",
                    ]
                },
            )
            if not methods_html:
                raise ParseError
            else:
                method_html = methods_html[0]
                part_or_method = method_html.find(
                    "div", {"class": "method_label"}
                )
                if part_or_method:
                    t = part_or_method.text
                    if t.lower().startswith("part"):
                        parts_articles.append(html_path)
    print(len(parts_articles))

    with open("data/wikihow/parts_articles.json", "w+") as f:
        json.dump(parts_articles, f)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw_wikihow")
    parser.add_argument(
        "--output_path", type=str, default="data/wikihow.jsonl"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = config()
    # remove the output file if it exists
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
    main(args)
