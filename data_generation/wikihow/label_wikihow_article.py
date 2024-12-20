"""Label each wikihow article on whether it is a qualified article or not."""
import glob
import json
import os
import subprocess

import yaml
from dotenv import load_dotenv
from llms.providers.openai_utils import safe_batch_chat_completion

from data_generation.utils import (
    calc_openai_cost,
    reorder_response_file,
)

load_dotenv()


def extract_browswer_data():
    qualified = []
    tot = 0
    for file_name in glob.glob("data/wikihow/web_sep/wh_*.jsonl"):
        with open(file_name, "r") as f:
            for line in f:
                example = json.loads(line)
                title = example["title"]
                _methods = []
                for method in example["methods"]:
                    if method["qualified"]:
                        _methods.append(method)
                        tot += 1
                if len(_methods) > 0:
                    example["methods"] = _methods
                    qualified.append(example)

    print(f"Total qualified examples: {tot}")
    with open("data/wikihow/wikihow.browser.jsonl", "w+") as f:
        for example in qualified:
            f.write(json.dumps(example) + "\n")


def label_by_content(
    file_name: str,
    prompt_file: str,
    prompt_version: str,
    model: str,
    rate_limit: int,
    token_limit: int,
    max_tokens: int,
):
    with open(prompt_file, "r") as f:
        prompt = yaml.safe_load(f)[prompt_version]

    data = []
    with open(file_name, "r") as f:
        for line in f:
            example = json.loads(line)
            data.append(example)

    tot = 0
    request_file = file_name.replace(".jsonl", ".requests.jsonl")
    index_file = file_name.replace(".jsonl", ".index.jsonl")
    response_file = file_name.replace(".jsonl", ".responses.jsonl")
    indexes = []
    with open(request_file, "w+") as f:
        for e_idx, example in enumerate(data):
            title = example["title"]
            for m_idx, method in enumerate(example["methods"]):
                method_title = method["title"]
                method_steps = "\n".join(method["steps"])
                cur_article = f"{title} ({method_title})\n\n {method_steps}"
                cur_title = f"{title} ({method_title})"
                cur_messages = [
                    {"role": "system", "content": prompt["system"]},
                    {
                        "role": "user",
                        "content": prompt["user_message"].replace(
                            "__article__", cur_article
                        )
                        if "__article__" in prompt["user_message"]
                        else prompt["user_message"].replace(
                            "__title__", cur_title
                        ),
                    },
                ]
                cur_body = {
                    "model": model,
                    "messages": cur_messages,
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "top_p": 1.0,
                }
                f.write(json.dumps(cur_body) + "\n")
                tot += 1
                indexes.append((e_idx, m_idx))

    print(f"Total requests: {tot}")

    with open(index_file, "w+") as f:
        json.dump(indexes, f)

    process = subprocess.Popen(
        [
            "python",
            "llms/providers/openai_request_parallel.py",
            "--requests_filepath",
            request_file,
            "--save_filepath",
            response_file,
            "--max_requests_per_minute",
            str(rate_limit),
            "--max_tokens_per_minute",
            str(token_limit),
        ]
    )
    process.wait()
    reorder_response_file(response_file, request_file)
    calc_openai_cost(response_file)


def parse_label_by_content(data_file: str) -> None:
    request_file = data_file.replace(".jsonl", ".requests.jsonl")
    response_file = data_file.replace(".jsonl", ".responses.jsonl")
    index_file = data_file.replace(".jsonl", ".index.jsonl")

    requests = []
    responses = []
    with open(request_file, "r") as f:
        for line in f:
            requests.append(json.loads(line))
    with open(response_file, "r") as f:
        for line in f:
            responses.append(json.loads(line))
    with open(index_file, "r") as f:
        indexes = json.load(f)

    with open(data_file, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    assert len(requests) == len(responses) and len(requests) == len(indexes)

    request_keys = [
        " ".join([x["content"] for x in request["messages"]])
        for request in requests
    ]
    qualified = []
    error = 0
    for response in responses:
        cur_request_key = " ".join(
            [x["content"] for x in response[0]["messages"]]
        )
        index = request_keys.index(cur_request_key)
        e_idx, m_idx = indexes[index]
        try:
            response_str = response[1]["choices"][0]["message"]["content"]
        except TypeError:
            error += 1
            continue
        if "Yes" in response_str:
            cur_data = {
                "task_title": data[e_idx]["title"],
                "method_title": data[e_idx]["methods"][m_idx]["title"],
                "steps": data[e_idx]["methods"][m_idx]["steps"],
            }
            qualified.append(cur_data)
    print(f"Total qualified: {len(qualified)}")
    print(f"Total error: {error}")

    with open(data_file.replace(".jsonl", ".digital.jsonl"), "w+") as f:
        for example in qualified:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    label_by_content(
        "data/wikihow/wikihow.web.jsonl",
        prompt_file="data_generation/wikihow/prompts/classification.yaml",
        prompt_version="v4",
        model="gpt-3.5-turbo",
        rate_limit=8_000,
        token_limit=1_000_000,
        max_tokens=512,
    )
    parse_label_by_content("data/wikihow/wikihow.web.jsonl")
