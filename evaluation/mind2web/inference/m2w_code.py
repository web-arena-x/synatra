import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import openai
import transformers
from openai import OpenAI


def generate_from_openai_chat_api_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:1528/v1"
    client = OpenAI(
        api_key=openai.api_key,
        base_url=openai.api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    stream = False
    message = messages[-1]["content"]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        stream=stream,
        temperature=temperature,
    )

    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)
    response_message = completion.choices[0].message.content
    return response_message


def get_result_data(file_name, model_name):
    err_cnt_dict = defaultdict(int)
    result_data = []
    with open(f"../data/{file_name}", "r") as f:
        data = json.load(f)
        result_data = []
        for entry in data:
            result_data.append(
                {
                    "prompt": entry["prompt"],
                    "label": entry["response"].split("\n")[-1],
                }
            )
        return result_data


def query_model(entry, file_name, model_name):
    err_cnt_dict = defaultdict(int)
    start_time = time.time()
    current = entry["prompt"]

    message = [{"role": "system", "content": "You are a helpful assistant"}]
    message.append({"role": "user", "content": current})

    tknzr = transformers.AutoTokenizer.from_pretrained("MODEL_PATH")
    tot = 0
    for d in message:
        tot += len(tknzr.encode(d["role"]))
        tot += len(tknzr.encode(d["content"]))
        tot += 10
    if tot > 3800:
        print(tot)
        err_cnt_dict["prompt>3800"] += 1
        return

    response = generate_from_openai_chat_api_completion(
        messages=message,
        model="placeholder",
        temperature=0,
        max_tokens=4096 - tot,
        top_p=0.1,
        context_length=4096,
    )

    action = response.split("\n")[-2].strip()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f" run time: {execution_time} seconds.")
    print(err_cnt_dict)
    with open(
        file_name.replace(
            ".json", f"_{model_name}_prediction_codeprompt.jsonl"
        ),
        "a",
    ) as f_of:
        json_line = json.dumps({"predict": action, "label": entry["label"]})
        f_of.write(f"{json_line}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    parser.add_argument("model_name", type=str)

    args = parser.parse_args()
    result_data = get_result_data(args.file_name, args.model_name)
    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                query_model,
                result_data,
                repeat(args.file_name),
                repeat(args.model_name),
            )
        )


if __name__ == "__main__":
    main()
