"""Generate the trajectories from articles

Step 1: make the article concrete"""
import glob
import json
import os
import random
import re
import subprocess

import numpy as np
import tiktoken
import yaml
from dotenv import load_dotenv
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
)

from data_generation.utils import (
    calc_openai_cost,
    check_overwrite,
    get_step_save_file,
    print_generation,
    reorder_response_file,
)

load_dotenv()
random.seed(42)
np.random.seed(42)


def step_1(
    data_file: str,
    model: str,
    rate_limit: int,
    token_limit: int,
    max_tokens: int,
    prompt_file: str,
    prompt_version: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tqdm_disable: bool = True,
) -> None:
    """Convert the article to concrete steps with examples"""
    save_file = get_step_save_file(data_file, 1, 1, "response")
    with open(prompt_file, "r") as f:
        prompt = yaml.safe_load(f)[prompt_version]

    data = []
    with open(data_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    batch = []
    for _, method in enumerate(data):
        task_title = method["task_title"]
        title = method["method_title"]
        if title.lower().startswith("how to"):
            title = title[6:].strip()
        steps = [re.sub(r"\n+", " ", step) for step in method["steps"]]
        steps_str = "\n".join(steps)
        article = f"{task_title} ({title})\n\n{steps_str}"
        cur_messages = [
            # {"role": "system", "content": prompt["system"]},
            {
                "role": "user",
                "content": prompt["user_message"]
                .replace("__article__", article)
                .strip(),
            },
        ]
        # construct the json for the request body
        cur_body = {
            "model": model,
            "messages": cur_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        batch.append(cur_body)

    print(f"Number of examples: {len(batch)}")

    # save the request body to a file
    request_file = f"{data_file.replace('.jsonl', '.step1.request.jsonl')}"

    with open(request_file, "w+") as f:
        for example in batch:
            f.write(json.dumps(example) + "\n")

    check_overwrite(save_file)

    # use subprocess to call the openai api
    process = subprocess.Popen(
        [
            "python",
            "llms/providers/openai_request_parallel.py",
            "--request_url",
            "https://api.openai.com/v1/chat/completions",
            "--api_key",
            os.environ["OPENAI_API_KEY"],
            "--requests_filepath",
            request_file,
            "--save_filepath",
            save_file,
            "--max_requests_per_minute",
            str(rate_limit),
            "--max_tokens_per_minute",
            str(token_limit),
        ]
    )
    process.wait()

    reorder_response_file(save_file, request_file)
    os.remove(request_file)


def parse_step1_result(
    step1_result_file: str,
    max_num_instances: int = 3,
) -> tuple[list[str], list[str]]:
    skip_list = {"skip": 0, "no_summary": 0}
    task_summaries = []
    implementations = []
    with open(step1_result_file, "r") as f:
        for _, line in enumerate(f):
            cur_request = json.loads(line)
            step1_response = cur_request[1]["choices"][0]["message"]["content"]
            # split by ```python
            scenarios = step1_response.split("```python")
            for scenario in scenarios:
                if not scenario.strip():
                    continue

                if "SKIP" in scenario:
                    skip_list["skip"] += 1
                    continue
                for i in range(max_num_instances):
                    scenario = scenario.replace(f"# scenario {i+1}\n", "")
                # extract the task summary from line # task: xxx
                task_summary = re.findall(r"(# task:.*?\n)", scenario)
                if not task_summary:
                    skip_list["no_summary"] += 1
                    continue
                task_summary = task_summary[0].strip()
                scenario = scenario.replace(task_summary + "\n", "")
                scenario = scenario.replace("click_and_type", "type")
                task_summary = task_summary.replace("# task:", "").strip()
                task_summaries.append(task_summary)
                implementations.append("```python\n" + scenario.strip())
    print(skip_list)
    print(f"Number of valid implementations: {len(implementations)}")
    return implementations, task_summaries


def is_qualified_api_call(line_str: str) -> bool:
    valid_actions = [
        "click",
        "hover",
        "click_and_type",
        "key_press",
        "goto",
        "go_back",
        "go_forward",
        "new_tab",
        "close_tab",
        "switch_tab",
        "type",
    ]
    if not line_str.strip():
        return False
    if not any([f"{action}(" in line_str for action in valid_actions]):
        return False
    return True


def split_and_sample(api_call: str, k: int = 1) -> list[tuple[str, str]]:
    """Find a random break point, treat the first part as the past actions, and the next line as the next action"""
    # find lines that are qualified api calls
    lines = api_call.split("\n")
    api_line_nums = [
        i for i, x in enumerate(lines) if is_qualified_api_call(x)
    ]
    if len(api_line_nums) <= 1:
        return []

    samples = []
    # sample spliting points
    sample_line_num_idxes = np.random.choice(
        list(range(1, len(api_line_nums))),
        size=min(k, len(api_line_nums) - 1),
        replace=False,
    )
    for sample_line_num_idx in sample_line_num_idxes:
        sample_line_num = api_line_nums[sample_line_num_idx]
        last_api_call_line_num = api_line_nums[sample_line_num_idx - 1]
        past_actions = (
            "\n".join(lines[: last_api_call_line_num + 1])
            .strip()
            .replace("```python", "")
            .strip()
        )

        next_action = "\n".join(
            lines[last_api_call_line_num + 1 : sample_line_num + 1]
        ).strip()
        samples.append((past_actions, next_action))
    return samples


def sample_pairs_from_api_calls(
    api_call: str, num: int = 3
) -> list[tuple[str, str]]:
    """sample action pairs from the api calls"""
    lines = api_call.split("\n")
    api_line_nums = [x.strip() for x in lines if is_qualified_api_call(x)]
    # generate pairs
    pairs = list(zip(api_line_nums[:-1], api_line_nums[1:]))
    num = min(num, len(pairs))
    sampled_pairs = random.choices(pairs, k=num)
    return sampled_pairs


def step_2(
    data_file: str,
    model: str,
    rate_limit: int,
    token_limit: int,
    max_tokens: int,
    prompt_file: str,
    prompt_version: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    htmls_per_example: int = 1,
) -> None:
    # parse the results from step 1
    step1_result_file = get_step_save_file(data_file, 1, 1, "response")
    api_calls, task_summaries = parse_step1_result(step1_result_file)

    with open(prompt_file, "r") as f:
        prompt = yaml.safe_load(f)[prompt_version]
        round1_prompt = prompt["user_message"][0]

    round1_request_file = get_step_save_file(data_file, 2, 1, "request")
    round1_save_file = get_step_save_file(data_file, 2, 1, "response")
    round1_index_file = get_step_save_file(data_file, 2, 1, "index")
    r1_indexes = []

    # construct the request file
    tot = 0
    with open(round1_request_file, "w+") as f:
        for e_idx, (cur_api_call, cur_task) in enumerate(
            zip(api_calls, task_summaries)
        ):
            for (past_actions, next_action) in split_and_sample(
                cur_api_call, k=htmls_per_example
            ):
                if not past_actions and not next_action:
                    continue

                cur_messages = [
                    {"role": "system", "content": prompt["system"]},
                    {
                        "role": "user",
                        "content": round1_prompt.replace("__task__", cur_task)
                        .replace("__past_actions__", past_actions)
                        .replace("__next_action__", next_action),
                    },
                ]
                # construct the json for the request body
                cur_body = {
                    "model": model,
                    "messages": cur_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
                f.write(json.dumps(cur_body) + "\n")
                tot += 1
                r1_indexes.append(e_idx)

    # save the indexes to a file
    with open(round1_index_file, "w+") as f:
        f.write(json.dumps(r1_indexes))

    print(f"Number of examples: {tot}")

    check_overwrite(round1_save_file)

    process = subprocess.Popen(
        [
            "python",
            "llms/providers/openai_request_parallel.py",
            "--request_url",
            "https://api.openai.com/v1/chat/completions",
            "--api_key",
            os.environ["OPENAI_API_KEY"],
            "--requests_filepath",
            round1_request_file,
            "--save_filepath",
            round1_save_file,
            "--max_requests_per_minute",
            str(rate_limit),
            "--max_tokens_per_minute",
            str(token_limit),
        ]
    )
    process.wait()
    reorder_response_file(round1_save_file, round1_request_file)

    os.remove(round1_request_file)


if __name__ == "__main__":
    steps = [1]
    costs = []
    for file_idx in ["10", "21"]:
        data_file = f"data/wikihow/wikihow_digital_sep/wh_{file_idx}.jsonl"
        if 1 in steps:
            step_1(
                data_file,
                # model='gpt-3.5-turbo-1106',
                model="gpt-4-turbo-2024-04-09",
                rate_limit=1500,
                token_limit=200_000,
                temperature=1.0,
                max_tokens=4096,
                tqdm_disable=False,
                prompt_file="data_generation/wikihow/prompts/prompt_step1.yaml",
                prompt_version="v5_batch",
            )
            costs.append(
                calc_openai_cost(
                    get_step_save_file(data_file, 1, 1, "response")
                )
            )
        if 2 in steps:
            step_2(
                data_file,
                model="gpt-4-turbo-2024-04-09",
                rate_limit=1500,
                token_limit=200_000,
                temperature=1.0,
                max_tokens=4096,
                prompt_file="data_generation/wikihow/prompts/prompt_step2.yaml",
                prompt_version="v6",
                htmls_per_example=5,
            )
            costs.append(
                calc_openai_cost(
                    get_step_save_file(data_file, 2, 1, "response")
                )
            )

        print(costs)
