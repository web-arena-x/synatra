import json
import os
import random
import subprocess

import numpy as np
import yaml
from dotenv import load_dotenv

random.seed(3367)

from data_generation.clueweb.constants import (
    M2W_CNT,
    MEAN_SUBTASK_SIZE,
    MEAN_SUBTREE_WINDOW,
    MIN_SUBTREE_LENGTH,
    STD_SUBTASK_SIZE,
    STD_SUBTREE_WINDOW,
    TYPE_ROLES,
)
from data_generation.clueweb.extract_data import remove_tabs
from data_generation.utils import (
    calc_openai_cost,
    check_overwrite,
    get_step_save_file,
    print_generation,
    reorder_response_file,
)

load_dotenv()


def generate_history_prompt_template(
    history_length: int, mean_subtask_size: int, std_subtask_size: int
):
    ed = 0
    sub_task_positions = []
    while ed < history_length:
        sub_task_positions.append(ed)
        while True:
            gap = int(random.gauss(mean_subtask_size, std_subtask_size))
            if gap != 0:
                break
        ed += gap
    sub_task_positions = sub_task_positions[::-1]
    sub_task_template = "# sub-task {0}: <sub-task description>"
    action_step_template = """# step {0}: <step description>
<action>"""
    template = []
    sub_task_cntr = 0
    for i in range(history_length):
        if sub_task_positions and i == sub_task_positions[-1]:
            sub_task_cntr += 1
            template.append(sub_task_template.format(sub_task_cntr))
            sub_task_positions.pop()
        template.append(action_step_template.format(i + 1))
    template = "\n".join(template)
    return template


def generate_template_for_batch(
    sampled_task_indexes: list[int],
    history_lengths: list[int],
    prompt_file: str,
    batch_template_version: str,
):
    with open(prompt_file, "r") as f:
        batch_template = yaml.safe_load(f)[batch_template_version]["message"]

    template = ""
    for i in range(len(sampled_task_indexes)):
        history_template = generate_history_prompt_template(
            history_lengths[i],
            mean_subtask_size=MEAN_SUBTASK_SIZE,
            std_subtask_size=STD_SUBTASK_SIZE,
        )
        next_action_subtask_start = (
            "# sub-task <index>: <sub-task description>"
            if history_lengths[i] == 0 or random.randint(0, 10) < 2
            else ""
        )
        template += (
            batch_template.replace(
                "__LIST_task_index__",
                " ".join(f"#{i}" for i in sampled_task_indexes),
            )
            .replace("__sample_task_index__", str(sampled_task_indexes[i]))
            .replace("__history_template__", str(history_template))
            .replace(
                "__next_is_subtask_start__", str(next_action_subtask_start)
            )
            + "\n"
        )
    return template


def sample_subtree(subtree: str) -> str:
    # print(subtree)
    n = int(random.gauss(MEAN_SUBTREE_WINDOW, STD_SUBTREE_WINDOW))
    if (
        len(subtree.split("\n")) < MIN_SUBTREE_LENGTH
        or len(subtree.split("\n")) - n + 1 < 1
    ):
        return ""
    start = random.randint(0, len(subtree.split("\n")) - n + 1)
    end = start + n - 1
    sampled_subtree = "\n".join(
        remove_tabs(subtree.split("\n")[start:end], keep_n=0)
    )
    return sampled_subtree


def sample_subtree_type(subtree: str) -> str:
    if len(subtree.split("\n")) < MIN_SUBTREE_LENGTH:
        return ""

    tries = 5
    while tries > 0:
        n = int(random.gauss(MEAN_SUBTREE_WINDOW, STD_SUBTREE_WINDOW))
        if len(subtree.split("\n")) - n + 1 < 1:
            tries -= 1
            continue
        start = random.randint(0, len(subtree.split("\n")) - n + 1)
        end = start + n - 1
        sampled_subtree = "\n".join(subtree.split("\n")[start:end])
        if any(role in sampled_subtree for role in TYPE_ROLES):
            sampled_subtree = "\n".join(
                remove_tabs(sampled_subtree.split("\n"), keep_n=0)
            )
            return sampled_subtree
        else:
            tries -= 1
            continue
    return ""


def sample_type_element_id(subtree: str) -> str:
    possible_ids = []
    for node in subtree.split("\n"):
        node = node.strip()
        if len(node.split()) < 2:
            continue
        role = node.split()[1].strip()
        if role in TYPE_ROLES:
            id = node.split()[0].strip()[1:-1]
            # upsample non search box elements
            if role != "searchbox":
                possible_ids.extend([id] * 2)
            else:
                possible_ids.append(id)
    if not possible_ids:
        return ""
    return random.choice(possible_ids)


def main(
    data_file: str,
    model: str,
    rate_limit: int,
    token_limit: int,
    max_tokens: int,
    prompt_file: str,
    prompt_version: str,
    batch_template_version: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_of_task_categories: int = 10,
    num_of_concrete_tasks: int = 4,
    mode_tag: str = "",
    sample_subtree_fn=sample_subtree,
    sample_interactive_elements_fn=None,
) -> None:
    """Convert the article to concrete steps with examples"""
    save_file = get_step_save_file(
        data_file, 1, 1, "response" if not mode_tag else f"{mode_tag}_response"
    )
    with open(prompt_file, "r") as f:
        prompt = yaml.safe_load(f)[prompt_version]

    data = []
    with open(data_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    batch = []
    for _, element in enumerate(data):
        subtree = sample_subtree_fn(element["subtree"])
        if not subtree:
            continue

        # interactive element
        interative_element_id = ""
        if sample_interactive_elements_fn:
            interative_element_id = sample_interactive_elements_fn(subtree)
            if not interative_element_id:
                continue

        sampled_task_indexes = random.sample(
            range(1, num_of_task_categories + 1), num_of_concrete_tasks
        )
        sampled_task_indexes.sort()
        sum_val = sum(M2W_CNT)
        discrete_dist = [x / sum_val for x in M2W_CNT]

        history_lengths = []
        for i in range(num_of_concrete_tasks):
            history_lengths.append(
                np.random.choice(
                    np.arange(0, len(discrete_dist)), p=discrete_dist
                )
            )

        task_index_history_length = "; ".join(
            [
                f"task #{i} with roughly {j} past actions"
                for i, j in zip(sampled_task_indexes, history_lengths)
            ]
        )
        batch_action_template = generate_template_for_batch(
            sampled_task_indexes,
            history_lengths,
            prompt_file,
            batch_template_version,
        )

        cur_messages = [
            {"role": "system", "content": prompt["system"]},
            {
                "role": "user",
                "content": prompt["user_message"]
                .replace("__ax_tree__", subtree)
                .replace("__num_of_tasks__", str(num_of_task_categories))
                .replace(
                    "__LIST_task_index_history_length__",
                    task_index_history_length,
                )
                .replace("__batch_action_template__", batch_action_template)
                .replace(
                    "__LIST_task_index__",
                    " ".join(f"#{i}" for i in sampled_task_indexes),
                )
                .replace("__element_id__", interative_element_id)
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
        # for example in batch[5:7]:
        for example in batch:
            f.write(json.dumps(example) + "\n")
    check_overwrite(save_file)
    process = subprocess.Popen(
        [
            "python",
            "./../../llms/providers/openai_request_parallel.py",
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


if __name__ == "__main__":
    data_files = ["./data/WEB_SNAPSHOTS.jsonl"]
    mode = 2

    mode_tag = ""
    prompt_file = "./prompts/prompt.yaml"
    for data_file in data_files:
        main(
            data_file,
            model="model_name",
            rate_limit=10_000,
            token_limit=180_000,
            temperature=1.0,
            max_tokens=4096,
            prompt_file=prompt_file,
            # prompt_version="all_in_one_v6",
            # batch_template_version="batch_template_v1",
            # prompt_version="all_in_one_v6_type_stop_empty",
            # batch_template_version="batch_template_v1_stop_empty",
            prompt_version="all_in_one_v6_type_stop_ans",
            batch_template_version="batch_template_v1_stop_ans",
            # prompt_version="all_in_one_v6_type",
            # batch_template_version="batch_template_v1_type",
            num_of_task_categories=8,
            num_of_concrete_tasks=5,
            # sample_subtree_fn=sample_subtree_type,
            # sample_interactive_elements_fn=sample_type_element_id,
        )
        c1 = calc_openai_cost(get_step_save_file(data_file, 1, 1, "response"))
        print(f"Cost: {c1}")
