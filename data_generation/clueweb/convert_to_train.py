import argparse
import ast
import copy
import glob
import json
import os
import random
import re
from collections import defaultdict
from typing import Any

from transformers import LlamaTokenizer

from data_generation.clueweb.constants import VALID_ACTIONS
from data_generation.utils import get_step_save_file

SEP = "# --------------------"


def print_result(data_file: str, step: int, round: int) -> None:
    save_file = get_step_save_file(data_file, step, round, "response")
    with open(save_file, "r") as f:
        for idx, line in enumerate(f):
            cur_data = json.loads(line)
            print(cur_data[0]["messages"][0]["content"])
            print(cur_data[1]["choices"][0]["message"]["content"])
            print("================================")


def get_task_title(python_code: str) -> str:
    task = python_code.split(SEP)[0].strip()
    task = task.replace("# task:", "").strip()
    return task


def get_prev_actions(python_code: str) -> str:
    prev_action = python_code.split(SEP)[1].strip()
    prev_action = prev_action.replace("# past actions (history)", "").strip()
    return prev_action


def get_next_action(python_code: str) -> str:
    next_action = python_code.split(SEP)[2].strip()
    next_action = next_action.replace("# next action:", "").strip()
    return next_action


def get_observation(text: str) -> str:
    ax_tree = text.split("## The Accessibility Tree")[1].strip()
    return ax_tree


def extract_components(
    data_file: str, result_file: str, save_file: str, write_mode: str = "a"
) -> None:
    tot = 0
    err_cnt = defaultdict(int)

    with open(data_file, "r") as f:
        raw_info = []
        for line in f:
            raw_info.append(json.loads(line))

    with open(result_file, "r") as fin, open(save_file, write_mode) as fout:
        for line in fin:
            cur_data = json.loads(line)
            if "choices" not in cur_data[1]:
                err_cnt["generation_error"] += 1
                continue

            response = cur_data[1]["choices"][0]["message"]["content"]
            python_code_list = re.findall(
                "```python\s(.*?)\s?```", response, re.DOTALL
            )
            task_titles, prev_actions, next_actions = [], [], []
            for python_code in python_code_list:
                try:
                    task_titles.append(get_task_title(python_code))
                    prev_actions.append(get_prev_actions(python_code))
                    next_actions.append(get_next_action(python_code))
                except:
                    err_cnt["key_content_missing"] += 1
                    continue

            request = cur_data[0]["messages"][1]["content"]
            obs = get_observation(request)
            found_raw_obs = False
            # find the original longer observation
            for item in raw_info:
                if obs.replace("\t", "") in item["subtree"].replace("\t", ""):
                    obs = item["subtree"]
                    found_raw_obs = True
                    break
            if not found_raw_obs:
                err_cnt["observation_mismatch"] += 1

            for i in range(
                min(len(task_titles), len(prev_actions), len(next_actions))
            ):
                data = {
                    "task_title": task_titles[i],
                    "prev_actions": prev_actions[i],
                    "next_action": next_actions[i],
                    "ax_tree": obs,
                }
                fout.write(json.dumps(data) + "\n")
                tot += 1
    print("===============================")
    print(f"Summary on parsing the response: [{result_file}]")
    print(f"Successful: {tot}")
    for k, v in err_cnt.items():
        print(f"{k}: {v}")
    invalid_c = sum(err_cnt.values())
    print(
        f"Invalid rate = {invalid_c}/{tot + invalid_c}= {invalid_c/(tot + invalid_c)}"
    )


def extract_arguments(code_str: str) -> dict[str, Any]:
    # Function to extract keys and values from the function call
    def extract_keys_values(node):
        keys_values = {}
        # Handle positional arguments
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Str):
                value = arg.s
            elif isinstance(arg, ast.Num):
                value = arg.n
            else:
                value = None
            keys_values[f"arg{i}"] = value
        # Handle keyword arguments
        for keyword in node.keywords:
            key = keyword.arg
            if isinstance(keyword.value, ast.Str):
                value = keyword.value.s
            elif isinstance(keyword.value, ast.Num):
                value = keyword.value.n
            else:
                value = None
            keys_values[key] = value
        return keys_values

    parsed_code = ast.parse(code_str)
    # Traverse the AST to find the function call
    for node in ast.walk(parsed_code):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in VALID_ACTIONS
        ):
            result = extract_keys_values(node)
            return result


def reorg_data(
    data_file: str, save_file: str, element_id_only: bool = True
) -> None:
    tot = 0
    with open(data_file, "r") as in_file:
        err_dict = defaultdict(int)
        for line in in_file:
            try:
                orig_data = json.loads(line)
                data = {}
                actn_type = "0"

                data["task"] = orig_data["task_title"]
                data["axt"] = orig_data["ax_tree"]
                prev_actions = orig_data["prev_actions"]
                prev_actn_lines = prev_actions.split("\n")
                prev_actn_lines = [x for x in prev_actn_lines if len(x) > 1]

                last_step_cnt = 0
                actn_type = "2"
                for i in range(len(prev_actn_lines)):
                    if (
                        "#" in prev_actn_lines[i]
                        and "step" in prev_actn_lines[i]
                        and ":" in prev_actn_lines[i]
                        and "(" not in prev_actn_lines[i]
                    ):
                        # print(prev_actn_lines[i])
                        last_step_cnt = re.search(
                            r"step\s+(\d+)", prev_actn_lines[i]
                        ).group(1)
                data["prev_actions"] = "\n".join(prev_actn_lines)

                actn = orig_data["next_action"]
                axt_nodeid = 0
                if (
                    "click" in actn
                    or "hover" in actn
                    or "click_and_type" in actn
                ):
                    actn_type = "3"
                    axt_nodeid = re.search(r"(\d+)\)", actn).group(1)
                    # print(axt_nodeid)
                actn_type = "none"
                data["axt_nodeid"] = axt_nodeid

                subtask_line = ""
                actn_comment = ""
                actn_only_line = ""
                actn_lines = actn.split("\n")
                actn_summary = actn_lines[-1]
                for line in actn_lines:
                    if (
                        "click_and_type(" in line
                        or "click(" in line
                        or "hover(" in line
                    ):
                        actn_only_line = line
                    if (
                        "# step" in line
                        and ":" in line
                        and "# step summary:" not in line
                    ):
                        actn_comment = line
                    if "sub-task" in line:
                        subtask_line = line

                # reorg the line to be comment, action, action summary
                if "click_and_type" in actn_only_line:
                    actn_type = "click_and_type"
                    arguments = extract_arguments(actn_only_line)
                    for v in ["content", "arg1"]:
                        if v in arguments:
                            type_content = arguments[v]
                            break
                    for v in ["element", "arg0"]:
                        if v in arguments:
                            type_node_descpt = arguments[v]
                            break

                    if subtask_line:
                        actn = subtask_line + "\n"
                    else:
                        actn = ""

                    if element_id_only:
                        actn += (
                            actn_comment
                            + "\n"
                            + f'type(element_id="{str(axt_nodeid)}",string="{type_content}")'
                            + "\n"
                            + actn_summary
                        )
                    else:
                        actn += (
                            actn_comment
                            + "\n"
                            + f'type(element="{type_node_descpt}", element_id="{str(axt_nodeid)}",string="{type_content}")'
                            + "\n"
                            + actn_summary
                        )

                elif "click" in actn_only_line:
                    actn_type = "click"
                    arguments = extract_arguments(actn_only_line)
                    for v in ["element", "arg0"]:
                        if v in arguments:
                            click_node_descpt = arguments[v]
                            break

                    if subtask_line:
                        actn = subtask_line + "\n"
                    else:
                        actn = ""
                    if element_id_only:
                        actn += (
                            actn_comment
                            + "\n"
                            + f'click(element_id="{str(axt_nodeid)}")'
                            + "\n"
                            + actn_summary
                        )
                    else:
                        actn += (
                            actn_comment
                            + "\n"
                            + f'click(element="{click_node_descpt}", element_id="{str(axt_nodeid)}")'
                            + "\n"
                            + actn_summary
                        )

                elif "hover" in actn_only_line:
                    actn_type = "hover"
                    arguments = extract_arguments(actn_only_line)
                    for v in ["element", "arg0"]:
                        if v in arguments:
                            hover_node_descpt = arguments[v]
                            break
                    if subtask_line:
                        actn = subtask_line + "\n"
                    else:
                        actn = ""
                    if element_id_only:
                        actn += (
                            actn_comment
                            + "\n"
                            + f'hover(element_id="{str(axt_nodeid)}")'
                            + "\n"
                            + actn_summary
                        )
                    else:
                        actn += (
                            actn_comment
                            + "\n"
                            + f'hover(element="{hover_node_descpt}", element_id="{str(axt_nodeid)}")'
                            + "\n"
                            + actn_summary
                        )

                actn = actn.replace("stop(answer: ", "stop(answer=")
                actn = actn.replace("stop(answer:", "stop(answer=")
                data["next_action"] = actn.replace("# next action\n", "")
                with open(save_file, "a") as jsonl_file:
                    tot += 1
                    jsonl_file.write(json.dumps(data) + "\n")

            except:
                err_dict[actn_type] += 1
                pass
    print("===============================")
    print("Errors durning re-organize the content")
    for k, v in err_dict.items():
        print(f"{k}: {v}")
    invalid_c = sum(err_dict.values())
    print(
        f"Invalid rate: {invalid_c}/{tot + invalid_c} = {invalid_c/(tot + invalid_c)}"
    )


def parse_to_code(
    data_file: str,
    save_file: str,
    shuffle: bool = True,
    element_id_only: bool = True,
):
    data = []
    with open(data_file, "r") as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)

    err_cnt = defaultdict(int)

    id = 0
    json_list = []

    for entry in data:
        json_dict = {}
        intent = entry["task"]
        next_actn = entry["next_action"]

        # node id does not exist
        if f"[{entry['axt_nodeid']}]" not in entry["axt"] and (
            "click" in next_actn or "type" in next_actn or "hover" in next_actn
        ):
            err_cnt["nodeid not in tree"] += 1
            continue

        axt = entry["axt"].replace("RootWebArea", "")
        website = ""
        prev_actn = entry["prev_actions"]
        prev_actn_lines = prev_actn.split("\n")
        bad_histroy = False
        bad_line = ""
        # clean the data
        for i in range(len(prev_actn_lines)):
            # copy the template
            if "<" in prev_actn_lines[i] and ">" in prev_actn_lines[i]:
                bad_histroy = True
                bad_line = prev_actn_lines[i]
                err_cnt["<>"] += 1
                break
            # omit details
            if "..." in prev_actn_lines[i] and len(prev_actn_lines[i]) < 10:
                bad_histroy = True
                bad_line = prev_actn_lines[i]
                err_cnt["..."] += 1
                break
            # history has invalid actions
            if i > 0 and "# step " in prev_actn_lines[i - 1]:
                valid_action = False
                for action in VALID_ACTIONS:
                    if action in prev_actn_lines[i]:
                        valid_action = True
                        break
                if not valid_action:
                    bad_line = prev_actn_lines[i]
                    # print(bad_line)
                    err_cnt["invalid_action_in_history"] += 1
                    bad_histroy = True
                    break
            if (
                "#" not in prev_actn_lines[i]
                and "step" not in prev_actn_lines[i]
            ):
                if element_id_only:
                    if (
                        "click" in prev_actn_lines[i]
                        or "hover" in prev_actn_lines[i]
                        or "click_and_type" in prev_actn_lines[i]
                    ):
                        rand_id = random.randint(1, 10000)
                        if "click_and_type" in prev_actn_lines[i]:
                            try:
                                prev_actn_lines[i] = prev_actn_lines[
                                    i
                                ].replace("content=", "")
                                type_content = (
                                    re.search(
                                        r",\s*([^,]+)\s*", prev_actn_lines[i]
                                    )
                                    .group(1)
                                    .strip()
                                    .strip(",")
                                )
                                type_content = re.sub(
                                    r"[^a-zA-Z ]", "", type_content
                                ).strip()
                                prev_actn_lines[
                                    i
                                ] = f'type(element_id="{str(rand_id)}",string="{type_content}")'
                            except:
                                # print('----')
                                prev_actn_lines[i] = ""
                        elif "click" in prev_actn_lines[i]:
                            prev_actn_lines[
                                i
                            ] = f'click(element_id="{str(rand_id)}")'
                        elif "hover" in prev_actn_lines[i]:
                            prev_actn_lines[
                                i
                            ] = f'hover(element_id="{str(rand_id)}")'
                else:
                    if "click_and_type" in prev_actn_lines[i]:
                        prev_actn_lines[i] = prev_actn_lines[i].replace(
                            "click_and_type(", "type("
                        )
        if bad_histroy:
            continue
        prev_actn = "\t" + "\n\t".join(
            [line for line in prev_actn_lines if len(line) > 0]
        )

        prompt_str = f'''"""You are given an observation of a web page, an objective and past actions, your goal is to generate the next action given the current web page"""

# website
website = "{website}"

# observation of the current web page
observation = """{axt}"""

# objective
objective = "{intent}"

# past actions
def solve():
{prev_actn}'''
        json_dict = {}
        json_dict["id"] = id
        id += 1
        json_dict["conversations"] = [
            {"from": "human", "value": prompt_str},
            {
                "from": "gpt",
                "value": next_actn,
            },
        ]
        json_list.append(
            {
                "prompt": copy.deepcopy(
                    json_dict["conversations"][0]["value"]
                ),
                "response": copy.deepcopy(
                    json_dict["conversations"][1]["value"]
                ),
            }
        )

    if shuffle:
        random.shuffle(json_list)

    with open(save_file, "w+") as f:
        json.dump(json_list, f, indent=4)

    print("===============================")
    print("Errors durning parsing to code")
    for k, v in err_cnt.items():
        print(f"{k}: {v}")

    invalid_c = sum(err_cnt.values())
    print(
        f"Invalid rate: {invalid_c}/{len(data) + invalid_c} = {invalid_c/(len(data) + invalid_c)}"
    )
    print("================================")
    print(f"Number of data: {len(json_list)}")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON input")
    parser.add_argument(
        "--data_files",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--response_files",
        nargs="+",
        required=True,
    )

    parser.add_argument("--save_file_prefix", type=str, required=True)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    assert len(args.data_files) == len(
        args.response_files
    ), "Number of data files and response files should be the same"
    return args


def main(args: argparse.Namespace) -> None:
    base_folder = "./data/clueweb"

    save_file_s1 = f"{base_folder}/{args.save_file_prefix}.extracted.jsonl"
    save_file_s2 = f"{base_folder}/{args.save_file_prefix}.reorg.jsonl"
    save_file_s3 = f"{base_folder}/{args.save_file_prefix}.code.jsonl"

    data_files = [f"{base_folder}/{f}" for f in args.data_files]
    response_files = [f"{base_folder}/{f}" for f in args.response_files]

    for save_file in [save_file_s1, save_file_s2, save_file_s3]:
        if os.path.exists(save_file):
            os.remove(save_file)

    for data_file, response_file in zip(data_files, response_files):
        extract_components(data_file, response_file, save_file_s1, "a")
    reorg_data(save_file_s1, save_file_s2)
    parse_to_code(save_file_s2, save_file_s3, shuffle=args.shuffle)


if __name__ == "__main__":
    args = config()
    main(args)
