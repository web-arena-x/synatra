import collections
import glob
import json
import os
import re
import uuid

from data_vis import Step, Trajectory

from data_generation.utils import get_step_save_file

SAVE_PATH = "data/to_local"
REL_STEP_SAVE_PATH = "./steps"
INTERACTIVE_TAG = 'id="next-action-target-element"'


def visualize_trajectories(
    data_file: str, result_step_idx: int, result_round_idx: int
) -> None:
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    step2_index_file = get_step_save_file(
        data_file, result_step_idx, result_round_idx, "index"
    )
    with open(step2_index_file, "r") as f:
        step2_index = json.load(f)

    step2_response_file = get_step_save_file(
        data_file, result_step_idx, result_round_idx, "response"
    )
    with open(step2_response_file, "r") as f:
        step2_response = [json.loads(line) for line in f.readlines()]

    idx_to_response = collections.defaultdict(list)
    for idx, response in zip(step2_index, step2_response):
        idx_to_response[idx].append(response)

    for e_idx, cur_data in enumerate(data):
        intent = f"{cur_data['task_title']} ({cur_data['title']})"
        # replace special character with _
        _intent = re.sub(r"[^a-zA-Z]+", "_", intent)
        responses = idx_to_response[e_idx]
        traj = Trajectory(intent=intent)
        for s_idx, step in enumerate(responses):
            # extratc the action from the request
            user_request = [
                x for x in step[0]["messages"] if x["role"] == "user"
            ][0]
            step_request = user_request["content"]
            action1 = re.findall("past action: (.+)", step_request)[0]
            action2 = re.findall("next action: (.+)", step_request)[0]
            # extract the observation
            observation = (
                re.search(
                    "```(html)?(.*?)```",
                    step[1]["choices"][0]["message"]["content"],
                    re.DOTALL,
                )
                .group(2)
                .strip()
            )
            step = Step(
                observation=observation,
                action=action2,
                history=[action1],
                id=f"{_intent}_{s_idx}",
            )
            # save step observation to a file for iframe
            with open(f"{SAVE_PATH}/{_intent}_{s_idx}.html", "w") as f:
                f.write(step.observation)
            traj.add_step(step)

        html = traj.to_html()

        with open(f"{SAVE_PATH}/{_intent}.final.html", "w") as f:
            f.write(html)


def get_task_title(text: str) -> str:
    match = re.findall("task: (.+)", text)
    if match:
        task = match[0]
    else:
        task = ""
    return task


def get_prev_actions(text: str) -> str:
    python_code = re.findall("```python\s(.*?)\s?```", text, re.DOTALL)
    prev_actions = python_code[0]
    return prev_actions


def get_next_action(text: str) -> str:
    python_code = re.findall("```python\s(.*?)\s?```", text, re.DOTALL)
    next_action = python_code[1]
    return next_action


def get_observation(text: str) -> str:
    match = re.search("```(html)?(.*?)```", text, re.DOTALL)
    if match:
        obs = match.group(2).strip()
    else:
        obs = ""
    return obs


def get_cot(text: str) -> str:
    match = text.split("```")[-1].strip()
    return match


def visualize_step_wise(data_file: str, data_per_file: int = 20):
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    traj = Trajectory(intent="Overall")
    errors = {"miss_tag": 0}
    tot = 0
    node_num = []
    for item in data:
        request = item[0]["messages"][1]["content"]
        # get the task, past actions and the next action
        task = "Task: "
        gen_task_match = get_task_title(request)
        if gen_task_match:
            task += gen_task_match

        prev_actions = get_prev_actions(request)
        next_action = get_next_action(request)

        response = item[1]["choices"][0]["message"]["content"]
        gen_obs_match = get_observation(response)

        cot = get_cot(response)
        next_action = f"CoT: {cot}\n{next_action}"

        if INTERACTIVE_TAG not in gen_obs_match:
            continue

        step = Step(
            observation=gen_obs_match,
            action=next_action,
            history=[task, prev_actions],
            id=f"{traj.id}.{len(traj) + 1}",
            save_path=REL_STEP_SAVE_PATH,
        )
        traj.add_step(step)
        # save step observation to a file for iframe
        with open(
            f"{SAVE_PATH}/{REL_STEP_SAVE_PATH}/{step.id}.html", "w"
        ) as f:
            f.write(step.observation)

        node_num.append(step.html_complexity)
        tot += 1

        if len(traj) >= data_per_file:
            html = traj.to_html()
            with open(f"{SAVE_PATH}/{traj.id}.final.html", "w") as f:
                f.write(html)
            traj = Trajectory(intent="Overall")

    print(f"Toal {tot} steps")
    print(f"Average node number: {sum(node_num) / len(node_num)}")
    for key, value in errors.items():
        print(f"{key}: {value}")


def save_data(data_file: str, save_file: str, write_mode: str = "a") -> None:
    tot = 0
    skip = 0
    with open(data_file, "r") as fin, open(save_file, write_mode) as fout:
        for line in fin:
            item = json.loads(line)
            request = item[0]["messages"][1]["content"]
            task_title = get_task_title(request)
            prev_actions = get_prev_actions(request)
            next_action = get_next_action(request)
            response = item[1]["choices"][0]["message"]["content"]
            gen_obs = get_observation(response)
            cot = get_cot(response)
            if INTERACTIVE_TAG not in gen_obs:
                skip += 1
                continue
            save_item = {
                "task": task_title,
                "prev_actions": prev_actions,
                "next_action": next_action,
                "html": gen_obs,
                "cot": cot,
            }
            fout.write(json.dumps(save_item) + "\n")
            tot += 1
    print(f"Total: {tot}, skip: {skip}")


if __name__ == "__main__":
    save_file = "FILE_TO_SAVE.jsonl"
    # remove the save_file if it exists
    if os.path.exists(save_file):
        os.remove(save_file)

    for file_idx in ["00", "01", ...]:  # index of files to process
        data_file = f"PATH_TO_WIKIHOW_FILE/wh_{file_idx}.jsonl"
        result_file = get_step_save_file(data_file, 2, 1, "response")
        save_data(result_file, save_file, "a")
        # visualize_step_wise(result_file)
