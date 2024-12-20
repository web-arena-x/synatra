import json
import os


def calc_openai_cost(response_file: str) -> float:
    in_tok_tot = 0
    out_tok_tot = 0

    errors = 0
    if not os.path.exists(response_file):
        print(f"{response_file} does not exist")
        return 0.0

    with open(response_file, "r") as f:
        model_checked = False
        for l_idx, line in enumerate(f):
            cur_request = json.loads(line)
            if not model_checked:
                model = cur_request[0]["model"]
                match model:
                    case "gpt-3.5-turbo" | "gpt-3.5-turbo-1106":
                        in_cost = 0.001
                        out_cost = 0.002
                    case "gpt-4-1106-preview" | "vijay-gpt-4" | "gpt-4-turbo-2024-04-09":
                        in_cost = 0.01
                        out_cost = 0.03
                    case "gpt-4":
                        in_cost = 0.03
                        out_cost = 0.06
                    case "gpt-4o":
                        in_cost = 0.005
                        out_cost = 0.015
                    case _:
                        raise ValueError(f"Unknown model: {model}")
                model_checked = True
            try:
                in_tok_tot += cur_request[1]["usage"]["prompt_tokens"]
                out_tok_tot += cur_request[1]["usage"]["completion_tokens"]
            except TypeError:
                errors += 1

    # calc the cost
    cost = in_tok_tot / 1000 * in_cost + out_tok_tot / 1000 * out_cost
    print(f"Input cost: {in_cost}, Output cost: {out_cost}")
    print(
        f"Input cost: {in_tok_tot / 1000 * in_cost}, Output cost: {out_tok_tot / 1000 * out_cost}"
    )
    print(f"Errors: {errors}")
    return cost


def print_generation(response_file: str, print_query: bool = False) -> None:
    with open(response_file, "r") as f:
        for cur_request in f:
            cur_request = json.loads(cur_request)
            try:
                in_str = cur_request[0]["messages"][0]["content"]
                out_str = cur_request[1]["choices"][0]["message"]["content"]
            except TypeError:
                continue
            if print_query:
                print(in_str)
                print("-----------------------")
            print(out_str)
            print("=======================\n")
    exit()


def get_step_save_file(
    data_file: str, step_idx: int, round_idx: int, kw: str
) -> str:
    return f"{data_file.replace('.jsonl', f'.step{step_idx}.r{round_idx}.{kw}.jsonl')}"


def reorder_response_file(response_file: str, request_file: str) -> None:
    """The response file is saved in the order the request is completed, re-order to the the same as request_file"""
    with open(response_file, "r") as fsave, open(request_file, "r") as freq:
        request_keys = [
            " ".join([x["content"] for x in json.loads(line)["messages"]])
            for line in freq
        ]
        key_used = [False for _ in request_keys]
        responses = [json.loads(line) for line in fsave]
        reordered_responses = [None for _ in responses]

        for response in responses:
            cur_request_key = " ".join(
                [x["content"] for x in response[0]["messages"]]
            )
            # Find the first unused key that matches
            for j, request_key in enumerate(request_keys):
                if cur_request_key == request_key and not key_used[j]:
                    key_used[j] = True
                    reordered_responses[j] = response
                    break

        none_tot = reordered_responses.count(None)
        print(f"{none_tot} requests are not completed")

    with open(response_file, "w+") as f:
        for response in reordered_responses:
            f.write(json.dumps(response) + "\n")


def check_overwrite(file_path: str) -> None:
    if os.path.exists(file_path):
        # flag = input(f"{file_path} already exists, overwrite? (y/n)")
        # if flag.lower() != "y":
        # raise ValueError("File already exists")
        os.remove(file_path)
