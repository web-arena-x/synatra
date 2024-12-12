import argparse
import json
import re
from collections import defaultdict

file_list = [
    "PREDICTION_FILE_NAME",
]


def count(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        tot = match = 0
        for line in lines:
            entry = json.loads(line)
            label = entry["label"]
            label = label.replace("hover(element_id", "click(element_id")
            label = (
                label.replace("(element_id", "")
                .replace(")", "")
                .replace(", string", "")
            )
            label = label.replace('="', " [").replace('"', "] ").strip()

            predict = entry["predict"]
            if "(" in predict and ")" in predict:
                predict = (
                    predict.replace("(element_id", "")
                    .replace(")", "")
                    .replace(", string", "")
                )
                predict = (
                    predict.replace('="', " [").replace('"', "] ").strip()
                )
            if predict.replace(" ", "") == label.replace(" ", ""):
                match += 1
            elif "type" in label and re.sub(
                r"[^a-zA-Z0-9]", "", predict
            ) == re.sub(r"[^a-zA-Z0-9]", "", label):
                match += 1
            # else:
            # if len(predict) < 200:
            # print('predict', predict, end=' ||')
            # print('label', label)
            tot += 1
        print(file_name, end=" || ")
        print(match, tot, match / tot)


for file in file_list:
    count(file)
