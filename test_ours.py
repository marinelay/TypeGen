import json
from pathlib import Path
import libcst as cst
import ast
import re, os, csv
from hityper.typeobject import TypeObject
import shutil
import subprocess
from typet5.type_check import parse_type_str
from collections import defaultdict, deque

def get_diff(original, modified):
    only_in_original = []
    only_in_modified = []

    for orig in original:
        if orig not in modified:
            only_in_original.append(orig)

    for mod in modified:
        if mod not in original:
            only_in_modified.append(mod)

    return only_in_original, only_in_modified


def filter_result(analysis_results):
    # analysis_groups = defaultdict(list)

    # for idx, result in analysis_results.items():
    #     analysis_key = json.dumps(result)
    #     analysis_groups[analysis_key].append((idx, result))

    # if len(analysis_groups) == 1:
    #     print("All analysis results are the same")
    #     return

    # sorted_results = sorted(analysis_groups.items(), key=lambda x: len(json.loadsx[0]), reverse=True)

    result_by_length = dict()

    for idx, result in analysis_results.items():
        data = result_by_length.get(len(result), set())
        data.add((idx, json.dumps(result)))
        result_by_length[len(result)] = data


    sorted_results = sorted(result_by_length.items(), key=lambda x: x[0])

    new_results = []

    for length, results in sorted_results:
        for idx, result in analysis_results.items():
            if length == len(result):
                new_results.append(idx)

    return new_results

def run():
    typegen_path = Path.home() / "TypeGen"
    pred_path = Path("data/predictions")
    testset_path = Path("data/new_testset.json")

    with open(pred_path / "typegen.json", "r") as f:
        typegen_result = json.load(f)

    with open(testset_path, "r") as f:
        testset = json.load(f)

    kind_set = set() 

    total_num = 0
    n = 10
    top_n = [0] * n
    sorted_top_n = [0] * n 

    for repo in typegen_result.keys():
        result_path = Path("analysis_result") / repo.replace("/", "+")
        if not os.path.exists(result_path):
            continue

        if not os.path.exists(result_path / "info.json"):
            continue

        total_num += 1
        analysis_results = {}

        for i in range(0, 10):
            modified_path = result_path / f"modified_{i}.json"

            if not modified_path.exists():
                continue

            with open(modified_path, "r") as f:
                modified = json.load(f)
            
            analysis_results[i] = modified['generalDiagnostics']

        sorted_result = filter_result(analysis_results)
        
        with open(result_path / "info.json", "r") as f:
            info = json.load(f)

        correct = info['correct']
        incorrect = info['incorrect']

        if len(correct) > 0:
            first_correct = correct[0]
            sorted_idx = sorted_result.index(first_correct)

            for i in range(first_correct, n):
                top_n[i] += 1

        for k, idx in enumerate(sorted_result):
            if idx in correct:
                for i in range(k, n):
                    sorted_top_n[i] += 1
                break

    print(f"Total: {total_num}")

    print("---")

    for i in range(0, n):
        print(f"Top {i+1}: {top_n[i]} ==> {sorted_top_n[i]}")

    print("---")

    for i in range(0, n):
        print(f"Top {i+1}: {round(top_n[i] / total_num, 2) * 100} ==> {round(sorted_top_n[i] / total_num, 2) * 100}")

    # print(top_n)
    # print(sorted_top_n)


if __name__ == "__main__":
    run()