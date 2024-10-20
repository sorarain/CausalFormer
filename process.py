import os
import json

SEARCH_PATH = "/root/autodl-tmp/users/hyb/recommend/recpJPQ/ml-1m_search"
folder_list = os.listdir(SEARCH_PATH)

result = {}

for folder in folder_list:
    if not "use_causal=True" in folder:
        continue
    alpha = float(folder[21:29])
    p_lambda = float(folder[-8:])
    file_list = os.listdir(os.path.join(SEARCH_PATH, folder))
    for file in file_list:
        if not ".txt" in file or "args" in "file":
            continue
        with open(os.path.join(SEARCH_PATH,folder,file),"r") as f:
            for line in f.readlines():
                if "epoch" in line or (not "600" in line):
                    continue
                result[f"{alpha}, {p_lambda}"] = float(line.split("(")[-1].split(")")[0].split(",")[0])

result_str = json.dumps(result)

with open("process.json","w", encoding='utf-8') as f:
    f.write(result_str)