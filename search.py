import os
import time
import torch
import argparse

from model import CauseFormer
from utils import *

import json

from train_causal_function import train


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=200, type=int)
parser.add_argument("--hidden_units", default=64, type=int)
parser.add_argument("--num_blocks", default=4, type=int)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=False, type=str2bool)
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--use_causal", default=True, type=str2bool)
parser.add_argument("--use_heads", default=False, type=str2bool)
parser.add_argument("--num_linera_heads", default=8, type=int)
parser.add_argument("--use_causal_filter", default=False, type=str2bool)
parser.add_argument("--alpha", default=0.0, type=float)
parser.add_argument("--p_lambda", default=0.0, type=float)
parser.add_argument("--beta1", default=0.5, type=float)

args = parser.parse_args()

p_lambda_list = [0.000001, 0.0000001, 0.00001, 0.00000001, 0.0001, 0]
beta1_list = [0.5 * x for x in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]]
alpha_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
result = {}
for p_lambda in p_lambda_list:
# for beta1 in beta1_list:
    for alpha in alpha_list:
        args.alpha = alpha
        args.p_lambda = p_lambda
        # args.beta1 = beta1
        args.train_dir = f"search/use_causal_v2={args.use_causal}.alpha{args.alpha:.2e}.p_lambda={args.p_lambda:.2e}"
        result[f"{alpha}, {p_lambda}"] = [*train(args)]
        # args.train_dir = f"search/use_causal={args.use_causal}.alpha{args.alpha:.2e}.beta1={args.beta1:.2e}"
        # result[f"{alpha}, {beta1}"] = [*train(args)]


        result_str = json.dumps(result)
        with open("search.json","w", encoding='utf-8') as f:
            f.write(result_str)

result_str = json.dumps(result)

with open("search.json","w", encoding='utf-8') as f:
    f.write(result_str)


