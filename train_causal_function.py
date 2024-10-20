import os
import time
import torch
import argparse

from model import CauseFormer
from utils import *


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
parser.add_argument("--num_epochs", default=1000, type=int)
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

p_lambda = 0.000001
beta1 = 0.5
beta2 = 0.5
last_W = None
kappa_1 = 2.1
kappa_2 = 0.9

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4
 
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank
 
        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

@torch.no_grad()
def update_beta1(W, K):
    global beta1, beta2, last_W, kappa_1, kappa_2
    diagonal = (torch.exp(W * W)).diagonal(offset=0, dim1=-2, dim2=-1)
    trace = diagonal.sum(dim=-1)
    beta1 = 0.1 * beta1 + 0.1 * beta2 * (trace - K).mean()
    beta1 = 0.1 if beta1 > 5 else beta1
    # self.beta1 = self.beta1

@torch.no_grad()
def update_beta2(W, K):
    # self.beta2=0.1
    global beta1, beta2, last_W, kappa_1, kappa_2
    diagonal = (torch.exp(W * W)).diagonal(offset=0, dim1=-2, dim2=-1)
    trace = diagonal.sum(dim=-1)
    left_value = torch.abs(trace - K).mean()
    if last_W is not None:
        diagonal = (torch.exp(last_W * last_W)).diagonal(offset=0, dim1=-2, dim2=-1)
        trace = diagonal.sum(dim=-1)
        right_value = kappa_2 * torch.abs(trace - K).mean()
    else:
        right_value = left_value
    if left_value >= right_value:
        beta2 = kappa_1 * beta2
    last_W = copy.copy(W)

    beta2 = 0.1 if beta2 > 5 else beta2

def train(args):
    # args = parser.parse_args()
    if hasattr(args,"p_lambda"):
        p_lambda = args.p_lambda
    if hasattr(args,"beta1"):
        beta1 = args.beta1
    if not os.path.isdir(args.dataset + "_" + args.train_dir):
        os.makedirs(args.dataset + "_" + args.train_dir)
    with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f:
        f.write("\n".join([str(k) + "," + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    u2i_index, i2u_index = build_index(args.dataset)

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + "_" + args.train_dir, f"epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.use_causal={args.use_causal}.use_heads={args.use_heads}.log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    sampler = WarpSampler(
        user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3
    )
    model = CauseFormer(usernum, itemnum, args).to(
        args.device
    )  # no ReLU activation in original SASRec implementation?
    model_structure(model)

    # initialize item code
    # model.item_code.assign_codes(user_train)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        # try:
        #     model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        #     tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
        #     epoch_start_idx = int(tail[: tail.find(".")]) + 1
        # except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        #     print("failed loading state_dicts, pls check file path: ", end="")
        #     print(args.state_dict_path)
        #     print("pdb enabled for your quick check, pls type exit() if you do not need it")
        #     import pdb

        #     pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    patience = 0
    tolerance = 5
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break  # just to decrease identition
        for step in range(
            num_batch
        ):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits, causal_mask = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            if args.use_causal:
                loss += p_lambda * torch.norm(causal_mask, 1, dim=[1,2]).mean()
                K = causal_mask.size(-1)
                diagonal = (torch.exp((causal_mask * causal_mask).clip(max=14))).diagonal(offset=0, dim1=-2, dim2=-1)
                trace = diagonal.sum(dim=-1)

                # loss += beta1 * abs(trace - K).mean()

                loss += beta1 * abs(trace - K).mean()
                loss += (beta2 / 2.0) * torch.pow(torch.abs(trace - K), 2).mean()
                # print(f"trace loss:{(trace - K).mean()}")
                # print(f"norm loss:{torch.norm(causal_mask, 1)}")
                assert not torch.any(torch.isinf(causal_mask)),"causal mask"
                assert not torch.any(torch.isnan(causal_mask)),"causal mask"
                update_beta1(causal_mask, K)
                update_beta2(causal_mask, K)
                pass
            lr_reg = 0.0
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
                lr_reg += args.l2_emb * torch.norm(param)
            
            assert not torch.isnan(loss),"loss"
            loss.backward()
            adam_optimizer.step()
            print(
                "loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())
            ) 
            f.write("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()) + "\n")
            f.flush()
            print(
                "loss in epoch {} iteration {}: {} pos loss:{} neg loss:{} norm loss:{} trace loss:{} l2_reg:{}".format(epoch, step, bce_criterion(pos_logits[indices], pos_labels[indices]).item() + bce_criterion(neg_logits[indices], neg_labels[indices]).item(), 
                                                                                                            bce_criterion(pos_logits[indices], pos_labels[indices]).item(),
                                                                                                            bce_criterion(neg_logits[indices], neg_labels[indices]).item(),
                                                                                                            torch.norm(causal_mask, 1, dim=[1,2]).mean(),
                                                                                                            abs(trace - K).mean().item(),
                                                                                                            lr_reg.item())
            )  # expected 0.4~0.6 after init few epochs
            f.write(
                "loss in epoch {} iteration {}: {} pos loss:{} neg loss:{} norm loss:{} trace loss:{} l2_reg:{}".format(epoch, step, bce_criterion(pos_logits[indices], pos_labels[indices]).item() + bce_criterion(neg_logits[indices], neg_labels[indices]).item(), 
                                                                                                            bce_criterion(pos_logits[indices], pos_labels[indices]).item(),
                                                                                                            bce_criterion(neg_logits[indices], neg_labels[indices]).item(),
                                                                                                            torch.norm(causal_mask, 1, dim=[1,2]).mean(),
                                                                                                            abs(trace - K).mean().item(),
                                                                                                            lr_reg.item()) + "\n"
            )

        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating", end="")
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(
                "epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)"
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            )

            if (
                t_valid[0] > best_val_ndcg
                or t_valid[1] > best_val_hr
                or t_test[0] > best_test_ndcg
                or t_test[1] > best_test_hr
            ):
                patience = 0
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = args.dataset + "_" + args.train_dir
                fname = "Causal.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.use_causal={}.use_heads={}.pth"
                fname = fname.format(
                    epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen,args.use_causal,args.use_heads
                )
                torch.save(model.state_dict(), os.path.join(folder, fname))
            else:
                patience += 1

            f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs or patience == tolerance:
            folder = args.dataset + "_" + args.train_dir
            fname = "Causal.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.use_causal={}.use_heads={}.pth"
            fname = fname.format(
                args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen,args.use_causal,args.use_heads
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
    return best_test_ndcg,best_test_hr


if __name__ == '__main__':
    args = parser.parse_args()
    result = train(args)
    print(result)