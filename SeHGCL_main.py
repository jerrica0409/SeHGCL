import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)
# from early_stopper import *
# from hin_loader import HIN
# from evaluation import *
# import util_funcs as uf
from SeHGCL import Model
import warnings
import time
from time import time
import torch
# import argparse
# import utility
from utility.load_data import load_data
from utility.dataloader import Data
from utility.paeser_yelp import parse_args
from utility.batch_test import set_seed
import utility
from utility.hg_file import load_g
# from hgmae.noise_hg_file import load_g


warnings.filterwarnings('ignore')
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]


def main():
    args = parse_args()
    # uf.seed_init(args.seed)
    set_seed(100)

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    dataset = Data(args.data_path + args.dataset)

    # ! Load Graph
    g, meta_paths, user_key, item_key, user_num, item_num = load_g(args.dataset)
    g = g.to(device)
    # g = load_g().to(device)

    (feats_u, mps_u) = \
        load_data(args.dataset, args.user, args.ratio, args.type_num)
    feats_dim_list_u = [i.shape[1] for i in feats_u]
    focused_feature_dim_u = feats_dim_list_u[0]
    num_mp_u = int(len(mps_u))

    (feats_i, mps_i) = \
        load_data(args.dataset, args.item, args.ratio, args.type_num)
    feats_dim_list_i = [i.shape[1] for i in feats_i]
    focused_feature_dim_i = feats_dim_list_i[0]
    num_mp_i = int(len(mps_i))

    feats_u = [feat.to(device) for feat in feats_u]
    mps_u = [mp.to(device) for mp in mps_u]
    feats_i = [feat.to(device) for feat in feats_i]
    mps_i = [mp.to(device) for mp in mps_i]

    model = Model(args, g, meta_paths, user_key, item_key, user_num, item_num, args.in_size, args.out_size, args.dropout, device,
                  num_mp_u, focused_feature_dim_u,
                  num_mp_i, focused_feature_dim_i,
                  feats_u, mps_u, feats_i, mps_i)#,
    model.to(device)


    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)#

    best_report_recall = 0.
    best_report_ndcg = 0.
    best_report_epoch = 0
    early_stop = 0
    for epoch in range(args.epochs):
        start_time = time()
        if epoch>=0:
            result = utility.batch_test.Test(dataset, model, device, eval(args.topK), args.multicore,
                                             args.test_batch_size, long_tail=False)
            if result['recall'][0] > best_report_recall:
                early_stop = 0
                best_report_epoch = epoch + 1
                best_report_recall = result['recall'][0]
                best_report_ndcg = result['ndcg'][0]
            else:
                early_stop += 1

            if early_stop >= 50:
                print("early stop! best epoch:", best_report_epoch, "bset_recall:", best_report_recall, ',best_ndcg:',
                      best_report_ndcg)
                with open('./result/' + args.dataset + "/result.txt", "a") as f:
                    f.write(str(args.dataset) + " ")
                    f.write(str(args.lr) + " ")
                    f.write(str(args.dropout) + " ")
                    f.write(str(args.feat_drop) + " ")
                    f.write(str(args.info) + " ")
                    f.write(str(args.tem) + " ")
                    f.write(str(args.mask_rate) + " ")
                    f.write(str(best_report_epoch) + " ")
                    f.write(str(best_report_recall) + " ")
                    f.write(str(best_report_ndcg) + "\n")
                break
            else:
                print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])

        model.train()
        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        users, pos_items, neg_items = utility.batch_test.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // args.batch_size + 1
        average_loss = 0.
        average_reg_loss = 0.
        average_cl_loss = 0.


        for batch_i, (batch_users, batch_positive, batch_negative) in enumerate(
                utility.batch_test.mini_batch(users, pos_items, neg_items, batch_size=args.batch_size)):
            batch_mf_loss, batch_emb_loss, infonce_loss = model.total_loss(batch_users, batch_positive, batch_negative, epoch)#
            batch_emb_loss = eval(args.regs)[0] * batch_emb_loss
            batch_loss = batch_emb_loss + batch_mf_loss + args.info * infonce_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            average_loss += batch_mf_loss.item()
            average_reg_loss += batch_emb_loss.item()
            average_cl_loss += infonce_loss.item()

        average_loss = average_loss / num_batch
        average_reg_loss = average_reg_loss / num_batch
        average_cl_loss = average_cl_loss / num_batch
        end_time = time()
        print("\t Epoch: %4d| train time: %.3f | train_loss:%.4f + %.4f + %.4f" % (
            epoch + 1, end_time - start_time, average_loss, average_reg_loss, average_cl_loss))

    print("best epoch:", best_report_epoch)
    print("best recall:", best_report_recall)
    print("best ndcg:", best_report_ndcg)

    # user_emb = torch.tensor(model.feature_dict.user, dtype=torch.float32)
    # item_emb = torch.tensor(model.feature_dict.business, dtype=torch.float32)
    # torch.save(user_emb, './analysis/user_emb_yelp.pth')
    # torch.save(item_emb, './analysis/item_emb_yelp.pth')


if __name__ == "__main__":

    main()
