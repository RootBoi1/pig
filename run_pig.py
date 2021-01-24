import os
import json
import random
import argparse
import matplotlib.pyplot as plt
from time import time
from collections import Counter, defaultdict


homedir = "/scratch/bi01/guest02"

def run_all(num_neg, fs_method, param, dr):
    print(dr)
    dtext = ""
    for j in param:
        for i in num_neg:
            if dr[1]:
                for k in dr[1]:
                    dim = f"_{dr[0]}_{k}"
                    dtext = f" --{dr[0]} {k}"
                    os.system(f"""python pig.py --numneg {i} -f --clf 'from sklearn.ensemble import GradientBoostingClassifier"""\
                              f""" as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")' --{fs_method} {j}""" + dtext)
                    if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim):
                        os.system(f"rm -r {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)
                    os.system(f"mkdir {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)
                    os.system(f"mv {homedir}/tmp/task_results/*.json {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)
            else:
                dim = "__"
                os.system(f"""python pig.py --numneg {i} -f --clf 'from sklearn.ensemble import GradientBoostingClassifier"""\
                          f""" as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")' --{fs_method} {j}""")
                if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim):
                    os.system(f"rm -r {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)
                os.system(f"mkdir {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)
                os.system(f"mv {homedir}/tmp/task_results/*.json {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)



def get_score(result_dir, fl=False, fl_len=False):
    n = 0
    avg_fl_len = 0
    f1_list = []
    score_d = defaultdict(list)
    avg_score_d = {}
    for k in os.listdir(result_dir):
        task_id = int(k.split(".")[0])
        f = open(f"{result_dir}/{k}")
        data = json.load(f)
        if fl_len:
            avg_fl_len += len(data[3])
            n += 1
            continue
        if "random" in result_dir: 
            score_d[task_id % 1000].append((data[1][2][0], data[1][2][2]))
        else:
            score_d[task_id].append((data[1][2][0], data[1][2][2]))
        n += 1
    if fl_len:
        return avg_fl_len / n
    for key in score_d.keys():
        sum_tpr, sum_precision = 0, 0
        for tpr, precision in score_d[key]:
            sum_tpr += tpr
            sum_precision += precision
        avg_tpr, avg_precision = sum_tpr / len(score_d[key]), sum_precision / len(score_d[key])
        if avg_precision + avg_tpr != 0:
            f1 = 2 * ((avg_precision * avg_tpr) / (avg_precision + avg_tpr))
        else:
            f1 = 0
        f1_list.append(f1)
        avg_score_d[key] = f1
    if fl:
        return avg_score_d
    else:
        return sum(f1_list) / len(f1_list)


def get_best_fl(num_randf, num_neg, fl=True):
    best_fl= {}
    for i in num_randf:
        for j in num_neg:
            feature_list = get_score(f"{homedir}/pig/task_results_{j}_{i}_random", fl=True)
            best_fl_id = sorted(feature_list.items(), key=lambda x: x[1])[-1][0]
            f = open(f"{homedir}/pig/task_results_{j}_{i}_random/{best_fl_id}.json")
            best_fl[f"({i}, {j})"] = (best_fl_id, json.load(f)[3])
    o = open(f"{homedir}/pig/results/best_ft_list.json", "w")
    json.dump(best_fl[f"({num_randf[-1]}, {num_neg[-1]})"], o)
    if fl:
        print(best_fl[f"({num_randf[-1]}, {num_neg[-1]})"])
        return best_fl
    else:
        return best_fl[f"({num_randf[-1]}, {num_neg[-1]})"]


def makeplot(num_neg, types):
    starttime = time()
    folders = []
    for task_results in os.listdir(f"{homedir}/pig"):
        if "task_results" in task_results:
            split = task_results.split("_")
            if split[4] in types:
                folders.append(task_results)
    folders.sort(key=lambda x: (x.split("_")[4], x.split("_")[5], x.split("_")[6], float(x.split("_")[3])))
    plot_dataX = []
    plot_dataY = []
    for i in range(len(folders)):
        if i != 0:
            if folders[i-1].split("_")[6] != folders[i].split("_")[6]:
                splits = folders[i-1].split("_")
                plt.plot(plot_dataX, plot_dataY, label=f"{splits[4]}-{split[5]}-{splits[6]}", marker='o')
                plot_dataX = []
                plot_dataY = []
        plot_dataX.append(get_score(f"{homedir}/pig/{folders[i]}", fl_len=True))
        plot_dataY.append(get_score(f"{homedir}/pig/{folders[i]}"))
    plt.plot(plot_dataX, plot_dataY, label=f"{splits[4]}-{split[5]}-{splits[6]}", marker='o')
    plt.xlabel("Featurelist length")
    plt.ylabel("F1 scores")
    plt.ylim(0.2, 0.3)
    plt.legend(fontsize=9, loc=(1.05, 0))
    plt.tight_layout()
    plt.savefig(f"{homedir}/pig/results/F1_scores_{types}")
    print(f"{time() - starttime}: Done")
    

if __name__ == "__main__":
    starttime = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bestfl', action='store_true', help='Used for computing the best feature lists and f1 scores')
    parser.add_argument('--random', nargs='+', default=[], help='Used for executing with random features')
    parser.add_argument('--svc1', nargs='+', default=[], help='Used for executing with svc1 feature selection')
    parser.add_argument('--svc2', nargs='+', default=[], help='Used for executing with svc2 feature selection')
    parser.add_argument('--kbest', nargs='+', default=[], help='Used for executing with kbest feature selection')
    parser.add_argument('--forest', nargs='+', default=[], help='Used for executing with kbest feature selection')
    parser.add_argument('--lasso', nargs='+', default=[], help='Used for executing with lasso feature selection')
    parser.add_argument('--relief', nargs='+', default=[], help='Used for executing with relief feature selection')

    parser.add_argument('--pca', nargs='+', default=[], help='Used for executing with PCA dimension reduction')
    parser.add_argument('--lda', nargs='+', default=[], help='Used for executing with LDA dimension reduction')
    parser.add_argument('--umap', nargs='+', default=[], help='Used for executing with UMAP dimension reduction')
    parser.add_argument('--autoencoder', nargs='+', default=[], help='Used for executing with lasso feature selection')

    parser.add_argument('--plot', nargs='+', type=str, default=[], help='Used for generating a results plot')

    num_neg = [50000]

    args = parser.parse_args()

    fs_methods = ["relief", "svc1", "kbest", "random", "default", "lasso", "svc2", "forest"]
    dr_methods = {'pca': args.pca, 'lda': args.lda, 'umap': args.umap, 'autoencoder': args.autoencoder}

    for method in vars(args).keys():
        for dmethod in dr_methods.keys():
            if dr_methods[dmethod]:
                if method in fs_methods and vars(args)[method]:
                    if method == "default":
                        run_all(num_neg, method, [""], [dmethod, dr_methods[dmethod]])
                    else:
                        run_all(num_neg, method, vars(args)[method], [dmethod, dr_methods[dmethod]])       
    if args.bestfl:
        get_best_fl(num_randf, num_neg)
    if args.plot:
        types = args.plot
        makeplot(num_neg, types)
    print(f"Total time: {time() - starttime}")
