import os
import json
import random
import argparse
import matplotlib.pyplot as plt
from time import time
from collections import Counter, defaultdict


homedir = "/scratch/bi01/guest02"

def run_all(num_neg, fs_method, param, dr, clf):
    print(fs_method, param, dr)
    dtext = ""
    if clf:
        classifier = """ --clf 'from sklearn.ensemble import GradientBoostingClassifier"""\
                     """as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")'"""
    else:
        classifier = """ --clf"""
    for j in param:
        for i in num_neg:
            if dr[1]:
                for k in dr[1]:
                    dim = f"_{dr[0]}_{k}"
                    dtext = f" --{dr[0]} {k}"
                    os.system(f"""python pig.py --numneg {i} -f --{fs_method} {j}""" + classifier + dtext)
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
    f1 = 0
    avg_fl_len = 0
    for k in os.listdir(result_dir):
        f = open(f"{result_dir}/{k}")
        data = json.load(f)
        if fl_len:
            avg_fl_len += len(data[3])
            n += 1
            continue
        f1 += float(data[1][1])
        n += 1
    if fl_len:
        return avg_fl_len / n
    if fl:
        return avg_score_d
    else:
        return f1/n


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
    folders = []
    for task_results in os.listdir(f"{homedir}/pig"):
        if "task_results" in task_results:
            split = task_results.split("_")
            if split[4] in types:
                folders.append(task_results)
    folders.sort(key=lambda x: (x.split("_")[4], x.split("_")[5], x.split("_")[6], float(x.split("_")[3])))
    plot_dataX = []
    plot_dataY = []
    default_plot = []
    plt.figure(1)
    for i in range(len(folders)):
        curr_split = folders[i].split("_")
        if i != 0:
            last_split = folders[i-1].split("_")
            if last_split[4] != curr_split[4]:
                default_plot = []
            if last_split[6] != curr_split[6]:
                splits = last_split
                if last_split[6] == "" and last_split[5] == "":
                    default_plot.append(plot_dataX)
                    default_plot.append(plot_dataY)
                    default_plot.append(f"{splits[4]}-{splits[5]}-{splits[6]}")
                    plot_dataX = []
                    plot_dataY = []
                    continue
                plt.plot(plot_dataX, plot_dataY, label=f"{splits[4]}-{splits[5]}-{splits[6]}", marker='o')
                plot_dataX = []
                plot_dataY = []
            if last_split[5] != curr_split[5] and curr_split[5] != "":
                params = [folders[i-3].split("_")[3], folders[i-2].split("_")[3], last_split[3]]
                for x, y, z in zip(default_plot[0], default_plot[1], params):
                    plt.annotate(z, (x,y), textcoords="offset points", xytext=(3,40), ha='center')
                plt.plot(default_plot[0], default_plot[1], label=default_plot[2], marker='o')
                plt.xlabel("Featurelist length")
                plt.ylabel("F1 scores")
                plt.ylim(0.0, 0.3)
                plt.legend(fontsize=9, loc=(1.05, 0))
                plt.tight_layout()
                plt.savefig(f"{homedir}/pig/results/F1_scores_{types}_{folders[i-1].split('_')[5]}")
                plt.figure(i)
        plot_dataX.append(get_score(f"{homedir}/pig/{folders[i]}", fl_len=True))
        plot_dataY.append(get_score(f"{homedir}/pig/{folders[i]}"))
    splits = folders[-1].split("_")
    params = [folders[-3].split("_")[3], folders[-2].split("_")[3], folders[-1].split("_")[3]]
    for x, y, z in zip(default_plot[0], default_plot[1], params):
        plt.annotate(z, (x,y), textcoords="offset points", xytext=(3,40), ha='center')
    plt.plot(plot_dataX, plot_dataY, label=f"{splits[4]}-{splits[5]}-{splits[6]}", marker='o')
    plt.plot(default_plot[0], default_plot[1], label=default_plot[2], marker='o')
    plt.xlabel("Featurelist length")
    plt.ylabel("F1 scores")
    plt.ylim(0.0, 0.3)
    plt.legend(fontsize=9, loc=(1.05, 0))
    plt.tight_layout()
    plt.savefig(f"{homedir}/pig/results/F1_scores_{types}_{folders[-1].split('_')[5]}")
    

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
    parser.add_argument('--default', nargs='+', default=[], help='Used for executing with set feature list')

    parser.add_argument('--pca', nargs='+', default=[], help='Used for executing with PCA dimension reduction')
    parser.add_argument('--lda', nargs='+', default=[], help='Used for executing with LDA dimension reduction')
    parser.add_argument('--umap', nargs='+', default=[], help='Used for executing with UMAP dimension reduction')
    parser.add_argument('--autoencoder', nargs='+', default=[], help='Used for executing with lasso feature selection')

    parser.add_argument('--plot', nargs='+', type=str, default=[], help='Used for generating a results plot')
    parser.add_argument('--clf', action='store_true', help='True for using default clf, False for Randomized search clf')

    num_neg = [30000]

    args = parser.parse_args()

    fs_methods = {"relief": args.relief, "svc1": args.svc1, "kbest": args.kbest, "random": args.random,
                  "default": args.default, "lasso": args.lasso, "svc2": args.svc2, "forest": args.forest}
    dr_methods = {'pca': args.pca, 'lda': args.lda, 'umap': args.umap, 'autoencoder': args.autoencoder}
    fs_methods = {k: fs_methods[k] for k in fs_methods.keys() if fs_methods[k]}
    dr_methods = {k: dr_methods[k] for k in dr_methods.keys() if dr_methods[k]}

    for fs in fs_methods.keys():
        if dr_methods:
            for dr in dr_methods.keys():
                if fs == "default":
                    run_all(num_neg, fs, [""], [dr, dr_methods[dr]], args.clf)
                else:
                    run_all(num_neg, fs, fs_methods[fs], [dr, dr_methods[dr]], args.clf)
        else:
            if fs == "default":
                run_all(num_neg, fs, [""], ["", ""], args.clf)
            else:
                run_all(num_neg, fs, fs_methods[fs], ["", ""], args.clf)

                    
    if args.bestfl:
        get_best_fl(num_randf, num_neg)
    if args.plot:
        types = args.plot
        makeplot(num_neg, types)
    print(f"Total time: {time() - starttime}")
