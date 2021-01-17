import os
import json
import random
import argparse
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


homedir = "/scratch/bi01/guest02"

def run_all(num_neg, fs_method, param, dr=""):
    dimension_r = ""
    if dr:
        dimension_r = " -dr " + dr
    for j in param:
        for i in num_neg:
            os.system(f"""python pig.py --numneg {i} -f --clf 'from sklearn.ensemble import GradientBoostingClassifier"""\
                      f""" as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")' --{fs_method} {j}""" + dimension_r)
            if dr:
                dim = f"_{dr}"
            else:
                dim = ""
            if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim):
                os.system(f"rm -r {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)
            os.system(f"mkdir {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)
            os.system(f"mv {homedir}/tmp/task_results/*.json {homedir}/pig/task_results_{i}_{j}_{fs_method}" + dim)

def run_random(num_randf, numneg, dr=""):
    dimension_r = ""
    if dr:
        dimension_r = " -dr " + dr
    for i in num_neg:
        for j in num_randf:
            os.system(f"""python pig.py --numneg {i} --random {j} 1 -f --clf 'from sklearn.ensemble import"""\
                    """ GradientBoostingClassifier as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")'""" + dimension_r)
            if dr:
                dim = f"_{dr}"
            else:
                dim = ""
            if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_random" + dim):
                os.system(f"rm -r {homedir}/pig/task_results_{i}_{j}_random" + dim)
            os.system(f"mkdir {homedir}/pig/task_results_{i}_{j}_random" + dim)
            os.system(f"mv {homedir}/tmp/task_results/*.json {homedir}/pig/task_results_{i}_{j}_random" + dim)


def run_svc1(num_neg, c, dr=""):
    dimension_r = ""
    if dr:
        dimension_r = " -dr " + dr
    for j in c:
        for i in num_neg:
            os.system(f"""python pig.py --numneg {i} --svc1 {j} -f --clf 'from sklearn.ensemble import"""\
                    """ GradientBoostingClassifier as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")'""" + dimension_r)
            if dr:
                dim = f"_{dr}"
            else:
                dim = ""
            if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_svc1" + dim):
                os.system(f"rm -r {homedir}/pig/task_results_{i}_{j}_svc1" + dim)
            os.system(f"mkdir {homedir}/pig/task_results_{i}_{j}_svc1" + dim)
            os.system(f"mv {homedir}/tmp/task_results/*.json {homedir}/pig/task_results_{i}_{j}_svc1" + dim)


def run_default(num_neg, dr=""):
    o = open(f"{homedir}/pig/results/best_ft_list.json", "r")
    fl = json.load(o)
    dimension_r = ""
    if dr:
        dimension_r = " -dr " + dr
    for i in num_neg:
        os.system("""python pig.py --numneg %d --featurelist "%s" -f --clf 'from sklearn.ensemble import"""\
                """ GradientBoostingClassifier as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")'""" % (i,  fl) + dimension_r)
        if dr:
            dim = f"_{dr}"
        else:
            dim = ""
        if os.path.exists(f"{homedir}/pig/task_results_{i}_default" + dim):
            os.system(f"rm -r {homedir}/pig/task_results_{i}_default" + dim)
        os.system(f"mkdir {homedir}/pig/task_results_{i}_default" + dim)
        os.system(f"mv {homedir}/tmp/task_results/*.json {homedir}/pig/task_results_{i}_default" + dim)
        

def run_kbest(num_neg, k, dr=""):
    dimension_r = ""
    if dr:
        dimension_r = " -dr " + dr
    for j in k:
        for i in num_neg:
            os.system(f"""python pig.py --numneg {i} --kbest {j} -f --clf 'from sklearn.ensemble import"""\
                    """ GradientBoostingClassifier as gbc; clf=gbc(n_estimators=300, learning_rate=0.3, max_features="sqrt")'""" + dimension_r)
            if dr:
                dim = f"_{dr}"
            else:
                dim = ""
            if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_kbest" + dim):
                os.system(f"rm -r {homedir}/pig/task_results_{i}_{j}_kbest" + dim)
            os.system(f"mkdir {homedir}/pig/task_results_{i}_{j}_kbest" + dim)
            os.system(f"mv {homedir}/tmp/task_results/*.json {homedir}/pig/task_results_{i}_{j}_kbest" + dim)

        
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
    from time import time
    starttime = time()
    plot_data_random = defaultdict(list)
    plot_data_svc1 = defaultdict(list)
    plot_data_kbest = defaultdict(list)
    plot_data_default = []
    num_randf = []
    c = []
    k = []
    for d in os.listdir(f"{homedir}/pig"):
        if "task_results" in d:
            if "default" in d:
                continue
            else:
                param = d.split("_")[3]
                if "svc1" in d:
                    if float(param) in c:
                        continue
                    else:
                        if float(param) % 1 == 0:
                            c.append(int(param))
                        else:
                            c.append(float(param))
                elif "random" in d:
                    if int(param) in num_randf:
                        continue
                    else:
                        num_randf.append(int(param))
                elif "kbest" in d:
                    if int(param) in k:
                        continue
                    else:
                        k.append(int(param))
    num_randf.sort()
    c.sort()
    k.sort()
    for task_results in os.listdir(f"{homedir}/pig"):
        if "task_results" in task_results:
            split = task_results.split("_")
            if split[4] in types:
                plt.plot(get_score(f"{homedir}/pig/{task_results}", fl_len=True),
                         get_score(f"{homedir}/pig/{task_results}"), 'o', label=f"{task_results[13:]}")

    """for i in num_neg:
        for j in num_randf:
            if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_random"):
                plot_data_random[j].append((i, get_score(f"{homedir}/pig/task_results_{i}_{j}_random")))
        for j in c:
            if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_svc1"):
                plot_data_svc1[j].append((i, get_score(f"{homedir}/pig/task_results_{i}_{j}_svc1")))
        for j in k:
            if os.path.exists(f"{homedir}/pig/task_results_{i}_{j}_kbest"):
                plot_data_kbest[j].append((i, get_score(f"{homedir}/pig/task_results_{i}_{j}_kbest")))
        if os.path.exists(f"{homedir}/pig/task_results_{i}_default"):
            plot_data_default.append(get_score(f"{homedir}/pig/task_results_{i}_default"))
    if "random" in types:
        for x in plot_data_random.keys():
            f1_scores = [0] * len(num_neg)
            for res in plot_data_random[x]:
                f1_scores[num_neg.index(res[0])] = res[1]
            plt.plot(num_neg, f1_scores, label=f"Random_{x}")
    if "svc1" in types:
        for x in plot_data_svc1.keys():
            f1_scores = [0] * len(num_neg) 
            for res in plot_data_svc1[x]:
                f1_scores[num_neg.index(res[0])] = res[1]
            plt.plot(num_neg, f1_scores, label=f"Svc1_{x}")
    if "kbest" in types:
        for x in plot_data_kbest.keys():
            f1_scores = [0] * len(num_neg)
            for res in plot_data_kbest[x]:
                f1_scores[num_neg.index(res[0])] = res[1]
            plt.plot(num_neg, f1_scores, label=f"Kbest_{x}")
    if "default" in types:
        plt.plot(num_neg, plot_data_default, label="Default")"""
    plt.xlabel("Featurelist length")
    plt.ylabel("F1 scores")
    plt.ylim(0, 0.5)
    plt.legend(fontsize=9, loc=(1.05, 0))
    plt.tight_layout()
    plt.savefig(f"{homedir}/pig/results/F1_scores")
    print(f"{time() - starttime}: Done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bestfl', action='store_true', help='Used for computing the best feature lists and f1 scores')
    parser.add_argument('--random', nargs='+', default=[], help='Used for executing with random features')
    parser.add_argument('--svc1', nargs='+', default=[], help='Used for executing with svc1 feature selection')
    parser.add_argument('--kbest', nargs='+', default=[], help='Used for executing with kbest feature selection')
    parser.add_argument('-d', '--default', action='store_true', help='Used for running with a set feature list.')
    parser.add_argument('--plot', nargs='+', type=str, default=[], help='Used for generating a results plot')
    parser.add_argument('-dr', '--dimension_reduction', nargs=1, type=str, default=[], help='Used for performing dimension reduction on the featurelist')
    num_neg = [10000, 30000, 50000]
    fs_methods = ["svc1", "kbest", "random", "default"]
    args = parser.parse_args()
    try:
        dr = args.dimension_reduction[0]
    except:
        dr = ""
    for method in vars(args).keys():
        if method in fs_methods and vars(args)[method]:
            if method == "default":
                run_all(num_neg, method, [""], dr)       
            else:
                run_all(num_neg, method, vars(args)[method], dr)       
    if args.bestfl:
        get_best_fl(num_randf, num_neg)
    if args.plot:
        types = args.plot
        makeplot(num_neg, types)
