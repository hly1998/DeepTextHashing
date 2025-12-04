import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()
import glob
try:
    from .eval_helpers import get_labels_and_indices
except:
    from eval_helpers import get_labels_and_indices
from scipy.stats import ttest_ind


def my_sig(val1, baseline, higherBetter=True, alpha=0.05):
    if higherBetter:
        if np.mean(val1) > np.mean(baseline):
            pval = ttest_ind(val1, baseline)[1]
            if pval < alpha:
                return "$^{\\blacktriangle}$"
    else:
        if np.mean(val1) < np.mean(baseline):
            pval = ttest_ind(val1, baseline)[1]
            if pval < alpha:
                return "$^{\\downarrow}$"
    return ""


methods = ["LCH", "STH", "vdshweak", "nash", "rbsh", "pairrec", "ours16"]
dnames = ["reuters", "TMC", "agnews"]
bits = ["32", "64"]
methods_name_mapper = {"LCH":"LCH", "STH":"STH", "SpH":"SpH", "vdshweak":"NbrReg", "nash":"NASH", "rbsh":"RBSH", "pairrec":"PairRec", "ours16":"MISH"}

def format_float(float):
    return "{:.4f}".format(float)

def format_float1(float):
    return "{:.1f}".format(float)


def prec_table():
    resfolder = "../results_baselines/savedres/"
    reses = {}

    for method in methods:
        reses[method] = {}
        for dname in dnames:

            if dname not in reses[method]:
                reses[method][dname] = {}

            for bit in bits:
                if bit not in reses[method][dname]:
                    reses[method][dname][bit] = []

                # [dname, bit, baseline, test_hitrate, test_violations, test_avg_topK, test_prec100_iter, test_prec100, lower_test_prec100_iter]
                vals = pickle.load(open(resfolder + dname + "_" + bit + "_" + method + ".pkl", "rb"))
                avg_prec = np.mean(vals[-3])
                wc_prec = np.mean(vals[-1])

                reses[method][dname][bit] = [vals[-1], vals[-3]]

    for method in methods:
        s = methods_name_mapper[method]
        for dname in dnames:
            for bit in bits:
                for idx in [0,1]: # worst case or avg case

                    other_values = [reses[m][dname][bit][idx] for m in methods if method not in m]
                    other_means = np.squeeze(np.mean(other_values, 1))

                    mean_idx = np.argmax(other_means)
                    best_baseline = other_values[mean_idx]
                    this_method = reses[method][dname][bit][idx]
                    this_mean = np.mean(this_method)

                    is_significant = my_sig(this_method, best_baseline)
                    if "ours" not in method:
                        is_significant = ""

                    all_means = np.concatenate((other_means, [this_mean]), -1)
                    all_means_one = np.zeros(len(all_means), dtype=np.int32)
                    all_means_one[-1] = 1
                    all_means_one = all_means_one[np.argsort(all_means)[::-1]]
                    rank = np.where(all_means_one == 1)[0][0]

                    if rank == 0:
                        highlighter = lambda x: "\\textbf{" + x + "}"
                    elif rank == 1:
                        highlighter = lambda x: "\\underline{" + x + "}"
                    else:
                        highlighter = lambda x: x

                    val = format_float(np.mean(this_method))
                    s += " & " + highlighter(val) + is_significant 
        s += "\\\\"
        print(s)

def time_table():
    resfolder = "../results_baselines/matlabformat/"# "../../timing/mih-master/saveruns/"

    reses = {}
    lintimes = {}

    for method in methods:
        reses[method] = {}
        for dname in dnames:

            if dname not in reses[method]:
                reses[method][dname] = {}
                lintimes[dname] = {}

            for bit in bits:
                if bit not in reses[method][dname]:
                    reses[method][dname][bit] = []
                    lintimes[dname][bit] = -1

                # [dname, bit, baseline, test_hitrate, test_violations, test_avg_topK, test_prec100_iter, test_prec100, lower_test_prec100_iter]
                a = pickle.load(open(resfolder + method + "_" + dname + "_" + bit + ".timepickle", "rb"))
                mih_time = a[0]
                mih_corr_time = a[1]
                lin_time = a[2]

                mih_time = np.array(mih_time)
                mih_corr_time = np.array(mih_corr_time)
                lin_time = np.median(lin_time)

                lintimes[dname][bit] = lin_time

                speedup_mih = lin_time / mih_time
                speedup_mih_corr = lin_time / mih_corr_time

                reses[method][dname][bit] = [speedup_mih, speedup_mih_corr]

    for method in methods:
        s = methods_name_mapper[method]
        for dname in dnames:
            for bit in bits:
                for idx in [0,1]: # mih and greedy-fixed (GSO)

                    other_values = [reses[m][dname][bit][0] for m in methods if method not in m]
                    other_values += [reses[m][dname][bit][1] for m in methods if method not in m and "our" not in m]
                    other_values += [reses[method][dname][bit][(idx+1)%2] ]

                    other_means = np.array([np.mean(other_values[i]) for i in range(len(other_values))])

                    mean_idx = np.argmax(other_means)
                    best_baseline = other_values[mean_idx]
                    this_method = reses[method][dname][bit][idx]
                    this_mean = np.mean(this_method)

                    is_significant = my_sig(this_method, best_baseline)
                    if "ours" not in method:
                        is_significant = ""

                    all_means = np.concatenate((other_means, [this_mean]), -1)
                    all_means_one = np.zeros(len(all_means), dtype=np.int32)
                    all_means_one[-1] = 1
                    all_means_one = all_means_one[np.argsort(all_means)[::-1]]
                    rank = np.where(all_means_one == 1)[0][0]

                    if rank == 0:
                        highlighter = lambda x: "\\textbf{" + x + "}"
                    elif rank == 1:
                        highlighter = lambda x: "\\underline{" + x + "}"
                    else:
                        highlighter = lambda x: x

                    val = format_float(np.mean(this_method))
                    if not ("our" in method and idx == 1):
                        s += " & " + highlighter(val) + is_significant 
                    else:
                        s += " & " + val 
        s += "\\\\"
        print(s)

    s = "Linear scan time"
    for dname in dnames:
        for bit in bits:
            s += " & \\multicolumn{2}{c|}{"+ "{:.6f}".format(lintimes[dname][bit])+" s}"
    s += " \\\\ "
    print(s)




def eff_eff_plot_old(bits):
    resfolder_prec = "../results_baselines/savedres/"
    resfolder_speed = "../results_baselines/matlabformat/"# "../../timing/mih-master/saveruns/"
    reses = {}

    for method in methods:

        if method not in reses:
            reses[method] = [[],[]] # prec-list, speedup-list

        for dname in dnames:

            for bit in bits:

                # [dname, bit, baseline, test_hitrate, test_violations, test_avg_topK, test_prec100_iter, test_prec100, lower_test_prec100_iter]
                a = pickle.load(open(resfolder_speed + method + "_" + dname + "_" + bit + ".timepickle", "rb")) #pickle.load(open(resfolder_speed + "16_" + method + "_" + dname + "_" + bit + ".mat.pkl", "rb"))
                mih_time = a[0]
                mih_corr_time = a[1]
                lin_time = a[2]

                mih_time = np.median(mih_time)
                mih_corr_time = np.median(mih_corr_time)
                lin_time = np.median(lin_time)

                lowest_time = min(mih_time, mih_corr_time)
                speedup = lin_time / lowest_time

                vals = pickle.load(open(resfolder_prec + dname + "_" + bit + "_" + method + ".pkl", "rb"))
                avg_prec = np.mean(vals[-3])
                wc_prec = np.mean(vals[-1])

                reses[method][0].append(np.mean(avg_prec))
                reses[method][1].append(speedup)

    dataset = []
    all_speeds = []
    for method in methods:
        all_speeds.append(reses[method][1])
    all_speeds = np.array(all_speeds)
    all_speeds_max = np.max(all_speeds, 0)

    for method in methods:
        v1 = np.mean(reses[method][0])
        v2 = np.mean(np.array(reses[method][1]))# / all_speeds_max)

        dataset.append([v1, v2, methods_name_mapper[method]])

    dataset = np.array(dataset)
    dataset = pd.DataFrame(dataset, columns = ["prec", "speedup", "method"])
    dataset["prec"] = pd.to_numeric(dataset["prec"])
    dataset["speedup"] = pd.to_numeric(dataset["speedup"])

    ax = sns.lmplot('prec', # Horizontal axis
           'speedup', # Vertical axis
           data=dataset, # Data source
           fit_reg=False, # Don't fix a regression line
          ) # size and dimension

    #plt.title('Example Plot')
    # Set x-axis label
    plt.xlabel('mean prec@100')
    # Set y-axis label
    plt.ylabel('mean speedup')

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            if "STH" in str(point['val']):
                ax.text(point['x']+.00, point['y']+0.09, str(point['val']))
            else:
                ax.text(point['x']-.02, point['y']+0.09, str(point['val']))
            #ax.text(point['x']-.02, point['y']+0.02, str(point['val']))

    label_point(dataset.prec, dataset.speedup, dataset.method, plt.gca())  


def eff_eff_plot(bits):
    resfolder_prec = "../results_baselines/savedres/"
    resfolder_speed = "../results_baselines/matlabformat/"# "../../timing/mih-master/saveruns/"
    reses = {}

    for method in methods:

        if method not in reses:
            reses[method] = [[],[]] # prec-list, speedup-list

        for dname in dnames:

            for bit in bits:

                # [dname, bit, baseline, test_hitrate, test_violations, test_avg_topK, test_prec100_iter, test_prec100, lower_test_prec100_iter]
                a = pickle.load(open(resfolder_speed + method + "_" + dname + "_" + bit + ".timepickle", "rb")) #pickle.load(open(resfolder_speed + "16_" + method + "_" + dname + "_" + bit + ".mat.pkl", "rb"))
                mih_time = a[0]
                mih_corr_time = a[1]
                lin_time = a[2]

                mih_time = np.median(mih_time)
                mih_corr_time = np.median(mih_corr_time)
                lin_time = np.median(lin_time)

                lowest_time = min(mih_time, mih_corr_time)
                speedup = lin_time / lowest_time

                vals = pickle.load(open(resfolder_prec + dname + "_" + bit + "_" + method + ".pkl", "rb"))
                avg_prec = np.mean(vals[-3])
                wc_prec = np.mean(vals[-1])

                reses[method][0].append(np.mean(avg_prec))
                reses[method][1].append(speedup)

    dataset = []
    all_speeds = []
    all_perfs = []
    for method in methods:
        all_speeds.append(reses[method][1])
        all_perfs.append(reses[method][0])

    all_speeds = np.array(all_speeds)
    all_speeds_max = np.max(all_speeds, 0)

    all_perfs = np.array(all_perfs)
    all_perfs_max = np.max(all_perfs, 0)

    for method in methods:
        v1 = np.mean(np.array(reses[method][0]) / all_perfs_max)
        v2 = np.mean(np.array(reses[method][1]) / all_speeds_max)

        dataset.append([v1, v2, methods_name_mapper[method], 0 if "ours" in method else 1])# "orange" if "ours" in method else "grey"])

    dataset = np.array(dataset)
    dataset = pd.DataFrame(dataset, columns = ["prec", "speedup", "method", "colorme"])
    dataset["prec"] = pd.to_numeric(dataset["prec"])
    dataset["speedup"] = pd.to_numeric(dataset["speedup"])

    sns.set(color_codes=True)
    ax = sns.lmplot(x='prec', # Horizontal axis
           y='speedup', # Vertical axis
           data=dataset, # Data source
           fit_reg=False, # Don't fix a regression line
          ) # size and dimension

    # Set x-axis label
    plt.xlabel('mean prec@100 relative to MISH')
    # Set y-axis label
    plt.ylabel('mean speedup relative to MISH')

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']-.01, point['y']+0.01, str(point['val']))

    label_point(dataset.prec, dataset.speedup, dataset.method, plt.gca())  


def rel_time_plot(bits):
    resfolder_prec = "../results_baselines/savedres/"
    resfolder_speed = "../results_baselines/matlabformat/"# "../../timing/mih-master/saveruns/"
    reses = {}

    for method in methods:

        if method not in reses:
            reses[method] = [[],[]] # prec-list, speedup-list

        for dname in dnames:

            for bit in bits:

                # [dname, bit, baseline, test_hitrate, test_violations, test_avg_topK, test_prec100_iter, test_prec100, lower_test_prec100_iter]
                a = pickle.load(open(resfolder_speed + method + "_" + dname + "_" + bit + ".timepickle", "rb")) #pickle.load(open(resfolder_speed + "16_" + method + "_" + dname + "_" + bit + ".mat.pkl", "rb"))
                mih_time = a[0]
                mih_corr_time = a[1]
                lin_time = a[2]

                mih_time = np.median(mih_time)
                mih_corr_time = np.median(mih_corr_time)
                lin_time = np.median(lin_time)

                lowest_time = min(mih_time, mih_corr_time)
                speedup = lin_time / lowest_time

                vals = pickle.load(open(resfolder_prec + dname + "_" + bit + "_" + method + ".pkl", "rb"))
                avg_prec = np.mean(vals[-3])
                wc_prec = np.mean(vals[-1])

                reses[method][0].append(np.mean(avg_prec))
                reses[method][1].append(speedup)

    dataset = []
    plt.figure()

    all_speeds = []
    for method in methods:
        all_speeds.append(reses[method][1])
    all_speeds = np.array(all_speeds)
    all_speeds_max = np.max(all_speeds, 0)

    for method in methods:
        v1 = np.mean(reses[method][0])
        v2 = np.mean(np.array(reses[method][1]) / all_speeds_max )

        dataset.append([v1, v2, methods_name_mapper[method]])

    dataset = np.array(dataset)
    v = dataset[:, 1].astype(float)

    dataset = dataset[np.argsort(dataset[:, 1].astype(float))]
    dataset = pd.DataFrame(dataset, columns = ["prec", "speedup", "method"])
    dataset["prec"] = pd.to_numeric(dataset["prec"])
    dataset["speedup"] = pd.to_numeric(dataset["speedup"])


    values = dataset["speedup"]
    clrs = ['grey' if (x < max(values)) else 'darkorange' for x in values ]

    sns.barplot(x="method", y="speedup", data=dataset, palette=clrs)
    plt.ylabel("avg. speedup relative to MISH")
    plt.xlabel("")


def plotgrid():
    speed = [[7.180, 7.430, 6.220, 6.980, 5.915],
            [9.304, 9.525, 9.525, 10.901, 9.369],
            [9.972, 10.481, 10.481, 12.054, 11.044],
            [11.388, 11.734, 11.734, 13.276, 13.282],
            [12.475, 13.786, 13.786, 13.891, 13.532],
            [13.960, 14.776, 14.776, 15.863, 15.801],
            [14.122, 14.601, 14.601, 15.625, 15.888],
            [14.057, 14.057, 13.929, 15.438, 15.382]]


    perf = [[0.8407, 0.8409, 0.8408, 0.8341, 0.8299],
            [0.8393, 0.8414, 0.8393, 0.8392, 0.8291],
            [0.8409, 0.8442, 0.8409, 0.8410, 0.8299],
            [0.8419, 0.8447, 0.8419, 0.8415, 0.8344],
            [0.8428, 0.8419, 0.8428, 0.8397, 0.8370],
            [0.8428, 0.8422, 0.8428, 0.8383, 0.8339],
            [0.8371, 0.8369, 0.8371, 0.8381, 0.8354],
            [0.8376, 0.8376, 0.8387, 0.8374, 0.8325]]

    speed = np.array(speed)
    perf = np.array(perf)

    header = ["0.0", "0.01", "0.05", "0.1", "0.2"]
    index = ["0.0", "1.0", "3.0", "5.0", "7.0", "9.0", "11.0", "13.0"]
    speed = pd.DataFrame(speed, index = index, columns=header)

    fig, ax = plt.subplots(figsize=(3.5,3))
    ax = sns.heatmap(speed, ax=ax, vmin=5, vmax=16, cmap="Blues")
    ax.set_title("Speedup")
    ax.set_xlabel("$\\alpha_2$")
    ax.set_ylabel("$\\alpha_1$")
    plt.tight_layout()
    plt.savefig('../figs/paramgrid-speed.eps')

    perf = pd.DataFrame(perf, index = index, columns=header)
    fig, ax = plt.subplots(figsize=(3.5,3))
    ax = sns.heatmap(perf, ax=ax, vmin=0.8, vmax=0.85, cmap="Blues")
    ax.set_title("Prec@100")
    ax.set_xlabel("$\\alpha_2$")
    ax.set_ylabel("$\\alpha_1$")
    plt.tight_layout()
    plt.savefig('../figs/paramgrid-perf.eps')

def plotgrid32():

    speed = [[28.123, 28.123, 23.879, 21.819, 26.770],
                [37.910, 37.910, 33.880, 35.661, 31.651],
                [41.819, 41.819, 40.013, 41.529, 36.783],
                [42.471, 41.617, 40.720, 49.428, 47.086],
                [42.826, 42.563, 42.826, 49.695, 42.890],
                [46.347, 40.648, 46.347, 49.693, 45.383],
                [44.961, 44.961, 41.827, 48.889, 47.713],
                [45.455, 45.455, 39.439, 45.997, 43.803]]

    perf = [[0.8351, 0.8351, 0.8335, 0.8265, 0.8257],
            [0.8366, 0.8366, 0.8358, 0.8307, 0.8267],
            [0.8349, 0.8349, 0.8340, 0.8328, 0.8281],
            [0.8342, 0.8350, 0.8375, 0.8342, 0.8349],
            [0.8347, 0.8339, 0.8347, 0.8321, 0.8286],
            [0.8313, 0.8344, 0.8313, 0.8330, 0.8303],
            [0.8285, 0.8285, 0.8311, 0.8316, 0.8289],
            [0.8282, 0.8282, 0.8311, 0.8265, 0.8244]]

    speed = np.array(speed)
    perf = np.array(perf)

    header = ["0.0", "0.01", "0.05", "0.1", "0.2"]
    index = ["0.0", "1.0", "3.0", "5.0", "7.0", "9.0", "11.0", "13.0"]
    speed = pd.DataFrame(speed, index = index, columns=header)

    fig, ax = plt.subplots(figsize=(3.5,3))
    ax = sns.heatmap(speed, ax=ax, vmin=20, vmax=50, cmap="Blues")
    ax.set_title("Speedup")
    ax.set_xlabel("$\\alpha_2$")
    ax.set_ylabel("$\\alpha_1$")
    plt.tight_layout()
    plt.savefig('../figs/paramgrid-speed32.eps')

    perf = pd.DataFrame(perf, index = index, columns=header)
    fig, ax = plt.subplots(figsize=(3.5,3))
    ax = sns.heatmap(perf, ax=ax, vmin=0.8, vmax=0.84, cmap="Blues")
    ax.set_title("Prec@100")
    ax.set_xlabel("$\\alpha_2$")
    ax.set_ylabel("$\\alpha_1$")
    plt.tight_layout()
    plt.savefig('../figs/paramgrid-perf32.eps')




def gap_table():
    resfolder = "../results_baselines/savedres/"
    reses = {}

    for method in methods:
        reses[method] = {}
        s = methods_name_mapper[method] + " & "
        for dname in dnames:

            if dname not in reses:
                reses[method][dname] = {}

            for bit in bits:
                if bit not in reses[method][dname]:
                    reses[method][dname][bit] = []

                # [dname, bit, baseline, test_hitrate, test_violations, test_avg_topK, test_prec100_iter, test_prec100, lower_test_prec100_iter]
                vals = pickle.load(open(resfolder + dname + "_" + bit + "_" + method + ".pkl", "rb"))

                avg_prec = np.mean(vals[-3])
                wc_prec = np.mean(vals[-1])

                reses[method][dname][bit] = [wc_prec, avg_prec]

    reses_best = {}
    for dname in dnames:
        reses_best[dname] = {}
        for bit in bits:
            reses_best[dname][bit] = []
            avg_precs = []
            wc_precs = []
            for method in methods:
                avg_precs.append(reses[method][dname][bit][1])
                wc_precs.append(reses[method][dname][bit][0])

            reses_best[dname][bit] = [max(wc_precs), max(avg_precs)]

    for method in methods:
        s = methods_name_mapper[method] + " & "
        for bit in bits:
            # get avg worst case decrease
            wc_prec = [reses[method][x][bit][0] for x in dnames]
            best_wc_prec = [reses_best[x][bit][0] for x in dnames]
            wc_decreases = [(best_wc_prec[i] - wc_prec[i]) / best_wc_prec[i] * 100 for i in range(len(wc_prec))]

            
            # get avg avg case decrease
            avg_prec = [reses[method][x][bit][1] for x in dnames]
            best_avg_prec = [reses_best[x][bit][1] for x in dnames]
            avg_prec_decreases = [(best_avg_prec[i] - avg_prec[i]) / best_avg_prec[i] * 100 for i in range(len(avg_prec))]

            #print(method, bit, np.mean(wc_decreases), np.mean(avg_prec_decreases))

            s += "-" + "{:.2f}".format(np.mean(wc_decreases)) + "\\% & -" + "{:.2f}".format(np.mean(avg_prec_decreases)) + "\\%"
            if bit == "32":
                s += " & "
            else:
                s += " \\\\"
        print(s)

def plot_candidate_set_size(dname, bit):
    path = "../candidate_set_sizes/cand_"

    reses = {}

    tmp_labels = []
    tmp_hits = []
    tmp_dist100 = []

    for method in methods:
        reses[method] = {}
        s = methods_name_mapper[method] + " & "
        if dname not in reses:
            reses[method][dname] = {}
        if bit not in reses[method][dname]:
            reses[method][dname][bit] = []


        content = pickle.load(open(path + method + "_" + dname + "_" + bit + ".pkl", "rb"))
        candidate_size = content[0][:,-1]
        dist100 = content[1]
        #if "reuters" in dname:
        #    print(method, bit, np.sum(candidate_size))

        candidate_size = np.array(candidate_size)
        selecter = np.argsort(candidate_size)[:int(len(candidate_size)*0.99)]
        candidate_size = candidate_size[selecter]
        #tmp_labels = tmp_labels[selecter]

        for i in range(len(candidate_size)):
            tmp_labels.append(methods_name_mapper[method])
            tmp_hits.append(candidate_size[i])
            #tmp_dist100.append(dist100[i])



    df_hits = pd.DataFrame(dict(num_candidates=np.array(tmp_hits), method=np.array(tmp_labels)))
    #df_dist100 = pd.DataFrame(dict(top100dist=np.array(tmp_dist100), method=np.array(tmp_labels)))

    plt.figure(figsize=(5, 3.1))
    sns.violinplot(x="method", y="num_candidates",
                data=df_hits, cut=0)#, whis=[1,99], showfliers = False)
    sns.despine(offset=0, trim=True)
    plt.xlabel("")
    plt.ylabel("num. candidates")
    #plt.ylim([0,10000])
    plt.title(dname + " " + bit + " bits")
    plt.tight_layout()
    plt.savefig('../figs/hits-' + dname + "-" + bit + '.eps')

    #plt.figure()
    #sns.violinplot(x="method", y="top100dist",
    #            data=df_dist100, whis=[1,99], showfliers = False)
    #sns.despine(offset=0, trim=True)
    #plt.xlabel("")
    #plt.ylabel("top-100 Hamming distance")
    #plt.title(dname + " " + bit + " bits")
    #plt.tight_layout()

from MulticoreTSNE import MulticoreTSNE as TSNE
def tsne_plot(path, dname, fromval, toval, title):


    content = pickle.load(open(path, "rb"))
    codes = content[1][-1][:,fromval:toval] # test codes
    labels = np.squeeze(content[2][-1])


    def make_plot(rank_label, rank_vect):
        uniques = np.unique(rank_label).tolist()
        rank_label = [uniques.index(rank_label[i]) for i in range(len(rank_label))]
        #cm = plt.get_cmap('gist_rainbow')
        colormap = {0:"blue", 1:"red", 2:"green", 3:"black"}
        rank_label = [colormap[v] for v in rank_label]

        #rank_label = [cm(rank_label[i]//3*3.0/len(uniques)) for i in range(len(rank_label))]

        #print("running t-tsne")
        '''
        tsne = TSNE(
            n_components=2, perplexity=250, learning_rate=100,
            n_jobs=7, initialization="pca", #metric="hamming",
            early_exaggeration_iter=250, early_exaggeration=12, n_iter=1000,
            neighbors="exact", negative_gradient_method="bh")

        tsne = TSNE(
            n_components=2, perplexity=200, learning_rate=100,
            n_jobs=7, initialization="pca",  # metric="hamming",
            early_exaggeration_iter=250, early_exaggeration=12, n_iter=1000,
            neighbors="exact", negative_gradient_method="bh")
        '''
        tsne = TSNE(n_jobs=28)

        embedding = tsne.fit_transform(rank_vect)
        print(title, embedding.shape)
        plt.figure()
        plt.scatter(embedding[:,0], embedding[:,1], c=rank_label, s=10, alpha=0.9)
        plt.title(title,  fontsize=25)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("../figs/" + title + ".pdf")
        plt.savefig("../figs/" + title + ".eps")
        plt.savefig("../figs/" + title + ".png")


    make_plot(labels, codes)
    #make_plot(norank_label, norank_vect, "16 bit RBSH w/o ranking")
    #make_plot(sth_label, sth_vect, "16 bit STH")
    plt.tight_layout()

if __name__ == '__main__':
    prec_table()
    
    time_table()

    eff_eff_plot(bits)
    rel_time_plot(bits)

    plotgrid()
    gap_table()


    for dname in dnames:
        for bit in bits:
            plot_candidate_set_size(dname, bit)

    plt.show()
    exit()

    tsne_plot("../results_baselines_greedy/pairrec_agnews_32.pkl", "agnews", 0, 32, "32 bit PairRec")
    tsne_plot("../results_baselines_greedy/pairrec_agnews_32.pkl", "agnews", 0, 16, "32 bit PairRec-GSO (Substring 1)")
    tsne_plot("../results_baselines/pairrec_agnews_32.pkl", "agnews", 0, 16, "32 bit PairRec (Substring 1)")
    tsne_plot("../results_baselines_greedy/pairrec_agnews_32.pkl", "agnews", 16, 32, "32 bit PairRec-GSO (Substring 2)")
    tsne_plot("../results_baselines/pairrec_agnews_32.pkl", "agnews", 16, 32, "32 bit PairRec (Substring 2)")

    tsne_plot("../results_baselines_greedy/STH_agnews_32.pkl", "agnews", 0, 32, "32 bit STH")
    tsne_plot("../results_baselines_greedy/STH_agnews_32.pkl", "agnews", 0, 16, "32 bit STH-GSO (Substring 1)")
    tsne_plot("../results_baselines/STH_agnews_32.pkl", "agnews", 0, 16, "32 bit STH (Substring 1)")
    tsne_plot("../results_baselines_greedy/STH_agnews_32.pkl", "agnews", 16, 32, "32 bit STH-GSO (Substring 2)")
    tsne_plot("../results_baselines/STH_agnews_32.pkl", "agnews", 16, 32, "32 bit STH (Substring 2)")

    tsne_plot("../results_baselines/ours16_agnews_32.pkl", "agnews", 0, 32, "32 bit MISH")
    tsne_plot("../results_baselines/ours16_agnews_32.pkl", "agnews", 0, 16, "32 bit MISH (Substring 1)")
    tsne_plot("../results_baselines/ours16_agnews_32.pkl", "agnews", 16, 32, "32 bit MISH (Substring 2)")

    plt.show()