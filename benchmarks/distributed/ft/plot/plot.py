import json
import numpy as np
import matplotlib.pyplot as plt
import palettable


def plot(path, label, ax, **kwargs):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return

    data = []
    ts = []
    for line in lines:
        json_string = line.strip("DLLL").strip()
        json_object = json.loads(json_string)
        if "data" in json_object and "average_loss" in json_object["data"]:
            x = json_object["data"]["average_loss"]
            if x != x:
                print("nan")
                break
            data.append(x)
            ts.append(float(json_object["elapsedtime"]))

    if "1-bit" in label:
        ts = []
        if "phase1" in path:
            with open("lans-1bit/dllogger_phase1.json", "r") as f:
                lines = f.readlines()
        else:
            with open("lans-1bit/dllogger_phase2.json", "r") as f:
                lines = f.readlines()

        for line in lines:
            json_string = line.strip("DLLL").strip()
            json_object = json.loads(json_string)
            if "data" in json_object and "average_loss" in json_object["data"]:
                ts.append(float(json_object["elapsedtime"]))
    elif "Dithering" in label:
        ts = []
        if "phase1" in path:
            with open("lans-dithering-14865/dllogger_phase1.json", "r") as f:
                lines = f.readlines()
        else:
            with open("lans-dithering/dllogger_phase2.json", "r") as f:
                lines = f.readlines()

        for line in lines:
            json_string = line.strip("DLLL").strip()
            json_object = json.loads(json_string)
            if "data" in json_object and "average_loss" in json_object["data"]:
                ts.append(float(json_object["elapsedtime"]))

    ts = np.array(ts) / 3600  # to hour
    data = np.array(data)

    length = len(data)
    step = int(length * 0.001)
    indices = np.array(list(range(length)[::step]) + [length-1])

    print(ts[-1], data[-1])
    ax.plot(ts[indices], data[indices], label=label,
            **kwargs)


def export_legend(legend, filename="bert_legend.pdf"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


plt.rcParams.update({'font.size': 32, 'font.family': 'Myriad Pro'})
palette = ["#88CCEE",
           "#CC6677",
           "#DDCC77",
           "#117733",
           "#332288",
           "#AA4499",
           "#44AA99"]

settings = [
    {'linestyle': 'solid', 'linewidth': 3, },
    {'linestyle': '-', 'linewidth': 3, },
    {'linestyle': '-', 'linewidth': 3, },
    {'linestyle': '-', 'linewidth': 3, },
    {'linestyle': '-', 'linewidth': 3, },
]


for phase in [1, 2]:
    fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
    ax.set_prop_cycle(
        'color', palette)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    ax.set_xlabel("Training Time (h)")
    ax.set_ylabel("Pretraining Loss")
    ax.grid(True, linewidth=1, color='gray')

    plot("lans/dllogger_phase%d.json" % phase, "LANS", ax, **settings[0])
    plot("lans-1bit-14865/dllogger_phase%d.json" %
         phase, "CLAN(Scaled 1-bit with EF)", ax, **settings[1])

    plot("lans-topk-1051/dllogger_phase%d.json" % phase,
         "CLAN(Top-k(k=0.1%) with EF)", ax, **settings[2])
    plot("lans-dithering/dllogger_phase%d.json" % phase,
         "CLAN(Linear Dithering (7 bits))", ax, **settings[3])
    if phase == 1:
        plot("lans-1bit-no-ef/dllogger_phase%d.json" %
             phase, "CLAN(Scaled 1-bit)", ax, **settings[4])
        plot("lans-topk-no-ef/dllogger_phase%d.json" % phase,
             "CLAN(Top-k(k=0.1%))", ax, **settings[4])

    if phase == 1:
        ax.set_xlim(left=0)
        ax.set_ylim((1.5, 4.5))
        ax.set_yticks(np.arange(1.5, 4.6, 0.5))
    else:
        ax.set_xlim(left=0)
        ax.set_ylim((1.5, 2))
        ax.set_xticks([0, 5, 10])

    ax.legend()
    fig.savefig("bert_base_phase%d_time_accu.pdf" % phase, bbox_inches='tight')

    # if phase == 1:
    #     handles, labels = ax.get_legend_handles_labels()
    #     legend = fig.legend(handles, labels, bbox_to_anchor=(
    #         0.5, 1.2),  ncol=2, loc='upper center', frameon=False)

    #     export_legend(legend)
