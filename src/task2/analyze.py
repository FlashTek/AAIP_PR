#analyze the result
import pickle
import numpy as np

data = None
with open("result.pkl", "rb") as f:
    data = pickle.load(f)


def split(lst, key):
    sorted_lst = sorted(lst, key=lambda x:x[key])

    result = []
    current_sub_list = [sorted_lst[0]]
    for i in range(len(sorted_lst)):
        if sorted_lst[i][key] == current_sub_list[-1][key]:
            current_sub_list.append(sorted_lst[i])
        else:
            result.append(current_sub_list)
            current_sub_list = [sorted_lst[i]]
    return result

split_batch_size = split(data, 0)
split_learning_rate = split(data, 1)

from matplotlib import pyplot as plt

def plot_data(splitted_data, key, label, n_col, n_row):
    f, axes = plt.subplots(n_row, n_col, sharex=False, sharey='row')
    if len(axes.shape) == 1:
        axes = axes.reshape((1, -1))
    print(axes.shape)
    print(len(splitted_data))

    f.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.14, hspace=0.09)
    for i in range(len(splitted_data)):
        plot_data = splitted_data[i]
        for triple in plot_data:
            axes[i//n_col, i%n_col].plot(triple[2], label=label.format(triple[key]))

        axes[i//n_col, i%n_col].set_xlim(0, 30)
        if i < n_col:
            labels = [item.get_text() for item in axes[0, i].get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            axes[0, i].set_xticklabels(empty_string_labels)
            axes[0, i].set_xticks([0.0, 15, 30])
        else:
            axes[1, i%n_col].set_xticks([0.0, 15.0, 30.0])

        if i < n_col:
            axes[i//n_col, i%n_col].text(10.0, 0.9, "n = {0}".format(plot_data[0][0]))
        else:
            axes[i//n_col, i%n_col].text(8.0, 0.9, "n = {0}".format(plot_data[0][0]))

    axes[0, 0].text(-15.0, 0.0, "Loss", rotation=90)
    axes[-1, 0].text(75.0, -0.15, r"Epoch")

    plt.title("")
    #plt.tight_layout()
    plt.legend(bbox_to_anchor=(-5.0, 2.12, 6.5, .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)

    plt.savefig("batch_size_learning_rate.pdf")
    plt.show()

def plot_data2(splitted_data, key, label, n_col, n_row):
    f, axes = plt.subplots(n_row, n_col, sharex=False, sharey='row')
    f.set_size_inches(6.5, 5.5)
    if len(axes.shape) == 1:
        axes = axes.reshape((1, -1))
    print(axes.shape)
    print(len(splitted_data))

    f.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.14, hspace=0.09)
    for i in range(len(splitted_data)):
        plot_data = splitted_data[i]
        for triple in plot_data:
            axes[i//n_col, i%n_col].plot(triple[2], label=label.format(triple[key]))

        axes[i//n_col, i%n_col].set_xlim(0, 30)
        if i < n_col:
            labels = [item.get_text() for item in axes[0, i].get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            axes[0, i].set_xticklabels(empty_string_labels)
            axes[0, i].set_xticks([0.0, 15, 30])
        else:
            axes[1, i%n_col].set_xticks([0.0, 15.0, 30.0])

        if i < n_col:
            axes[i//n_col, i%n_col].text(10.0, 0.9, r"$\eta = {0}$".format(plot_data[0][key]))
        else:
            axes[i//n_col, i%n_col].text(8.0, 0.9, r"$\eta = {0}$".format(plot_data[0][key]))

    axes[0, 0].text(-15.0, 0.5, "Loss", rotation=90)
    axes[0, 0].text(75.0, -0.15, r"Epoch")

    plt.title("")
    #plt.tight_layout()
    plt.legend(bbox_to_anchor=(-4.0, 1.02, 5.5, .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)

    plt.savefig("batch_size_learning_rate.pdf")
    plt.show()

#plot_data(split_batch_size, 1, r"$\eta = {0}$", 5, 2)
plot_data2(split_learning_rate, 0, r"$n = {0}$", 4, 1)