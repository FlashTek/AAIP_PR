"""
    analyzes the result with have been obtained by executing

    for the influence of the hyperparameters:
        python gridsearch.py

    and for the robustness test:
        python dataaugmentation.py
"""

import pickle
import numpy as np

from matplotlib import rc
import matplotlib
from matplotlib import pyplot as plt
plt.rc('text',usetex=True)
font = {'family':'serif','size':11}
plt.rc('font',**font)

def analyze_gridsearch():
    """
        Analyzes the results of the gridsearch, i.e. the influence of the hyperparameters
    """

    #load the data
    data = None
    with open("result.pkl", "rb") as f:
        data = pickle.load(f)

    def split(lst, key):
        """
            Splits the data and creates new sub groups either sorted by the batch size or the learning rate (key)

            parameters:
                key - int, the id to sort the lis tby

            returns:
                list - list of sorted groups
        """

        sorted_lst = sorted(lst, key=lambda x:x[key])

        result = []
        current_sub_list = []
        for i in range(len(sorted_lst)):
            if len(current_sub_list) == 0 or sorted_lst[i][key] == current_sub_list[-1][key]:
                current_sub_list.append(sorted_lst[i])
            else:
                result.append(current_sub_list)
                current_sub_list = [sorted_lst[i]]
        if len(current_sub_list) > 0:
            result.append(current_sub_list)
        return result

    #split the data up
    split_batch_size = split(data, 0)
    split_learning_rate = split(data, 1)

    def plot_data_by_batch_size(splitted_data, key, label, loss_mode, n_col, n_row):
        """
            Plots the losses seperated by the batch size and then by the learning rate

            parameters:
                splitted_data - array, data to be plotted
                key - int, 
                loss_mode - string, either Training or Test
                n_col - int, indicates the arangement of the subplots
                n_row - int, indicates the arangement of the subplots
        """

        #prepare the plots
        f, axes = plt.subplots(n_row, n_col, sharex=False, sharey='row')
        f.set_size_inches(6.29, 3.54)
        
        if len(axes.shape) == 1:
            axes = axes.reshape((1, -1))

        f.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.20, hspace=0.09)

        #remove batch_size=800 for plotting
        for i in range(len(splitted_data)):
            if splitted_data[i][0][0] == 800:
                splitted_data.remove(splitted_data[i])
                break

        #plot the data
        for i in range(len(splitted_data)):
            plot_data = splitted_data[i]
            for triple in plot_data:
                axes[i//n_col, i%n_col].plot(triple[3+loss_mode], label=label.format(triple[key]))
               
            axes[i//n_col, i%n_col].set_xlim(0, 30)
            if i < n_col:
                labels = [item.get_text() for item in axes[0, i].get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                axes[0, i].set_xticklabels(empty_string_labels)
                axes[0, i].set_xticks([0.0, 15, 30])
            else:
                axes[1, i%n_col].set_xticks([0.0, 15.0, 30.0])

            if i < n_col:
                axes[i//n_col, i%n_col].text(9.0, 0.85, "n = {0}".format(plot_data[0][0]))
            else:
                axes[i//n_col, i%n_col].text(7.0, 0.85, "n = {0}".format(plot_data[0][0]))

        axes[0, 0].text(-18.0, 0.0, "Loss", rotation=90)
        axes[-1, 0].text(78.5, -0.20, r"Epoch")

        plt.title("")
        plt.legend(bbox_to_anchor=(-5.0, 2.12, 6.25, .102), loc=3,
                ncol=5, mode="expand", borderaxespad=0.)

        plt.savefig("results/batch_size_learning_rate_{0}.pdf".format("train" if loss_mode == 0 else "test"))
        plt.show()

    def plot_data_by_eta(splitted_data, key, label, loss_mode, n_col, n_row):
        """
            Plots the losses seperated by the learning rate and then by the batch size

            parameters:
                splitted_data - array, data to be plotted
                key - int, 
                loss_mode - string, either Training or Test
                n_col - int, indicates the arangement of the subplots
                n_row - int, indicates the arangement of the subplots
        """

        #prepare the plots
        f, axes = plt.subplots(n_row, n_col, sharex=False, sharey='row')
        f.set_size_inches(6.29, 3.54)
        
        if len(axes.shape) == 1:
            axes = axes.reshape((1, -1))

        f.subplots_adjust(left=0.125, bottom=0.12, right=0.9, top=0.79, wspace=0.16, hspace=0.09)
        
        #plot the data
        for i in range(len(splitted_data)):
            plot_data = splitted_data[i]
            
            for triple in plot_data:
                if triple[key] == 800:
                    continue
                axes[i//n_col, i%n_col].plot(triple[3+loss_mode], label=label.format(triple[key]))
                
            axes[i//n_col, i%n_col].set_xlim(0, 30)
        
            axes[0, i].set_xticks([0.0, 15, 30])
    

            if i < n_col:
                axes[i//n_col, i%n_col].text(8.0, 0.9, r"$\eta = {0}$".format(plot_data[0][1]))

            axes[i//n_col, i%n_col].set_ylim([1e-2, 1.0])

        axes[0, 0].text(-15.0, 0.525, "Loss", rotation=90)
        axes[0, 0].text(76.5, -0.15, r"Epoch")

        plt.title("")
        plt.legend(bbox_to_anchor=(-4.8, 1.05, 6.0, .102), loc=3,
                ncol=5, mode="expand", borderaxespad=0.)

        plt.savefig("results/learning_rate_batch_size_{0}.pdf".format("train" if loss_mode == 0 else "test"))
        plt.show()

    #plot the data
    plot_data_by_batch_size(split_batch_size, 1, r"$\eta = {0}$", 0, 5, 2)
    plot_data_by_eta(split_learning_rate, 0, r"$n = {0}$", 0, 5, 1)

    plot_data_by_batch_size(split_batch_size, 1, r"$\eta = {0}$", 1, 5, 2)
    plot_data_by_eta(split_learning_rate, 0, r"$n = {0}$", 1, 5, 1)

    def plot_training_time():
        #plots the average training time as a function of the batch size n
        avg_times = []
        avg_err_times = []
        batch_sizes = []
        for i in range(len(split_batch_size)):
            times = [x[2] for x in split_batch_size[i]]
            avg_times.append(np.mean(times)/30)
            avg_err_times.append(np.std(times)/30)
            batch_sizes.append(split_batch_size[i][0][0])

        plt.figure(figsize=(6.29, 3.54))
        plt.plot(batch_sizes, avg_times, color="C1", ls="--")
        plt.errorbar(batch_sizes, avg_times, yerr=avg_err_times, fmt=".", color="C1", capsize=2, label="Time")
        plt.legend()
        plt.ylabel("Training Time per Epoch [s]")
        plt.xlabel("Batch Size")
        plt.savefig("results/training_time_epoch.pdf")
        plt.show()

    plot_training_time()

def analyze_robustness():
    """
        Plots the training/test loss for the different network architectures used to examine the influence of noise
    """

    #load the data
    data = None
    with open("total_data.pkl", "rb") as f:
        data = pickle.load(f)

    #prepare the plots
    f, axes = plt.subplots(1, 2, sharex=False)
    f.set_size_inches(6.29, 3.54)
    f.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.9, wspace=0.05, hspace=0.20)
    axes[1].set_yticks([])
    axes[1] = axes[1].twinx()

    #plot the functions
    for j, ylabel in [(0, "Training Loss"), (1, "Test Loss")]:
        for i in range(len(data)):
            axes[j].plot(data[i][j], label="$N_{0}$".format(i))
        
        axes[j].set_ylabel(ylabel)

    axes[0].text(30, -0.065, r"Epoch")
    plt.legend(bbox_to_anchor=(-0.8, 1.0125, 1.5, .102), loc=3,
                ncol=5, mode="expand", borderaxespad=0.)
    plt.savefig("results/robustness_comparison.pdf")
    plt.show()

analyze_gridsearch()
analyze_robustness()