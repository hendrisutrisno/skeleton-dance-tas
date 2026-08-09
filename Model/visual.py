import math
import numpy as np
from action_dict import action_label
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm

classes = list(action_label.keys())
classes_number = len(action_label)

def color_list():
    colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    "#cd94df", '#46f0f0', '#f032e6',
    '#008080', '#e6beff', '#9a6324', "#f9ffc895", '#800000',
    '#aaffc3', '#808000', "#e6a05a", '#000075', '#808080',
    "#d10da0", '#000000', '#a9a9a9', '#d2691e', "#8418e9"
    ]
    return colors

def chart(label, predict, color_list, save_dir):
    barprops = dict(aspect='auto', cmap=ListedColormap(color_list[:classes_number+1]), interpolation='nearest', vmin =0, vmax=classes_number+1)
    if label is None:
        fig = plt.figure(figsize=(15, 1.5))
        interval = 1
        ax = fig.add_axes([0, 0, 1, interval])
        ax.imshow([predict], **barprops)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("Predict", rotation=0, labelpad=40, fontsize=12)
        legend_ax = fig.add_axes([0.96, 0.1, 0.03, 0.8])
        legend_ax.axis('off')

        patches = [mpatches.Patch(color=color_list[i], label=str(classes[i])) for i in range(classes_number)]
        #legend = legend_ax.legend(handles=patches, loc='center left', title='Class', fontsize=8, title_fontsize=9)
        #fig.add_artist(legend)
        #fig.suptitle(save_dir.stem, fontsize=14)
        plt.savefig(save_dir)
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 3), sharex= True)
        ylabels = ["label", "predict"]
        series_list = [label, predict]
        max_f = np.max([label.shape[0], predict.shape[0]])
        for i in range(2):
            axes[i].imshow([series_list[i]], **barprops)
            axes[i].set_yticks([])
            axes[i].set_ylabel(ylabels[i], rotation=0, labelpad=40, fontsize=12)
            axes[i].set_xticks(range(0, max_f, 1000))
        legend_ax = fig.add_axes([0.96, 0.1, 0.03, 0.8])
        legend_ax.axis('off')

        patches = [mpatches.Patch(color=color_list[i], label=str(classes[i])) for i in range(classes_number)]
        #legend = legend_ax.legend(handles=patches, loc='center left', title='Class', fontsize=8, title_fontsize=9)
        #fig.add_artist(legend)
        #plt.tight_layout(rect=[0, 0, 0.95, 1]) 
        #plt.title(save_dir.stem)
        plt.savefig(save_dir)

def before_after_chart(label, predict, refinement, color_list, save_dir, refine):
    barprops = dict(aspect='auto', cmap=ListedColormap(color_list), interpolation='nearest', vmin =0, vmax=len(color_list) - 1)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 4.5), sharex= True)
    if refine:
        ylabels = ["label", "predict", 'filter + refine']
    else:
        ylabels = ["label", "predict", 'filter']
    series_list = [label, predict, refinement]
    max_f = np.max([label.shape[0], predict.shape[0], refinement.shape[0]])
    for i in range(3):
        axes[i].imshow([series_list[i]], **barprops)
        axes[i].set_yticks([])
        axes[i].set_ylabel(ylabels[i], rotation=0, labelpad=40, fontsize=12)
        axes[i].set_xticks(range(0, max_f, 1000))
    legend_ax = fig.add_axes([0.96, 0.1, 0.03, 0.8])
    legend_ax.axis('off')
    patches = [mpatches.Patch(color=color_list[i], label=str(classes[i])) for i in range(7)]
    legend = legend_ax.legend(handles=patches, loc='center left', title='Class', fontsize=8, title_fontsize=9)
    fig.add_artist(legend)
    #plt.tight_layout(rect=[0, 0, 0.95, 1]) 
    plt.title(save_dir.stem)
    plt.savefig(save_dir)
    plt.close(fig)

def comparsion2_chart(label, predict, color_list, save_dir):
    barprops = dict(aspect='auto', cmap=ListedColormap(color_list), interpolation='nearest', vmin =0, vmax=len(color_list) - 1)
    ylabels = ["GT", "FACT_BODY"]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 3), sharex= True)
    series_list = [label, predict]
    #max_f = np.max([label.shape[0],  len(predict[0]), len(predict[1]), len(predict[2]), len(predict[3])])
    for i in range(5):
        axes[i].imshow([series_list[i]], **barprops)
        axes[i].set_yticks([])
        axes[i].set_ylabel(ylabels[i], rotation=0, labelpad=40, fontsize=12)
    legend_ax = fig.add_axes([0.96, 0.1, 0.03, 0.8])
    legend_ax.axis('off')
    patches = [mpatches.Patch(color=color_list[i], label=str(classes[i])) for i in range(7)]
    legend = legend_ax.legend(handles=patches, loc='center left', title='Class', fontsize=8, title_fontsize=9)
    fig.add_artist(legend)
    #plt.tight_layout(rect=[0, 0, 0.95, 1]) 
    plt.title(save_dir.stem)
    plt.savefig(save_dir)
    plt.close(fig)


def comparsion_chart(label, predict, color_list, save_dir):
    barprops = dict(aspect='auto', cmap=ListedColormap(color_list), interpolation='nearest', vmin =0, vmax=len(color_list) - 1)
    ylabels = ["GT", "FACT_BODY"]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 3), sharex= True)
    series_list = [label, predict[0]]
    # max_f = np.max([label.shape[0],  len(predict[0])])
    for i in range(2):
        axes[i].imshow([series_list[i]], **barprops)
        axes[i].set_yticks([])
        axes[i].set_ylabel(ylabels[i], rotation=0, labelpad=40, fontsize=12)
    legend_ax = fig.add_axes([0.96, 0.1, 0.03, 0.8])
    legend_ax.axis('off')
    patches = [mpatches.Patch(color=color_list[i], label=str(classes[i])) for i in range(7)]
    legend = legend_ax.legend(handles=patches, loc='center right', title='Class', fontsize=8, title_fontsize=9)
    fig.add_artist(legend)
    #plt.tight_layout(rect=[0, 0, 0.95, 1]) 
    plt.title(save_dir.stem)
    plt.savefig(save_dir/"result")
    plt.close(fig)

def stage_test_vis(predictions, result_dir):
    rows = len(predictions)
    colors = color_list()
    barprops = dict(aspect='auto', cmap=ListedColormap(colors), interpolation='nearest', vmin =0, vmax=len(colors) - 1)
    fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(15, 4.5), sharex= True)
    max_f = predictions[0].shape[0]
    for i in range(rows):
        axes[i].imshow([predictions[i]], **barprops)
        axes[i].set_yticks([])
        axes[i].set_ylabel(f"stage_{i+1}", rotation=0, labelpad=40, fontsize=12)
        axes[i].set_xticks(range(0, max_f, 1000))
    legend_ax = fig.add_axes([0.96, 0.1, 0.03, 0.8])
    legend_ax.axis('off')
    patches = [mpatches.Patch(color=colors[i], label=str(classes[i])) for i in range(7)]
    legend = legend_ax.legend(handles=patches, loc='center left', title='Class', fontsize=8, title_fontsize=9)
    fig.add_artist(legend)
    #plt.tight_layout(rect=[0, 0, 0.95, 1]) 
    plt.title(result_dir.stem)
    plt.savefig(result_dir)
    plt.close(fig)

def boundary_curve(boundary, title, save_path):
    boundary = boundary.squeeze(0)
    fig, ax = plt.subplots()
    ax.plot(boundary)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    plt.savefig(save_path / f"{title}.png")
    plt.close(fig)

def classifiction(classes, title, save_path):
    peak = np.concatenate(
        [   
            ((classes[:-2] < classes[1:-1]) & (classes[2:] < classes[1:-1])|
            (classes[:-2] > classes[1:-1]) & (classes[2:] > classes[1:-1])),
        ], axis = 0
    )
    peak_idx = np.where(peak)[0].tolist()
    y = [classes[idx] for idx in peak_idx]
    fig, ax = plt.subplots()
    ax.plot(classes)
    ax.scatter(peak_idx, y, c = 'red')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(0, len(classes), 1000))
    plt.savefig(save_path / f"{title}.png")
    plt.close(fig)

def loss_curve(loss_list, result_path):
    title = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    for i in range(len(loss_list)):
        plt.plot(loss_list[i], label = title[i])

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("loss curve")
    plt.savefig(result_path+"/loss.png")
    plt.show()