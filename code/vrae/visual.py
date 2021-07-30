import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import torch
import numpy as np

def confusion_matrix_plot(y, y_pred, state):
    print('plotting conf matrix')
    cf_matrix = confusion_matrix(y, y_pred)
    # sns_plot = sns.heatmap(cf_matrix, annot=False, cmap='Blues')
    sns_plot = sns.heatmap(cf_matrix, cmap='Blues')
    sns_plot.get_figure().savefig('save/confmatrix_'+state+'.png')

def confusion_matrix_show(y, y_pred, state=None):
    print('showing conf matrix')
    cf_matrix = confusion_matrix(y, y_pred)
    plt.figure()
    # sns_plot = sns.heatmap(cf_matrix, annot=False, cmap='Blues')
    sns_plot = sns.heatmap(cf_matrix, cmap='Blues', annot=True)
    # sns_plot.get_figure().savefig('save/confmatrix_'+state+'.png')


def tsne(latent, y_ground_truth, state): 
    latent = latent.detach().numpy()
    y_ground_truth = y_ground_truth.detach().numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=y_ground_truth, 
        palette=sns.color_palette("hls", num_labels),
        # data=df_subset,
        legend="full"
        )

    sns_plot.get_figure().savefig('save/tsne_'+state+'.png')

def tsne_show(latent, y_ground_truth, p_label=None, property=None, state=None): 
    latent = latent.detach().numpy()
    y_ground_truth = y_ground_truth.detach().numpy()
    if property == 'stiffness':
        p_label = np.where(p_label==0, 'soft', 'hard') 
        p_label = np.where(p_label==1, 'mid', p_label)
    if property == 'roughness':
        p_label = np.where(p_label==0, 'smooth', 'rough') 
        p_label = np.where(p_label==1, 'mid', p_label) 
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    if property is not None:
        sns_plot = sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=y_ground_truth, 
            palette=sns.color_palette("hls", num_labels),
            style = p_label,
            legend="full"
            )
    else:
        sns_plot = sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=y_ground_truth, 
            palette=sns.color_palette("hls", num_labels),
            legend="full"
            )

    # sns_plot.get_figure().savefig('save/tsne_'+state+'.png')

def multiple_tsne(latent, y_ground_truth, stiff_y, rough_y, state): 
    latent = latent.detach().numpy()
    y_ground_truth, stiff_y, rough_y = y_ground_truth.detach().numpy(), stiff_y.detach().numpy(), rough_y.detach().numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    plt.figure(figsize=(16,10))
    fig, axes = plt.subplots(1,2)
    stiff_y = np.where(stiff_y==0, 'soft', 'hard') 
    rough_y = np.where(rough_y==0, 'smooth', 'rough') 
    sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=y_ground_truth, 
            style=stiff_y,
            palette=sns.color_palette("hls", num_labels),
            legend='auto',
            ax=axes[0])
    sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=y_ground_truth, 
            style=rough_y,
            palette=sns.color_palette("hls", num_labels),
            legend='auto',
            ax=axes[1])
    # lines = []
    # labels = []
    # for ax in fig.axes:
    #     axLine, axLabel = ax.get_legend_handles_labels()
    #     lines.extend(axLine)
    #     labels.extend(axLabel)
    # fig.legend(lines, labels, loc = 'upper right')
    # plt.legend(loc='upper right')
    fig.savefig('save/multiple_tsne_'+state+'.png')

def plot_grad_flow(named_parameters, figname, if_plot, if_save=True):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                print("None grad for param {}".format(n))
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    if if_save:
        plt.savefig(figname)
    if if_plot:
        plt.show()
    plt.clf()
    

def plot_grad_flow_v2(named_parameters, figname, if_plot):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(figname)
    if if_plot:
        plt.show()


def plot_stats(file_name, model_name, save=None):
    # plot stats
    fig, ax = plt.subplots(figsize=(15, 7))
    data = pd.read_pickle(file_name)
    epoch_train_loss = data['epoch_train_loss']
    ax.plot(epoch_train_loss[:], label='epoch_train_loss')
    epoch_train_acc = data['epoch_train_acc']
    ax.plot(epoch_train_acc[:], label='epoch_train_acc')
    epoch_val_loss = data['epoch_val_loss']
    ax.plot(epoch_val_loss[:], label='epoch_val_loss')
    epoch_val_acc = data['epoch_val_acc']
    ax.plot(epoch_val_acc[:], label='epoch_val_acc')
    epoch_train_stiffness_acc = data['epoch_train_stiffness_acc']
    ax.plot(epoch_train_stiffness_acc[:], label='epoch_train_stiffness_acc')
    epoch_val_stiffness_acc = data['epoch_val_stiffness_acc']
    ax.plot(epoch_val_stiffness_acc[:], label='epoch_val_stiffness_acc')
    epoch_train_roughness_acc = data['epoch_train_roughness_acc']
    ax.plot(epoch_train_roughness_acc[:], label='epoch_train_roughness_acc')
    epoch_val_roughness_acc = data['epoch_val_roughness_acc']
    ax.plot(epoch_val_roughness_acc[:], label='epoch_val_roughness_acc')
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.grid(True)
    ax.set_title('test acc after training'+str(data['test_acc1'])+'/'+str(data['test_acc2']))
    plt.legend(loc='upper right')
    if save is None:
        plt.savefig('save/'+model_name+'stats.png')

# file_name = './models_and_stats/for_plot_fold0_0.pkl'
# data = pd.read_pickle(file_name)
# fig, ax = plt.subplots(figsize=(15, 7))
# epoch_train_loss = data['epoch_train_loss']
# ax.plot(epoch_train_loss[:], label='epoch_train_loss')
# epoch_train_acc = data['epoch_train_acc']
# ax.plot(epoch_train_acc[:], label='epoch_train_acc')
# epoch_val_loss = data['epoch_val_loss']
# ax.plot(epoch_val_loss[:], label='epoch_val_loss')
# epoch_val_acc = data['epoch_val_acc']
# ax.plot(epoch_val_acc[:], label='epoch_val_acc')
# epoch_train_stiffness_acc = data['epoch_train_stiffness_acc']
# ax.plot(epoch_train_stiffness_acc[:], label='epoch_train_stiffness_acc')
# epoch_val_stiffness_acc = data['epoch_val_stiffness_acc']
# ax.plot(epoch_val_stiffness_acc[:], label='epoch_val_stiffness_acc')
# epoch_train_roughness_acc = data['epoch_train_roughness_acc']
# ax.plot(epoch_train_roughness_acc[:], label='epoch_train_roughness_acc')
# epoch_val_roughness_acc = data['epoch_val_roughness_acc']
# ax.plot(epoch_val_roughness_acc[:], label='epoch_val_roughness_acc')
# ax.set_xlabel('epoch')
# ax.set_ylabel('acc')
# ax.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('save/'+'rep_result_'+'stats.png')
    