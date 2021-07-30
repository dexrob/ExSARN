import numpy as np
import os, sys
import shutil

# create c20comb kfold
def create_c20comb_kfold():
    for i in range(10):
        for name in ['fold']:
            x_c20 = np.load('data/c20/'+'20_fold/'+"bio_" + name + '_' + str(i) + ".npy")
            y_c20 = np.load('data/c20/'+'20_fold/'+"label_" + name + '_' + str(i) + ".npy")
            print(x_c20.shape)
            x_c20new = np.load('data/c20new/'+'20_fold/'+"bio_" + name + '_' + str(i) + ".npy")
            y_c20new = np.load('data/c20new/'+'20_fold/'+"label_" + name + '_' + str(i) + ".npy")
            print(x_c20new.shape)
            x_c20comb = np.concatenate((x_c20, x_c20new), axis=0)
            y_c20comb = np.concatenate((y_c20, y_c20new), axis=0)
            print(x_c20comb.shape)
            print(y_c20comb.shape)
            np.save('data/c20comb/'+'20_fold/'+"bio_" + name + "_" + str(i) + ".npy", x_c20comb)
            np.save('data/c20comb/'+'20_fold/'+"label_" + name + "_" + str(i) + ".npy", y_c20comb)       


def generate_10fold_dataset(dataset_dir, num_class=20, K=5):
    X_all = np.load(dataset_dir + "bio_all.npy")
    Y_all = np.load(dataset_dir + "label_all.npy")
    # print(X_all.shape)
    # print(Y_all.shape)

    # print("save to ~/Desktop/Y_all.txt")
    # fname = "/Users/tian/Desktop/Y_all.txt"
    # np.savetxt(fname, Y_all)
    # # assume labels are contiguous (0,0,0,...1,1,1...,19,19,19,...), uncomment the following to check
    print(Y_all)
    print("check if Y_all is contiguous")
    num_sample_per_class = Y_all.shape[0]//num_class
    print('num_sample_per_class', num_sample_per_class)
    # generate index for one class
    ind = np.random.choice(range(num_sample_per_class), size=(K,int(num_sample_per_class/K)), replace=False)
    K, num_fold_per_fold = ind.shape
    print('K, num_fold_per_fold', K, num_fold_per_fold)
    for k in range(K):
        fold_idx = ind[k,:]
        X = []
        Y = []
        for j in range(num_class):
            class_fold_idx = num_sample_per_class*j+fold_idx
            X.extend(X_all[class_fold_idx])
            Y.extend(Y_all[class_fold_idx])
        X = np.array(X)
        Y = np.array(Y)
        print('x', X.shape, 'y', Y.shape)
        np.save(dataset_dir+'20_fold/'+"bio_fold_" + str(k) + ".npy", X)
        np.save(dataset_dir+'20_fold/'+"label_fold_" + str(k) + ".npy", Y)
    # separate train, val, test
    # fold_set = np.arange(K)
    # for i in range(K):
    #     test_idx = i
    #     val_idx = test_idx+1 if test_idx<(K-1) else 0 
    #     train_set = np.delete(fold_set,  np.where((fold_set == test_idx) | (fold_set == val_idx)))
    #     # direct copy for test/val loader
    #     for loader in ["test", "val"]:
    #         infile = dataset_dir+'20_fold/'+"bio_fold_" + str(i) + ".npy"
    #         outfile = infile.replace("_fold_", '_'+loader+'_')
    #         shutil.copy2(infile, outfile)

    #         infile = dataset_dir+'20_fold/'+"label_fold_" + str(i) + ".npy"
    #         outfile = infile.replace("_fold_", '_'+loader+'_')
    #         shutil.copy2(infile, outfile)
    #     # concatenate the rest of the files for train loader
    #     X_train = []
    #     Y_train = []
    #     for train_idx in train_set:
    #         X_train.extend(np.load(dataset_dir+'20_fold/'+"bio_fold_" + str(train_idx) + ".npy"))
    #         Y_train.extend(np.load(dataset_dir+'20_fold/'+"label_fold_" + str(train_idx) + ".npy"))

    #     np.save(dataset_dir+'20_fold/'+"bio_train_" + str(i) + ".npy", np.array(X_train))
    #     np.save(dataset_dir+'20_fold/'+"label_train_" + str(i) + ".npy", np.array(Y_train))

def generate_Kfold_dataset(dataset_dir, num_class=20, K=5):
    X_all = np.load(dataset_dir + "bio_all.npy")
    Y_all = np.load(dataset_dir + "label_all.npy")
    print(X_all.shape)
    print(Y_all.shape)

    # print("save to ~/Desktop/Y_all.txt")
    # fname = "/Users/tian/Desktop/Y_all.txt"
    # np.savetxt(fname, Y_all)
    # # assume labels are contiguous (0,0,0,...1,1,1...,19,19,19,...), uncomment the following to check
    print(Y_all)
    print("check if Y_all is contiguous")
    num_sample_per_class = Y_all.shape[0]//num_class
    print('num_sample_per_class', num_sample_per_class)
    # generate index for one class
    ind = np.random.choice(range(num_sample_per_class), size=(K,int(num_sample_per_class/K)), replace=False)
    K, num_fold_per_fold = ind.shape
    print('K, num_fold_per_fold', K, num_fold_per_fold)
    for k in range(K):
        fold_idx = ind[k,:]
        X = []
        Y = []
        for j in range(num_class):
            class_fold_idx = num_sample_per_class*j+fold_idx
            X.extend(X_all[class_fold_idx])
            Y.extend(Y_all[class_fold_idx])
        X = np.array(X)
        Y = np.array(Y)
        print('x', X.shape, 'y', Y.shape)
        np.save(dataset_dir+"bio_fold_" + str(k) + ".npy", X)
        np.save(dataset_dir+"label_fold_" + str(k) + ".npy", Y)
    # separate train, val, test
    fold_set = np.arange(K)
    for i in range(K):
        test_idx = i
        val_idx = test_idx+1 if test_idx<(K-1) else 0 
        train_set = np.delete(fold_set,  np.where((fold_set == test_idx) | (fold_set == val_idx)))
        # direct copy for test/val loader
        for loader in ["test", "val"]:
            infile = dataset_dir+"bio_fold_" + str(i) + ".npy"
            outfile = infile.replace("_fold_", '_'+loader+'_')
            shutil.copy2(infile, outfile)

            infile = dataset_dir+"label_fold_" + str(i) + ".npy"
            outfile = infile.replace("_fold_", '_'+loader+'_')
            shutil.copy2(infile, outfile)
        # concatenate the rest of the files for train loader
        X_train = []
        Y_train = []
        for train_idx in train_set:
            X_train.extend(np.load(dataset_dir+"bio_fold_" + str(train_idx) + ".npy"))
            Y_train.extend(np.load(dataset_dir+"label_fold_" + str(train_idx) + ".npy"))

        np.save(dataset_dir+"bio_train_" + str(i) + ".npy", np.array(X_train))
        np.save(dataset_dir+"label_train_" + str(i) + ".npy", np.array(Y_train))


K = 10
parent_dataset_dir = "data/"
for data_folder in ["c20icub/"]:
    X = []
    Y = []
    num_class = int(data_folder[1:3])
    print("process data_dolder {}, num_class {}".format(data_folder, num_class))
    dataset_dir = parent_dataset_dir + data_folder
    generate_10fold_dataset(dataset_dir, num_class=num_class, K=K)
K = 5
for data_folder in ["c20icub/"]:
    X = []
    Y = []
    num_class = int(data_folder[1:3])
    print("process data_dolder {}, num_class {}".format(data_folder, num_class))
    dataset_dir = parent_dataset_dir + data_folder
    generate_Kfold_dataset(dataset_dir, num_class=num_class, K=K)

# create_c20comb_kfold() 

for i in range(10):
    for data_folder in ["c20icub/"]:
        print(data_folder)
        y = np.load('data/'+data_folder+'20_fold/'+'label_fold'+'_'+str(i)+'.npy')
        print(y.shape)
        print(np.unique(y, return_counts=True))

for i in range(5):
    for data_folder in ["c20icub/"]:
        print(data_folder)
        for name in ['train', 'val', 'test']:
            y = np.load('data/'+data_folder+'label_'+name+'_'+str(i)+'.npy')
            print(y.shape)
            print(np.unique(y, return_counts=True))

# for data_folder in ["c20new/"]:
#     X = []
#     Y = []
#     num_class = int(data_folder[1:3])
#     print("process data_dolder {}, num_class {}".format(data_folder, num_class))
#     dataset_dir = parent_dataset_dir + data_folder
#     generate_Kfold_dataset(dataset_dir, num_class=num_class, K=K)