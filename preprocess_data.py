#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

'''
Change Log:
16 Oct 2020: add prefix option for getData, getDataset, getLoader, allowing transformation/preprocessing methods like z-normalization. 
    Default: "" (i.e. Nothing) compatible with previous implementation
    "_znorm_": use z normalization for data preprocessing
22 Oct 2020: add data_type
    Default: 'B' for BioTac data
    'I' for iCub data, the num_features in chop_data is different
    Note: in 'c20icub' folder, which is generated by function `create_icub_data`,
    the files are named as bio_xxx for consistency when calling the function, but it actully contains icub data
30 Oct 2020: add num_samples

'''

# In[1]:


import os, sys
import numpy as np
import glob
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split


# In[2]:


num_sample = 50
num_row = 800
seq_len = 400 # (take from the middle)


# In[3]:


# generate pt data from cvs files right after parse_bag.sh in iiwa_stack_ws
# both MatName and SampleNum start from index 1, change to starting from 0
# run twice to ensure every csv is okay
def check_bio_csv_row_number(num_class, num_sample, data_dir, num_row):
    MatName = []
    for i in range(1, num_class+1):
        MatName.append("mat"+str(i))
    for mat in MatName:
        for sample in range(1, num_sample+1):
            bio_csv = data_dir + mat + "/" + mat + "_" + str(sample) +"_bio.csv"
            df = pd.read_csv(bio_csv, header=0, index_col=False)                
            # the header automatically follow when write to csv
            if df.shape[0] == (num_row-1):
                print("append one row to mat {}, sample {}".format(mat, sample))
                last_row = df.iloc[-1:]
                df = df.append(last_row)
                df.to_csv(bio_csv, index=False)
            elif df.shape[0] == (num_row+1):
                print("drop the last row to mat {}, sample {}".format(mat, sample))
                df = df.drop([df.index[-1]])
                df.to_csv(bio_csv, index=False)
            elif df.shape[0] != num_row:
                print(df.shape) # [800, 44]
                print("wierd shape at mat {}, sample {}".format(mat, sample))
            else:
                # row number is just nic
                pass


# In[4]:


def filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name="sample_dict.pkl", start=0, end=800):
    print("filter timestamp from {} to {}, seq_len = {}".format(start, end, end-start))
    MatName = []
    for i in range(1, num_class+1):
        MatName.append("mat"+str(i))
    data_dict = {}
    for mat in MatName:
        key = str(int(mat[3:])-1) # label start index from 0
        data_dict[key]={"x":[], "y":[]}
        for sample in range(1, num_sample+1):
            bio_csv = data_dir + mat + "/" + mat + "_" + str(sample) +"_bio.csv"
            df = pd.read_csv(bio_csv, header=0, index_col=False)                
            df = df.drop(labels=['timestamp'], axis=1)
#             print(df.shape) # (800, 44)
            
            # filter ele
            ele_cols = []
            for col in df.columns:
                if 'ele' in col:
                    ele_cols.append(col)
            df_ele = df.loc[:,ele_cols]
#             print(df_ele.shape) # (800, 19)

            df_ele_cut = df_ele.iloc[start:end]
#             print(df_ele_cut.shape) # (400, 19)
            
            # convert to numpy array and transpose
            arr = df_ele_cut.to_numpy().transpose()
#             print(arr.shape) # (19, 400)

            data_dict[key]["x"].append(arr)
            data_dict[key]["y"].append(int(key))
    
    print(data_dict.keys())
#     print(data_dict['0']["x"].shape, data_dict['0']["y"].shape) #(50, 19, 400), (50,)
    print(data_dict['1']["y"])
                
    with open(data_dir+dict_name, "wb") as fout:
        pickle.dump(data_dict, fout)


# ### create torch dataset

# In[5]:

def create_torch_dataset(data_dir, dict_name, dataset_dir, k=0, num_samples=None):
    
    # load all data from data_dict
    with open(data_dir+dict_name, "rb") as fin:
        data_dict = pickle.load(fin)
    X = []
    Y = []
    for key in data_dict:
        X.extend(data_dict[key]["x"])
        Y.extend(data_dict[key]["y"])
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape) # (2500, 19, 400) (2500,)
    
    # add on 30 Oct, allow reducing the total number of samples for transfer learning
    if num_samples is None:
        num_samples = Y.shape[0]
    tot_samples = Y.shape[0]
    discard_size = 1.0-num_samples/tot_samples
    print("create torch_dataset for {} samples, discard {} portion of tot samples".format(num_samples, discard_size))
    # perform random split, train:val:test = 6:2:2
    np.random.seed(k)
    if discard_size == 0:
        pass
    else:
        X, X_discard, Y, Y_discard = train_test_split(X, Y, test_size=discard_size, stratify=Y)
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.25, shuffle=True, stratify=Y_trainval)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
    
    # check if the dataset is balanced
    unique_elements, counts_elements = np.unique(Y_test, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))
    
    # save the data in .npy files for fast retrieval
    np.save(dataset_dir+"bio_all.npy", X)
    np.save(dataset_dir+"bio_train_" + str(k) + ".npy", X_train)
    np.save(dataset_dir+"bio_val_" + str(k) + ".npy", X_val)
    np.save(dataset_dir+"bio_test_" + str(k) + ".npy", X_test)
    np.save(dataset_dir+"label_all.npy", Y)
    np.save(dataset_dir+"label_train_" + str(k) + ".npy", Y_train)
    np.save(dataset_dir+"label_val_" + str(k) + ".npy", Y_val)
    np.save(dataset_dir+"label_test_" + str(k) + ".npy", Y_test)

    


# In[6]:
# add on 19 Oct to chop the seq len
def chop_data_seq_len(X_train, X_val, X_test, Y_train, Y_val, Y_test, seq_len, data_type='B'):
    current_len = X_train.shape[-1]
    assert X_val.shape[-1] == current_len and X_test.shape[-1] == current_len, "different length for train, val, test data"
    num_seg = current_len // seq_len
    print("chop org data of length {} into {} segments, each of which is has length {}".format(current_len, num_seg, seq_len))
    X_train_seg = []
    X_val_seg = []
    X_test_seg = []
    Y_train_seg = []
    Y_val_seg = []
    Y_test_seg = []
    for i in range(num_seg):
        X_train_seg.append(X_train[:,:,i*seq_len:(i+1)*seq_len])
        X_val_seg.append(X_val[:,:,i*seq_len:(i+1)*seq_len])
        X_test_seg.append(X_test[:,:,i*seq_len:(i+1)*seq_len])
        Y_train_seg.append(Y_train)
        Y_val_seg.append(Y_val)
        Y_test_seg.append(Y_test)

    if data_type == 'B':
        num_seg, num_sample, num_feature, seq_len = np.array(X_train_seg).shape
        X_train = np.array(X_train_seg).reshape((-1, num_feature, seq_len))
        X_val = np.array(X_val_seg).reshape((-1, num_feature, seq_len))
        X_test = np.array(X_test_seg).reshape((-1, num_feature, seq_len))
    elif data_type == 'I':
        num_seg, num_sample, H, W, seq_len = np.array(X_train_seg).shape
        X_train = np.array(X_train_seg).reshape((-1, H, W, seq_len))
        X_val = np.array(X_val_seg).reshape((-1, H, W, seq_len))
        X_test = np.array(X_test_seg).reshape((-1, H, W, seq_len))    

    Y_train = np.array(Y_train_seg).reshape((-1))
    Y_val = np.array(Y_val_seg).reshape((-1))
    Y_test = np.array(Y_test_seg).reshape((-1))
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
        
        


def get_TrainValTestData(path, k=0, prefix="", seq_len=400, data_type='B'):
    X_train = np.load(path + 'bio_train_' + prefix + str(k) + '.npy')
    X_val = np.load(path + 'bio_val_' + prefix + str(k) + '.npy')
    X_test = np.load(path + 'bio_test_' + prefix + str(k) + '.npy')
    Y_train = np.load(path + 'label_train_' + str(k) + '.npy')
    Y_val = np.load(path + 'label_val_' + str(k) + '.npy')
    Y_test = np.load(path + 'label_test_' + str(k) + '.npy')
    X_train, X_val, X_test, Y_train, Y_val, Y_test = chop_data_seq_len(X_train, X_val, X_test, Y_train, Y_val, Y_test, seq_len, data_type)
    return torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test), torch.FloatTensor(Y_train),torch.FloatTensor(Y_val), torch.FloatTensor(Y_test)


# In[7]:


def get_TrainValTestLoader(path, k=0, batch_size=8, shuffle=True, prefix="", seq_len=400, data_type='B'):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_TrainValTestData(path, k, prefix=prefix, seq_len=seq_len, data_type=data_type)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size)

    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=shuffle,batch_size=batch_size)
    
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


# In[ ]:


def get_TrainValTestDataset(path, k=0, batch_size=8, shuffle=True, prefix="", seq_len=400, data_type='B'):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_TrainValTestData(path, k, prefix=prefix, seq_len=seq_len, data_type=data_type)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    return train_dataset, val_dataset, test_dataset


# In[8]:


def create_loaders_50_50():
    num_class = 50
    data_dir = "/home/ruihan/Documents/mat_data_2020/"
    check_bio_csv_row_number(num_class, num_sample, data_dir, num_row)
    dict_name = "class_dict_50_50.pkl"     
    filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name, start=200, end=600)
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c50/"
    create_torch_dataset(data_dir, dict_name, dataset_dir, k=0)
    train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=0, batch_size=8, shuffle=True)


# In[9]:


# create train, val, test loader for the first 20 classes with new data
def create_loaders_20_50_new():
    num_class = 20
    data_dir = "/home/ruihan/Documents/mat_data_2020/"
    check_bio_csv_row_number(num_class, num_sample, data_dir, num_row)
    dict_name = "class_dict_20_50_new.pkl"  
    filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name, start=200, end=600)
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c20new/"
    create_torch_dataset(data_dir, dict_name, dataset_dir, k=0)
#     train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=0, batch_size=8, shuffle=True)


# In[10]:


def create_loaders_20_50():
    num_class = 20
    data_dir = "/home/ruihan/Documents/material_data_2020Mar/"
    check_bio_csv_row_number(num_class, num_sample, data_dir, num_row)
    dict_name = "class_dict_20_50.pkl"  
    filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name, start=200, end=600)
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c20/"
    create_torch_dataset(data_dir, dict_name, dataset_dir, k=0)
#     train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=0, batch_size=8, shuffle=True)


# In[11]:


# create_loaders_20_50()


# In[12]:


# create_loaders_20_50_new()


# In[13]:


# create_loaders_50_50()







# In[15]:


def create_torch_dataset_iCub(data_dir, dataset_dir, k=0, num_samples=None):
    # load all data from pt storage
    
    icub_all = torch.load(pt_data_dir + "ICUB_all.pt")
    bio_all = torch.load(pt_data_dir + "Bio_all.pt")
    labels_all = np.load(pt_data_dir + "all_labels.npy")
    
    X = np.array(icub_all.numpy())
    Y = np.array(labels_all)
    print("total samples")
    print(X.shape, Y.shape)
    
    # add on 30 Oct, allow reducing the total number of samples for transfer learning
    if num_samples is None:
        num_samples = Y.shape[0]
    tot_samples = Y.shape[0]
    discard_size = 1.0-num_samples/tot_samples
    print("create torch_dataset for {} samples, discard {} portion of tot samples".format(num_samples, discard_size))
    # perform random split, train:val:test = 6:2:2
    np.random.seed(k)
    if discard_size == 0:
        pass
    else:
        X, X_discard, Y, Y_discard = train_test_split(X, Y, test_size=discard_size, stratify=Y)
    
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.25, shuffle=True, stratify=Y_trainval)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
    
    # check if the dataset is balanced
    unique_elements, counts_elements = np.unique(Y_test, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))
    
    # save the data in .npy files for fast retrieval
    np.save(dataset_dir+"bio_all.npy", X)
    np.save(dataset_dir+"bio_train_" + str(k) + ".npy", X_train)
    np.save(dataset_dir+"bio_val_" + str(k) + ".npy", X_val)
    np.save(dataset_dir+"bio_test_" + str(k) + ".npy", X_test)
    np.save(dataset_dir+"label_all.npy", Y)
    np.save(dataset_dir+"label_train_" + str(k) + ".npy", Y_train)
    np.save(dataset_dir+"label_val_" + str(k) + ".npy", Y_val)
    np.save(dataset_dir+"label_test_" + str(k) + ".npy", Y_test)

# In[16]:

def create_icub_data():
    # Note: the folder name has icub, but files are still named for consistency
    pt_data_dir = "/home/ruihan/Desktop/HCNC/data/tot_data_Feb/"
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c20icub/"
    create_torch_dataset_iCub(pt_data_dir, dataset_dir, k=0)


# In[2]:


# pt_data_dir = "/home/ruihan/Desktop/HCNC/data/tot_data_Feb/"
# dataset_dir = "/home/ruihan/Desktop/HCNC/data/c20icub/"
# create_torch_dataset_iCub(pt_data_dir, dataset_dir, k=0)


# ### create icub dataset of reduced size, for transfer learning

# In[3]:


def create_reduced_c20icub(num_samples_list):
    pt_data_dir = "/home/ruihan/Desktop/HCNC/data/tot_data_Feb/"
    dataset_parent_dir = "/home/ruihan/Desktop/HCNC/data/c20icub_n200/"
    for num_samples in num_samples_list:
#         num_samples = int(dataset_dir.split("_n")[-1][:-1])
        dataset_dir = dataset_parent_dir + "c20icub_n{}/".format(str(num_samples))
        if os.path.exists(dataset_dir):
            print("{} exists, update in place".format(dataset_dir))
        else:
            print("mkdir {} for datasets".format(dataset_dir))
            os.mkdir(dataset_dir)
        print("create c20icub dataset of {} samples".format(num_samples))
        create_torch_dataset_iCub(pt_data_dir, dataset_dir, k=0, num_samples=num_samples)
    
# num_samples_list = [200, 400, 600, 800]
# create_reduced_c20icub(num_samples_list)


# ### create BIoTac 30_50, different classes from icub

# In[4]:


# create train, val, test loader for the first 20 classes with new data
def create_loaders_30_50():
    num_class = 30
    data_dir = "/home/ruihan/Documents/mat_data_2020/"
    dict_name = "class_dict_30_50.pkl"  
    filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name, start=200, end=600, class_begin_idx=21)
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c30/"
    create_torch_dataset(data_dir, dict_name, dataset_dir, k=0)
#     train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=0, batch_size=8, shuffle=True)
# create_loaders_30_50()


# ### combine c20 and c20 new, assume they have the same label

# In[5]:


def combine_c20_c20new(data_dir):
    prefix = ""
    sequence_length = 400
    # c20
    data_folder = "c20/"
    dataset_dir = os.path.join(data_dir, data_folder)
    X_train_c20, X_val_c20, X_test_c20, Y_train_c20, Y_val_c20, Y_test_c20 = get_TrainValTestData(dataset_dir, k=0, prefix=prefix,seq_len=sequence_length)
    # c20
    data_folder = "c20new/"
    dataset_dir = os.path.join(data_dir, data_folder)
    X_train_c20new, X_val_c20new, X_test_c20new, Y_train_c20new, Y_val_c20new, Y_test_c20new = get_TrainValTestData(dataset_dir, k=0, prefix=prefix,seq_len=sequence_length)
    
    X_train = torch.cat((X_train_c20, X_train_c20new), 0)
    X_val = torch.cat((X_val_c20, X_val_c20new), 0)
    X_test = torch.cat((X_test_c20, X_test_c20new), 0)
    Y_train = torch.cat((Y_train_c20, Y_train_c20new), 0)
    Y_val = torch.cat((Y_val_c20, Y_val_c20new), 0)
    Y_test = torch.cat((Y_test_c20, Y_test_c20new), 0)
    X = torch.cat((X_train, X_val, X_test), 0)
    Y = torch.cat((Y_train, Y_val, Y_test), 0)
    
    data_folder = "c20comb/"
    k = 0
    dataset_dir = os.path.join(data_dir, data_folder)
    # save the data in .npy files for fast retrieval
    np.save(dataset_dir+"bio_all.npy", X)
    np.save(dataset_dir+"bio_train_" + str(k) + ".npy", X_train)
    np.save(dataset_dir+"bio_val_" + str(k) + ".npy", X_val)
    np.save(dataset_dir+"bio_test_" + str(k) + ".npy", X_test)
    np.save(dataset_dir+"label_all.npy", Y)
    np.save(dataset_dir+"label_train_" + str(k) + ".npy", Y_train)
    np.save(dataset_dir+"label_val_" + str(k) + ".npy", Y_val)
    np.save(dataset_dir+"label_test_" + str(k) + ".npy", Y_test)

# combine_c20_c20new(data_dir="/home/ruihan/Desktop/HCNC/data/")
