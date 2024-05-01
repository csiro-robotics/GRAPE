from args_parser import arg_parser_ecotox
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils_ecotox import compute_fingerprint_all, get_word_embedding
from eco_dataloader import EtoxDataset


def func_normalize_fit(train_data):
    # We normalize the 3 types of features independently, species node features, chemical node features, and edge features
    scaler_duration = StandardScaler()
    scaler_sp_feature = StandardScaler()
    scaler_chem_feature = StandardScaler()

    train_data_temp = [(*item[:4], torch.tensor(item[4].astype(np.float32)), item[5], item[6], torch.tensor(item[7].astype(np.float32)), torch.tensor(item[8].astype(np.float32))) for item in train_data]
    durations = torch.cat([item[4].reshape(-1, 1) for item in train_data_temp], dim=0)
    sp_features = torch.cat([item[7] for item in train_data_temp], dim=0)
    chem_features = torch.cat([item[8] for item in train_data_temp], dim=0)

    scaler_duration.fit(durations.to(torch.float32))
    scaler_sp_feature.fit(sp_features.to(torch.float32))
    scaler_chem_feature.fit(chem_features.to(torch.float32))

    return scaler_duration, scaler_sp_feature, scaler_chem_feature

def func_normalize_transform(scaler_duration, scaler_sp_feature, scaler_chem_feature, data):
    norm_data = [(item[0], item[1], item[2], item[3], scaler_duration.transform(item[4].reshape(-1, 1).astype(np.float32)), item[5], item[6], torch.tensor(scaler_sp_feature.transform(item[7].astype(np.float32))), torch.tensor(scaler_chem_feature.transform(item[8].astype(np.float32)))) for item in data]

    return norm_data

def func_prepare_data(tuple_list):
    data_selection = []
    for rows in tuple_list:
        data_row = []
        data_row.extend([rows[2],rows[3], rows[4][0,0], rows[5], rows[6]])
        data_row.extend(rows[7].numpy().tolist()[0])
        data_row.extend(rows[8].numpy().tolist()[0])
        
        data_selection.append(data_row)
        
    return np.array(data_selection)


if __name__ == '__main__':
    
    args = arg_parser_ecotox()
    # print(args)
    
    df = pd.read_csv(args.ecotox_file)
    df.head()
    
    if args.compute_feats==1:
        sp_fmat, sp_fmat.shape, word_embeddings, word_embeddings_df = get_word_embedding(df)
        word_embeddings_df.to_csv(args.u_filename_raw, index=False)
    
        fingerprints = compute_fingerprint_all(df)
        fingerprints.to_csv(args.v_filename_raw, index=False)
    
    
    word_embeddings_df1 = pd.read_csv(args.u_filename_raw)
    fingerprints = pd.read_csv(args.v_filename_raw)
    
    dataset = EtoxDataset(args.ecotox_file, args.conc_threshold, args.u_filename_raw, args.v_filename_raw)

    # Define sizes for train, validation and test splits
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
   
    spec_feats_arr = dataset.spec_feats.loc[:,'we_0':'we_'+str(word_embeddings_df1.shape[1]-2)].values
    chem_feats_arr = dataset.chem_feats.loc[:, 'fp_0':'fp_'+str(fingerprints.shape[1]-2)].values
    
    if args.save_feats==1:
        np.save(args.u_filename, spec_feats_arr)
        np.save(args.v_filename, chem_feats_arr) 
    
    # manual_seed(0) for split 1, manual_seed(10) for split 2 and manual_seed(100) for split 3
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(0))
    scaler_d, scaler_sp, scaler_ch = func_normalize_fit(train_dataset)

    train_data_normalized = func_normalize_transform(scaler_d, scaler_sp, scaler_ch, train_dataset)
    val_data_normalized = func_normalize_transform(scaler_d, scaler_sp, scaler_ch, val_dataset)
    test_data_normalized = func_normalize_transform(scaler_d, scaler_sp, scaler_ch, test_dataset)
 
    # If needed - but memory intensive: save all data as is using pickle
    # with open(args.train_norm_pkl', 'wb') as file:
    #     pickle.dump(train_data_normalized, file, protocol=4)
    
    # with open(args.val_norm_pkl', 'wb') as file:
    #     pickle.dump(val_data_normalized, file, protocol=4)
        
    # with open(args.test_norm_pkl', 'wb') as file:
    #     pickle.dump(test_data_normalized, file, protocol=4)
    
    # prepare data splits with normalized features as np.array
    train_select = func_prepare_data(train_data_normalized)
    val_select = func_prepare_data(val_data_normalized)
    test_select = func_prepare_data(test_data_normalized)
    
    # If needed - but memory intensive
    # # save all data including concatenated features
    # np.savetxt(args.train_norm_all_file, train_select, delimiter=',')
    # np.savetxt(args.val_norm_all_file, val_select, delimiter=',')
    # np.savetxt(args.test_norm_all_file, test_select, delimiter=',')
    
    # save only interactions and duration (edge feature)
    np.savetxt(args.train_file, train_select[:,:5], delimiter=',')
    np.savetxt(args.val_file, val_select[:,:5], delimiter=',')
    np.savetxt(args.test_file, test_select[:,:5], delimiter=',')     
        
