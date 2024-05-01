import argparse

def arg_parser_ecotox():

    parser = argparse.ArgumentParser(description='ecotox data preparation')
    parser.add_argument('--ecotox_rawdata_file', default='GRAPE_eToxIQ_data/ecotox_prep/envirotox_20230324.csv', help='file containing tests')
    parser.add_argument('--ecotox_file', default='GRAPE_eToxIQ_data/ecotox_prep/ecotox_cleaned_file.csv')
    
    
    parser.add_argument('--compute_feats', default='0', type=int, help='make it 1 for first time computing u and v features otherwise 0')
    parser.add_argument('--u_filename_raw', default='GRAPE_eToxIQ_data/ecotox_feats/word_embeddings_allsp_wiki.csv')
    parser.add_argument('--v_filename_raw', default='GRAPE_eToxIQ_data/ecotox_feats/fingerprints_allchems.csv')
    parser.add_argument('--conc_threshold', default=10.0, type=float, help='use to discretize concentration values in 2 classes')
    parser.add_argument('--save_feat', default=0, help='set as 1 if running code for first time to save processed node features, else 0')
    parser.add_argument('--u_filename', default='GRAPE_eToxIQ_data/ecotox_feats/spec_feats_arr.npy')
    parser.add_argument('--v_filename', default='GRAPE_eToxIQ_data/ecotox_feats/chem_feats_arr.npy')
   
        
    parser.add_argument('--train_file',default='GRAPE_eToxIQ_data/ecotox_data_split1/train_interactions.csv')
    parser.add_argument('--val_file', default='GRAPE_eToxIQ_data/ecotox_data_split1/val_interactions.csv')
    parser.add_argument('--test_file', default='GRAPE_eToxIQ_data/ecotox_data_split1/test_interactions.csv')
    
    parser.add_argument('--train_norm_all_file', default='GRAPE_eToxIQ_data/ecotox_data_split1/norm_train.csv', help='concatenate interactions with features for other ML methods')
    parser.add_argument('--val_norm_all_file', default='GRAPE_eToxIQ_data/ecotox_data_split1/norm_val.csv', help='concatenate interactions with features for other ML methods')
    parser.add_argument('--test_norm_all_file', default='GRAPE_eToxIQ_data/ecotox_data_split1/norm_test.csv', help='concatenate interactions with features for other ML methods')
    parser.add_argument('--train_norm_pkl', default='GRAPE_eToxIQ_data/ecotox_data_split1/train_data_normalized.pkl')
    parser.add_argument('--val_norm_pkl', default='GRAPE_eToxIQ_data/ecotox_data_split1/val_data_normalized.pkl')
    parser.add_argument('--test_norm_pkl', default='GRAPE_eToxIQ_data/ecotox_data_split1/test_data_normalized.pkl')    
    
    # set these arguments for training parameters and file paths    
    
    parser.add_argument('--batch_size', default='32', type=int, help='batch size')
    parser.add_argument('--epochs', default='100', type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.0001, type= float, help='learning rate')
    parser.add_argument('--outdim', default='32', type=int, help='output dimension if using mlps')
    parser.add_argument('--hdim1', default='128', type=int, help='hidden layer dimension if using mlp_u')
    parser.add_argument('--hdim2', default='128', type=int, help='hidden layer dimension if using mlp_v')
    parser.add_argument('--hdim3', default='128', type=int, help='hidden layer dimension if using gnn')
    parser.add_argument('--p_gcn', default=0.2, type= float, help='dropout rate for gcn')
    parser.add_argument('--model_folder',default='GRAPE_learned_models/eco_learned_models_split1/', help='folder path to save learned models')
    parser.add_argument('--save_after_ep', default=10, type=int, help='save model after these epochs')
    parser.add_argument('--name_str', help='prefix for model path for every run')
    parser.add_argument('--train_flag', default=1, help='make it 0 if want to inference from model, but feed model path')
    parser.add_argument('--results_folder', default='GRAPE_results/split1/', help='results folder to save predictions from trained model with train_flag as 0')
    parser.add_argument('--best_models_file', default='GRAPE_learned_models/best_models.txt',help='it stores the model name with the epoch that gives best validation auc')
    parser.add_argument('--line_num', type=int, help='To read from list of best models written in best_models.txt, supply the line number corresponding to the model to be loaded for inferencing')
    args = parser.parse_args() 
    
    return args