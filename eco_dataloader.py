import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class EtoxDataset(Dataset):
    def __init__(self, csv_path, concentration_threshold, word_embedding_path, fingerprint_path ):
        self.data = pd.read_csv(csv_path)
        # print(self.data.shape)
        self.chem_feats = pd.read_csv(fingerprint_path)
        self.spec_feats = pd.read_csv(word_embedding_path)
        self.num_sp = len(self.data['latin_name'].unique())
        self.num_ch = len(self.data['formatted_cas'].unique())
        # print(self.num_sp, self.num_ch)
        # print(list(self.data['formatted_cas'].unique()))
        # Create dictionaries for species and chemical IDs
        self.species_id = {name:idx for idx, name in enumerate(self.data['latin_name'].unique())}
        self.chemical_id = {cas:idx+self.num_sp for idx, cas in enumerate(self.data['formatted_cas'].unique())}

        # Create reverse lookup dictionaries
        list_species = list(zip(list(self.species_id.values()), list(self.species_id.keys())))
        list_chemicals = list(zip(list(self.chemical_id.values()), list(self.chemical_id.keys())))
        self.id2species = {idx: name for idx,name in list_species}   
        self.id2chemical = {idx: cas for idx,cas in list_chemicals}

        # Apply concentration threshold - create binary class 1: high 0: low
        self.data['concentration_class'] = (self.data['effect_value'] <= concentration_threshold).astype(int)

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cas = row['formatted_cas']
        sp_name = row['latin_name']
        duration = row['duration_hrs']
        conc_value = row['effect_value']
        conc_class = row['concentration_class']  # Binary class based on threshold

        # Fetch fingerprint descriptor
        chem_feature = self.chem_feats.loc[self.chem_feats['formatted_cas'] == cas].values[:,1:]

        # Fetch word embeddings
        sp_feature = self.spec_feats.loc[self.spec_feats['latin_name'] == sp_name].values[:,1:]

        # Use species and chemical IDs
        species_id = self.species_id[sp_name]
        chemical_id = self.chemical_id[cas]
    
        return sp_name, cas, species_id, chemical_id, duration, conc_value, conc_class, sp_feature, chem_feature
    
    
class Data_eco_variable(Dataset):
    def __init__(self, pos_links, neg_links, all_links):
        self.pos_links = pos_links
        self.neg_links = neg_links
        self.num_positives = pos_links.shape[1]
        self.num_negatives = neg_links.shape[1]
        self.num_all_links = all_links.shape[1]
        self.all_links = all_links
        # self.species_features = species_features
        # self.chemical_features = chemical_features

    def __len__(self):
        return max(self.num_positives, self.num_negatives, self.num_all_links)

    def __getitem__(self, index):
        positive_link = self.pos_links[:,index% self.num_positives]
        negative_link = self.neg_links[:,index% self.num_negatives]
        one_link = self.all_links[:, index % self.all_links.shape[1]]

        return torch.tensor(positive_link), torch.tensor(negative_link), torch.tensor(one_link)
    