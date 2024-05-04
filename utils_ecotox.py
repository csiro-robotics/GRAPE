from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pubchempy as pcp
import numpy as np
import pandas as pd 
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
# from args_parser import arg_parser_ecotox
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords, words, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import inflect

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def format_cas(cas_number):
    cas_number = str(cas_number)
    # Insert hyphens at the appropriate positions in the CAS number
    formatted_cas_number = '-'.join([cas_number[:-3], cas_number[-3:-1], cas_number[-1]])

    return formatted_cas_number


def fill_the_gap(fp, err_inds, corr_inds, num_ents):
    fp_len = fp.shape[1]
    fmat = np.zeros(shape=(num_ents, fp_len))    
    for j in range(num_ents):
        if j in err_inds:
            temp_ft = np.zeros(shape=(fp_len))
        elif j in corr_inds:
            index = corr_inds.index(j)
            temp_ft = fp[index, :]
        fmat[j,:] = temp_ft
    return fmat

def compute_fingerprint_all(df):
    fingerprints = {}
    
    for i,cas in enumerate(df['formatted_cas'].unique()):
        if i%50==0:
            print(i)
        
        
        try:
            print(cas)
            mol = Chem.MolFromSmiles(pcp.Compound.from_cid(pcp.get_cids(cas, 'name')[0]).isomeric_smiles)
            
            # compute Morgan fingerpring using rdkit
            fp1_arr = np.zeros((0,), dtype=np.int8)    
            fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            DataStructs.ConvertToNumpyArray(fp_obj,fp1_arr)
            
            # compute cactvs fingerpring from pubchempy
            compound = pcp.get_compounds(cas, 'name')[0]
            fp2_hex = compound.cactvs_fingerprint # cactvs is returned as a hex-encoded string
            fp2_arr = np.array(list(map(np.int8, fp2_hex)))
            
        except: 
            # if molecule doesn't have encoded fingerpint
            fp1_arr = np.zeros(1024, dtype=np.int8)    
            fp2_arr = np.zeros(881, dtype=np.int8) 
        
        fp_arr = np.zeros(1024+881, dtype= np.int8)
        fp_arr[:1024] = fp1_arr
        fp_arr[1024:] = fp2_arr
        fingerprints[cas] = fp_arr

    fp_columns = ['fp_%d' % i for i in range(1024+881)]
    fingerprints = pd.DataFrame.from_dict(fingerprints, orient='index', columns=fp_columns)

    fingerprints = fingerprints.reset_index().rename(columns={'index':'formatted_cas'})
    return fingerprints

def get_wordnet_pos(tag):
    # Map POS tag to first character used by WordNetLemmatizer
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def fetch_wikipedia_text(species_name):
    try:
        url = f'https://en.wikipedia.org/wiki/{species_name}'
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ''
        for paragraph in soup.find_all('p'):
            text += paragraph.text
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {species_name}: {str(e)}")
        return None
    
def preprocess_text(text, lemmatizer):
    text = re.sub(r'\[.*?\]+', '', text) # Remove citation numbers
    text = text.replace('\n', '') # Remove newline characters
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuations
    text = text.lower()  # Convert to lower case
    p = inflect.engine()
    text = re.sub('(\d+)', lambda m: p.number_to_words(m.group()), text) # Replace numbers with words

    # Remove stop words   
    text_tokens = word_tokenize(text)
    tokens_without_sw = ' '.join([word for word in text_tokens if not word in stopwords.words()])
    tagged_words = [nltk.pos_tag(word_tokenize(tokens_without_sw))]
    lemmatized_words = [[lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged] for tagged in tagged_words]
    lemmatized_sentences = [' '.join(words) for words in lemmatized_words]
    text = re.sub(r'[^\w\s]', '', lemmatized_sentences[0]) # Remove punctuations

    return text


def get_spec_feats_dict(sp_list):
    error_species_ind = []
    found_species_ind = []
    all_text = []
    lemmatizer = WordNetLemmatizer()

    for i,sp in enumerate(sp_list):
        if i%10==0:
            print(i)

        text = fetch_wikipedia_text(sp)

        if text is None:
            error_species_ind.append(i)
            continue
        else:
            # Preprocess the text
            text = preprocess_text(text, lemmatizer)

            found_species_ind.append(i)

            all_text.append(text)

    return all_text, found_species_ind, error_species_ind

def get_word_embedding(df):
    vectorizer = TfidfVectorizer()
    species_list = df['latin_name'].unique().tolist()
    sp_corpus, found_species_ind, error_species_ind = get_spec_feats_dict(species_list)
    sp_fp_sparse = vectorizer.fit_transform(sp_corpus)
    sp_fp_dense = sp_fp_sparse.toarray()
    sp_fp_len = sp_fp_dense.shape[1]
    
    sp_fmat = fill_the_gap(sp_fp_dense, error_species_ind, found_species_ind, len(species_list))
    
    word_embeddings = {}
    for row in range(len(species_list)):
        word_embeddings[species_list[row]] = sp_fmat[row,:]

    we_columns = ['we_%d' % i for i in range(sp_fmat.shape[1])]
    word_embeddings_df = pd.DataFrame.from_dict(word_embeddings, orient='index', columns=we_columns)

    word_embeddings_df = word_embeddings_df.reset_index().rename(columns={'index':'latin_name'})
    return sp_fmat, sp_fmat.shape,  word_embeddings, word_embeddings_df
