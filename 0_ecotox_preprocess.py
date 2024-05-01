from args_parser import arg_parser_ecotox
import numpy as np
import pandas as pd
from utils_ecotox import format_cas

if __name__ == '__main__':
    
    args = arg_parser_ecotox()
    print(args)
    
   
    data_csv = pd.read_csv(args.ecotox_rawdata_file)

    # Step 1: make duration, effect_value, effect, chemical names and cas consistent
    data_csv['duration_hrs'] = data_csv['Duration (hours)']
    data_csv['Effect1'] = data_csv['Effect'].str.extract(r'([a-zA-Z]+)')
    data_csv['effect'] = data_csv['Effect1'].str.lower()
    data_csv['Reported_chemical_name1'] = data_csv['Reported chemical name'].str.lower()
    data_csv['Reported_chemical_name'] = data_csv['Reported_chemical_name1'].str.split(';').str[0]
    data_csv['effect_value'] = data_csv['Effect value']
    data_csv['formatted_cas'] = data_csv['original CAS'].apply(format_cas)

    # Step 2: apply filter to extract fishMLC50 data
    fishMlc50_df = data_csv.loc[(data_csv['Test statistic'] == 'LC50') & (data_csv['Trophic Level'] == 'FISH') & (data_csv['effect'] == 'mortality')]

    # Step 3: process species latin_names to allow wikipedia webscraping : format - 'familyName_genusName' 
    fishMlc50_df.loc[:, 'latin_name_'] = fishMlc50_df['Latin name'].str.lower()
    fishMlc50_df['latin_name1'] = fishMlc50_df['latin_name_'].apply(lambda x: ' '.join(word for word in x.split()[:next((i for i, word in enumerate(x.split()) if 'ssp' in word), len(x.split()))]))
    fishMlc50_df.loc[:,'latin_name'] = fishMlc50_df['latin_name1'].apply(lambda x: x[:-3].strip() if x.endswith(' sp') else '_'.join(x.split()).strip()) 

    # Step 4: Drop all extra columns
    fishMlc50_df = fishMlc50_df.drop(['CAS', 'Chemical name', 'Source', 'version', 'Duration', 'Duration (days)', 'Effect', 'Effect1', 'Effect value','Duration (hours)','latin_name_', 'latin_name1', 'Latin name', 'Reported chemical name', 'Trophic Level', 'Effect is 5X above water solubility', 'Test type', 'Unit', 'Test statistic', 'original CAS', 'effect', 'Reported_chemical_name1'], axis=1)

    # Step 5: group data such that one-to-one relation between formatted_cas and Reported_chemical_name
    # group data such that the median value of effect_value concentration is associated to each unique experimental setting
    g_fishMlc50_df = fishMlc50_df.groupby(['formatted_cas', 'latin_name', 'duration_hrs'])['Reported_chemical_name','effect_value'].agg({'Reported_chemical_name': 'first', 'effect_value': 'median'})
    g_fishMlc50_df.to_csv(args.ecotox_file, index=True)