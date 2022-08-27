'''
Merchant Name Clustermaker, Namer and Validator
Author: Alexander Rosales
Company: RES
August 8, 2022

For use in Tableau

This script deduplicates, groups, names and validates identified clusters of similar merchant names. Naming is done by taking the most frequent
'Pass1' name in each cluster. Validation is done by checking the expense type distribution of each unique 'Pass2' name
in each cluster against the expense type distribution of the current cluster.


In progress: 
    Documentation

'''

import pandas as pd
import numpy as np
import re
import string
from rapidfuzz import fuzz
from sklearn.cluster import AffinityPropagation
from string_grouper import match_strings

def merchant_name_standardization(data):
    """
    This function prepares data for Levenshtein distance clustering by removing all special characters, making all characters lowercase, removing stopwords,
    and filtering out by known company names

        Input: data - a mx1 pandas dataframe with column title = 'Merchant'

        Output: a mx3 pandas dataframe with columns = ['Merchant','Pass1','Pass2'] containing the original merchant names, merchant names standardized
        once (lowercased, special characters removed, whitespace stripped), and merchant names standardized twice (stopwords removed, known merchant names sorted)
    """
    # Remove stopword (common words in many merchant names) to make unique parts of merchant names stand out
    vendor_stopwords=['biz', 'bv', 'co', 'comp', 'company', 
                'corp','corporation', 'dba', 
                'inc', 'incorp', 'incorporat', 
                'incorporate', 'incorporated', 'incorporation', 
                'international', 'intl', 'intnl', 
                'limited' ,'llc', 'ltd', 'llp', 
                'machines', 'pvt', 'pte', 'private','www','com','org','net',
                'restaurant','restaurants','restauran', 'gov', 'mart','express',
                'sq','tst','online','shop','service','services','the',
                'grill','airport','equipment','cafe','sp','paypal','hardware','house','hotel',
                'american','in']

    proscessed_m = data

    # Ensure there are no empty merchant names
    proscessed_m['Merchant'] = proscessed_m['Merchant'].replace(r'^\s*$', 'Blank Merchant', regex=True)

    # remove special characters, standardize whitespace, remove numbers longer than 2 digits, remove everything after a hashtag
    proscessed_m['Pass1'] = proscessed_m["Merchant"].astype(str)
    proscessed_m['Pass1'] = proscessed_m['Pass1'].str.replace('\d{2,}',' ', regex=True)
    proscessed_m['Pass1'] = proscessed_m['Pass1'].str.replace('#.*$','',regex=True)

    replace = {ord(elm):" " for elm in string.punctuation}
    proscessed_m['Pass1'] = proscessed_m["Pass1"].apply(lambda x: x.translate(replace).lower())
    proscessed_m['Pass1'] = proscessed_m['Pass1'].str.replace(' +', ' ',regex=True)
    proscessed_m['Pass1'] = proscessed_m['Pass1'].str.strip()

    # Remove common stopwords 
    proscessed_m['Pass2'] = proscessed_m['Pass1']
    for word in vendor_stopwords:
        pattern =  r'\b'+word+r'\b' 
        proscessed_m['Pass2']=proscessed_m['Pass2'].apply(lambda y: re.sub(pattern,' ',y))

    # Sort known big company names
    known_vendor_names = {'amazon':['amazon','amzn'],'airbnb':['airbnb'],'lyft':['lyft'],'doordash':['doordash'],
                            'uber':['uber trip', 'uber eats'],'starbucks':['starbucks'], 'msft':['microsoft']
                         }

    # Finds merchant names containing strings which the keys in known_vendor_names are mapped to and then maps those merchant names to the aforementioned keys          
    for replacement_word in known_vendor_names.keys():
        for sub in known_vendor_names[replacement_word]:
            replacement_regex = re.compile(rf'(^.*{sub}.*$)')
            proscessed_m['Pass2'] = proscessed_m['Pass2'].str.replace(replacement_regex, replacement_word ,regex=True)
            proscessed_m['Pass1'] = proscessed_m['Pass1'].str.replace(replacement_regex, replacement_word ,regex=True)  # this line not tested

    # standardize and strip extra whitespace
    proscessed_m['Pass2'] = proscessed_m['Pass2'].str.replace(' +', ' ',regex=True)
    proscessed_m['Pass2'] = proscessed_m['Pass2'].str.strip()
    
    # ensure no cells are empty
    proscessed_m.loc[proscessed_m['Pass2'] == '','Pass2'] = proscessed_m['Merchant'].apply(lambda x: x.translate(replace).lower())
    proscessed_m.loc[proscessed_m['Pass1'] == '','Pass1'] = proscessed_m['Merchant'].apply(lambda x: x.translate(replace).lower())

    return proscessed_m

def fuzz_measure(s1, s2):
    """
    Calculates the harmonic mean between the fuzzy token set ratio and the fuzzy partial ratio 

        Inputs: s1, s2 - strings

        Outputs -  a double between 0 and 100
    """
    r1 = fuzz.token_set_ratio(s1,s2) + 0.00000000001
    r2 = fuzz.partial_ratio(s1,s2) + 0.00000000001
    return 2*r1*r2/(r1+r2)

def fuzz_similarity(col):
    """
    Produces a similarity matrix that measures how similar each merchant name in col is similar to every other merchant name in col
        Inputs: col -  a nx1 pandas dataframe containing merchant names in the 'Pass2' as given by merchant_name_standardization
        Outputs: sim_array - an nxn matrix filled with doubles from 0 to 100 (100 on diagonal) where each sim_array[i][j] represents the fuzz_similarity() measure
        between string i and j in col.
    """
    merch_names = col.to_numpy()
    sim_array = pd.DataFrame(columns = merch_names,index=merch_names)

    # Apply fuzz_measure with column name, index name as arguments for each cell.
    sim_array = sim_array.apply(lambda col: [fuzz_measure(col.name,x) for x in col.index]).to_numpy()

    # Fill diagonal with 100, as each string is 100% similar to itself
    np.fill_diagonal(sim_array,100)

    return sim_array

def number_clusters(dataframe, sim_matrix):
    """
    This function computes clusters given a similarity matrix and dataframe containing cleaned and unique merchant names. It uses Affinity Propagation to create
    clusters from the given similarity matrix.

        Inputs: dataframe - a nx1 matrix containing cleaned ('Pass2' on merchant_name_standardization) merchant names
                sim_matrix - an nxn matrix filled with doubles from 0 to 100 (100 on diagonal) where each sim_array[i][j] represents the fuzz_similarity() measure
                between string i and j in dataframe

        Output: df_clusters - an nx2 matrix with columns ['Pass2','Clusters'] where 'Pass2' contains the input merchant names from dataframe and 'Clusters' Contains    
                the cluster the adjacent merchant name in column 'Pass2' was assigned to.
    """
    clust_ids = dataframe.to_list()
    # Preference affects how lax/strict cluster generation is. Higher preference makes AffinityPropagation assign more unique clusters; lower preference makes less clusters
    clusters = AffinityPropagation(affinity='precomputed',preference=70).fit_predict(sim_matrix)
    df_clusters = pd.DataFrame(list(zip(clust_ids, clusters)), columns = ['Pass2','Cluster'])

    return df_clusters

def generate_names(data):
    '''
    Generates names for each cluster produced by Merchant_name_cleaning_script

        Input: data - a pandas dataframe containing the columns ['Merchant','Expense Type','Expense Amount','Pass2','Pass1','Cluster']

        Output: grouped_merchants - a pandas dataframe containing the columns ['Cluster','generated_name']
    '''

    # Extract 'Pass1' column to generate cluster names on
    grouped_merchants = data[['Pass1', 'Cluster']]
    grouped_merchants = data.groupby('Cluster')
    grouped_merchants = grouped_merchants['Pass1'].apply(list)
    grouped_merchants = grouped_merchants.reset_index()

    grouped_merchants["generated_name"] = grouped_merchants['Pass1'].apply(lambda x: gen_name_helper(x))

    # Delete extra column, merge data back with original pandas dataframe containing rest of columns
    del grouped_merchants['Pass1']
    data = data.merge(grouped_merchants,on=['Cluster'])

    return data

def gen_name_helper(names):
    '''
    Extracts the most common merchant name from each cluster of merchant names.

        Input: names -  a list of merchant names

        Output: final_name - the most frequent merchant name in the list names.
    '''
    # Count occurences of each name
    counter = {}

    for name in names:
        if name in counter:
            counter[name] += 1
        else:
            counter[name] = 1

    # Pick unique name that occurs most frequently
    total_names = len(names)
    counter = {m: v/total_names for m,v in counter.items()} # Future revision: No need to divide by total names
    counter = list(counter.items())
    counter.sort(key = lambda y:y[1])
    final_name = counter[-1][0]

    return final_name

def expense_type_validator(merchant_data):
    """
    Validates clusters based on relative expense types. First, this function goes through each cluster to create an expense type frequency table in 
    each cluster. Then, for each unique name under 'Pass2' in each cluster, an expense type frequency table is also calculated. The expense types in
    common between the overall cluster and the unique name are then taken. If this group of shared expense types makes up less than 20% (subject to change)
    of the expense types of the overall cluster removed expense types from the current name, the current name is ejected from the cluster and assigned
    to a new cluster (as it is likely to not belong in the cluster).

    Mutates input pandas dataframe.

        Input: merchant_data - a pandas dataframe containing the columns ['Merchant','Cluster','Pass1','Pass2','Expense Amount'] (subject to change)
        Output: merchant_data - mutated input with validated clusters.
    """

    # Clean up expense data
    merchant_data = expense_type_cleaner(merchant_data)

    # Get unique cluster IDs; group merchant 'Pass2' names by cluster
    clust_unique_ids = merchant_data['Cluster'].unique()
    new_id = merchant_data['Cluster'].max() + 1
    clust_unique_names = merchant_data.groupby(by='Cluster')['Pass2'].unique().apply(list)

    for id in clust_unique_ids:
        
        cur_unique_names = clust_unique_names[id]

        # if there are 2 or less unique names in current cluster validation is skipped (if not skipped can cause inaccuracies)
        if len(cur_unique_names) <= 2:
            continue
        
        # Create initial frequency table for current cluster
        master_count = dict(merchant_data[merchant_data['Pass2'].isin(cur_unique_names)]['Expense Type'].value_counts())

        # if there are less than 3 expense type entries under current cluster validation is skipped
        if sum(master_count.values()) < 3:
            continue

        # creates frequency table and validates each unique name in current cluster
        for unique_name in cur_unique_names:

            # create frequency table for current unique name
            temp_un_count = dict(merchant_data.loc[merchant_data['Pass2'] == unique_name,'Expense Type'].value_counts())

            # creates copy of current cluster frequency count to modify
            master_count_removed_un = master_count.copy()
            
            # removes current unique name's expense types from overall cluster counts 
            for category in temp_un_count:
                master_count_removed_un[category] -= temp_un_count[category]

            # skips current name if current name has less than 2 expense type entries or if current unique name
            # makes up all of the current cluster
            if len(temp_un_count) == 0 or sum(temp_un_count.values()) <= 1 or sum(master_count_removed_un.values())<= .05:
                continue   

            # transforms raw expense type counts to frequency counts
            master_count_removed_un = {cat:count/sum(master_count_removed_un.values()) for cat,count in master_count_removed_un.items()}
            temp_un_count = {cat:count/sum(temp_un_count.values()) for cat,count in temp_un_count.items()}

            # initializes threshold and similarity counts for future comparison 
            un_threshhold = 0
            un_threshhold_categories = {}
            similarity = 0

            # get expense types that make up at least 90% of current name
            while un_threshhold <= .9:
                cur_max = max(temp_un_count, key = lambda y: temp_un_count[y])
                un_threshhold += temp_un_count[cur_max]
                un_threshhold_categories[cur_max] = temp_un_count[cur_max]
                del temp_un_count[cur_max]

            # ensure obtained expense types are in both the cluster removed current unique name and the current unique name
            un_threshhold_categories_keys = set(un_threshhold_categories.keys()).intersection(set(master_count_removed_un.keys()))

            for cat in un_threshhold_categories_keys:
                similarity += master_count_removed_un[cat]

            # if expense types that make up 90% of current unique name do not make up 20% (subject to change) of cluster name, 
            # current unique name is ejected

            if similarity <= .2:
                new_id += 1
                merchant_data.loc[merchant_data['Pass2'] == unique_name,'Cluster'] =  new_id
                # subtracts expense types in current unique name from expense types in cluster, as we are ejecting the current unique name from current cluster
                for category in temp_un_count:
                    master_count[category] -= temp_un_count[category] 

                print('new id added!', unique_name)

    return merchant_data

def expense_type_cleaner(dataframe):
    '''
    Consolidates existing expense categories to make expense type validation more accurate
        
        Input: dataframe - a pandas dataframe containing at least one column ['Expense Type']
        Output: dataframe - mutates and returns dataframe
        
    '''
    # consolidates multiple expense types (mapper values) to one cleaned expense type (mapper keys)
    mapper = {'Food':{'Dinner','Lunch','Breakfast','Meals (Client Attendees)','Meals (Multiple RES Employees)'},
            'Hotel':{'Hotel (Non-Project)','Hotel (Project Related)'},
            'Ignore': {'Project Costs','Undefined'}}

    for master_category in mapper.keys():
        for sub_category in mapper[master_category]:
            dataframe.loc[dataframe['Expense Type'] == sub_category, 'Expense Type'] = master_category

    dataframe.loc[dataframe['Expense Type'] == 'Ignore','Expense Type'] = np.nan

    return dataframe

def td_idf_regrouper(dataframe):
    """
    Does a quick regroup of ejected merchant names. Mutates input dataframe
        Input: dataframe - a pandas dataframe containing (at minimum) the columns ['generated_name']
        Output: None; mutates dataframe
    """

    # Match most similar generated names 
    matches = match_strings(dataframe['generated_name'].copy().drop_duplicates())
    matches = matches[matches['left_generated_name'] != matches['right_generated_name']]

    # Gets and groups list of matched names
    unique_names = set(matches['left_generated_name'].unique())

    grouped_matches = matches.groupby('left_generated_name')
    grouped_matches = grouped_matches['right_generated_name'].apply(list)

    # groups matched names to share one name. As names are grouped, they are removed from unique_names
    # to avoid extra work and key errors
    for name in unique_names.copy():

        if name not in unique_names:
            continue

        unique_names.remove(name)

        for matched_name in grouped_matches[name]:

            if matched_name not in unique_names:
                continue

            dataframe.loc[dataframe['generated_name'] == matched_name,'generated_name'] = name
            unique_names.remove(matched_name)

            print(name, "  |  ",matched_name)

def run_levenshtein(dirty_merchants):
    """
    Cleans and groups dirty merchant names into clusters of similarly named merchants. Generates a cluster id for each group.

        Input: dirty_merchants - a n x 1 (single column) pandas dataframe with column titles 'Merchant', consisting of unclean merchant names, and 'Expense Type', consisting of expense types

        Output: an n x 5 pandas dataframe consisting of the columns ['Merchant','Pass1','Pass2','Cluster','generated_name']
    """

    # clean and prep data - remove special characters, standardize whitespace, remove stopwords ('Pass2')
    clean_data = merchant_name_standardization(dirty_merchants)
    unique_names = pd.DataFrame()
    unique_names['Pass2'] = clean_data['Pass2']
    unique_names = unique_names.drop_duplicates()

    # initialize clusters to zero for merging later 
    unique_names['Cluster'] = 0

    # groups cleaned merchant names into groups separated by initial character. Creates similarity matrix & runs Affinity Propagation on each smaller cluster
    start_chars = set(unique_names['Pass2'].str[0]) # iterate for all unique initial characters. Loop ignores NaN values (caused by empty cells)
    iter = 0

    for s_char in start_chars:

        iter+=1; print(s_char,str(100*iter/len(start_chars))+'%') # progress bar

        if type(s_char) != type('c'):
            continue
        
        # group by initial character
        names_tocluster = pd.DataFrame()
        names_tocluster['Pass2'] = unique_names['Pass2'].loc[unique_names['Pass2'].str[0] == s_char]
        sim_array = fuzz_similarity(names_tocluster['Pass2'])

        # create new clusters on grouped names, ensure cluster IDs are unique
        numbered_clusters = number_clusters(names_tocluster['Pass2'],sim_array)
        cur_max = unique_names['Cluster'].max() + 1
        numbered_clusters['Cluster'] += cur_max
        unique_names.set_index(['Pass2'],inplace=True)
        unique_names.update(numbered_clusters.set_index(['Pass2']))
        unique_names.reset_index(inplace=True)

    # convert clusters to integers; merge based on 'Pass2' which is shared between unique_names and clean_data
    unique_names['Cluster'] = unique_names['Cluster'].astype(int)

    clean_data = clean_data.merge(unique_names,on='Pass2')
    #clean_data = clean_data.merge(merch_names,on='Cluster')

    return clean_data

def encode(input):
    '''
    Parses data from Tableau for use in clustering algorithm. Input is given by Tableau software as a dataframe

        Input: input - dataframe of dimensions specified by Tableau flow

        Output - given by get_output_schema()
    '''
    # Create initial clusters
    parsed_data = run_levenshtein(input)

    # Validate existing clusters
    parsed_data = expense_type_validator(parsed_data)

    # Generate cluster names
    parsed_data = generate_names(parsed_data)

    # Regroup ejected clusters (fast)
    td_idf_regrouper(parsed_data)

    return parsed_data

def get_output_schema():
    """
    Returns the format of the dataframe that this script returns to Tableau
    """
    return pd.DataFrame(
        {
        'Merchant' : prep_string(),
        'Pass1' : prep_string(),
        'Pass2' : prep_string(),
        'generated_name': prep_string(),
        #'Expense Type' : prep_string(),
        'Expense Amount': prep_decimal(),
        'UniqueID': prep_int()
        })
