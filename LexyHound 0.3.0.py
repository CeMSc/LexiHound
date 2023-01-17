import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent_i = SentimentIntensityAnalyzer()


## IMPORT FILES
def txt_to_dataframe():
    file_path = []
    for file in os.listdir("./TXT"):
        file_path.append(os.path.join("./TXT", file))
    file_name = re.compile('\\\\(.*)\.txt')
    data = {}
    for file in file_path:
        key = file_name.search(file)
        with open(file, "r", encoding='Latin-1') as read_file:
            if key is not None:
                data[key[1]] = [read_file.read()]
    df = pd.DataFrame(data).T.reset_index().rename(columns = {'index':'document', 0:'text'})
    codebook = df[['document']].copy()
    codebook_sentiment = codebook
    df.head(3)
    return df

## IMPORT search_strings, co_occurrences, doc_conditionals, keywords
def create_dataframes(xlsx_file):
    xlsx = pd.ExcelFile(xlsx_file)

    # Load each sheet into a separate dataframe
    set_search_strings = pd.read_excel(xlsx, sheet_name='set_search_strings')
    set_co_occurrences = pd.read_excel(xlsx, sheet_name='set_co_occurrences')
    set_doc_conditionals = pd.read_excel(xlsx, sheet_name='set_doc_conditionals')
    set_keywords = pd.read_excel(xlsx, sheet_name='set_keywords')

    # Return the dataframes as a tuple
    return set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords 

## ORGANIZE THE KEWYORDS DATAFRAME INTO A DICTIONARY
def organize_keywords(df):
    cols = df.columns
    key_dict = {}
    for col in cols:
        values = df[col].dropna().tolist()
        key_dict[col] = values
    return key_dict

## CLEAN TEXT and SPLIT SENTENCES
def clean_text(df): # Lowercase, clean and strip text.
    df["text"] = df.text.str.lower()
    df["text"] = df.text.str.replace("\ufeff", "")
    df["text"] = df.text.str.strip()
       
def split_sentences(df): # Splits text in a list of sentences and remove whole text fr
    df["sentences"] = df["text"].apply(nltk.sent_tokenize)
    return df.explode("sentences")

## CHECK OCCURRENCE OF GROUPS OF KEYWORDS
def check_groups(row, word_dict, *args): # Check if any of the words in a dictionary of lists are present in a given string as an exact match.
    for key, words in word_dict.items():
        for word in words:
            # Replace * character with '\w*'
            pattern = r"\b%s\b" % word.replace("*", "\w*")
            # Find all the occurrences of the pattern in the string
            matches = re.findall(pattern, row['sentences'])
            if len(matches) > 0:
                row[key] = True
                break
        else:
            row[key] = False
    return row

## CHECK OCCURRENCE OF INDIVIDUAL KEYWORDS (optional function)
def check_words(row, word_dict):
    # Iterate over the keys and values in the dictionary
    for key, words in word_dict.items():
        for word in words:
            # Replace * character with '\w*'
            pattern = r"\b%s\b" % word.replace("*", "\w*")
            matches = re.findall(pattern, row['sentences'])
            if len(matches) > 0:
                row[word] = True
            else:
                row[word] = False
    return row

## FIND co-OCCURRENCES
def find_co_occurrences(row, key1, key2, word_dict, distance, name):
    # Get the lists of words for the two keys
    words1 = word_dict[key1]
    words2 = word_dict[key2]
    
    # Convert the dictionary values to strings
    words1 = [str(word) for word in words1]
    words2 = [str(word) for word in words2]
    
    # Create regular expression patterns for the two lists of words
    patterns1 = [r"\b%s\b" % word.replace("*", "\w*") for word in words1]
    patterns2 = [r"\b%s\b" % word.replace("*", "\w*") for word in words2]

    # Find the occurrences of the patterns in the text column
    occurrences1 = []
    occurrences2 = []
    for pattern in patterns1:
        occurrences1 += [m.start() for m in re.finditer(pattern, row['sentences'])]
    for pattern in patterns2:
        occurrences2 += [m.start() for m in re.finditer(pattern, row['sentences'])]

    # Calculate the co-occurrence distances
    co_occurrences = []
    for occ1 in occurrences1:
        for occ2 in occurrences2:
            start = min(occ1, occ2)
            end = max(occ1, occ2)
            num_words = len(row['sentences'][start:end].split())
            co_occurrences.append(num_words)

    # If there are any co-occurrences, add a new column with the minimum distance
    if co_occurrences:
        if min(co_occurrences) <= distance:
            row[name] = True #min(co_occurrences)
        else:
            row[name] = False
    else:
        # If there are no co-occurrences, set the value to NaN
        row[name] = np.nan

    return row

def initiate_co_occurrences(df, set_co_occurrences, word_dict):
    # Convert the 'name' column to the 'object' data type
    set_co_occurrences['name'] = set_co_occurrences['name'].astype(object)
    
    for index, row in set_co_occurrences.iterrows():
        # Replace NaN values in the 'name' column with an empty string
        if pd.isnull(row['name']):
            row['name'] = ''
        
        key1 = row['group1']
        key2 = row['group2']
        distance = row['distance']
        name = row['name']
        
        df = df.apply(find_co_occurrences, key1=key1, key2=key2, distance=distance, name=name, word_dict=word_dict, axis=1)
        
    return df


## Mark True if at least one sentence in a document meets conditional   
def find_document_conditionals(df, name, conditional): 
    df[name] = df['document'].map(df.groupby('document').apply(lambda x: x[conditional].eq(1).any()))
    
def initiate_document_conditionals(df, set_doc_conditionals):
    set_doc_conditionals.apply(lambda row: find_document_conditionals(df, row['name'], row['group']), axis=1)


def vadar_sentiment_analysis(text):
        return sent_i.polarity_scores(text)['compound']

def group_to_variable(df, parent_variable_number, variable_number, group_1):
   for row in df:
        name = variable_number
        condition = group_1
        df[name] = (df[condition] == True)

def bool_to_variable(df, parent_variable_number, variable_number, *args):
    name = variable_number
    con1, operator_1, con2 = args[:3]
    if len(args) == 3:
        if operator_1 == 'and':
            df[name] = (df[con1] == True) & (df[con2] == True)
        elif operator_1 == 'or':
            df[name] = (df[con1] == True) | (df[con2] == True)
    else:
        operator_2, con3 = args[3:]
        if operator_1 == 'and' and operator_2 == 'and':
            df[name] = (df[con1] == True) & (df[con2] == True) & (df[con3] == True)
        elif operator_1 == 'and' and operator_2 == 'or':
            df[name] = (df[con1] == True) & ((df[con2] == True) | (df[con3] == True))
        elif operator_1 == 'or' and operator_2 == 'and':
            df[name] = (df[con1] == True) | ((df[con2] == True) & (df[con3] == True))
        elif operator_1 == 'or' and operator_2 == 'or':
            df[name] = (df[con1] == True) | ((df[con2] == True) | (df[con3] == True))

def aggregate_variables(df, previous_pvn, variable_numbers):
    df[previous_pvn] = False
    for index, row in df.iterrows():
        flag = False
        for vn in variable_numbers:
            # Check if the column with the name of vn is True
            if row[vn] == True:
                flag = True
                break
        df.at[index, previous_pvn] = flag

def aggregate_variables(prev_pvn, df):
    df[prev_pvn] = False
    var_nums = set_search_strings[set_search_strings['parent_variable_number'] == prev_pvn]['variable_number'].tolist()
    df[prev_pvn] = df[var_nums].any(axis=1)

def initiate_search_strings(search_strings, df):
    prev_pvn = None
    for index, row in search_strings.iterrows():
        if row['parent_variable_number'] != prev_pvn:
            if prev_pvn is not None:
                aggregate_variables(prev_pvn, df)
            prev_pvn = row['parent_variable_number']
        if row['search_type'] == 'g2v':
            group_to_variable(df, row['parent_variable_number'], row['variable_number'], row['group_1'])
        elif row['search_type'] == 'b2v':
            if pd.isnull(row['operator_2']):
                bool_to_variable(df, row['parent_variable_number'], row['variable_number'], row['group_1'], row['operator_1'], row['group_2'])
            else:
                bool_to_variable(df, row['parent_variable_number'], row['variable_number'], row['group_1'], row['operator_1'], row['group_2'], row['operator_2'], row['group_3'])

def create_codebook(df):
    v_list = []
    for col in df.columns:
        if bool(re.match('^[0-9\.]+$', str(col))):
            v_list.append(col)
            v_list.sort()

    # codebook of results
    codebook = df.groupby(['document'])[v_list].sum().astype(int).reset_index()

    # codebook of results but clipped to 1
    code_bool = codebook.copy()
    code_bool[v_list] = code_bool[v_list].clip(upper=1)

    # codebook of results as percentage of total no. of sentences per document
    codebook_count_sent = df.groupby(['document'])['sentences'].count().astype(int).reset_index()
    code_sent_percent = codebook.merge(codebook_count_sent, on='document', how='outer')
    code_sent_percent[v_list]=code_sent_percent[v_list].div(code_sent_percent['sentences'], axis=0)
    code_sent_percent[v_list]=code_sent_percent[v_list].multiply(100)
    code_sent_percent.drop('sentences', axis=1, inplace=True)

    # codebook of sentiment
    codebook_sentiment = df[['document']]
    for var in v_list:
        sentiment = df[df[var]].groupby('document', as_index=False).vadar_compound.mean()
        codebook_sentiment[var] = codebook_sentiment['document'].map(sentiment.set_index('document')['vadar_compound'])
    codebook_sentiment = codebook_sentiment.groupby('document').mean()
    save_codebook(codebook, code_sent_percent, code_bool, codebook_sentiment)


def save_codebook(codebook, code_percent, code_bool, codebook_sentiment):
    dfs = {'codebook':codebook, 'code_percent':code_percent, 'codebook_bool':code_bool, 'codebook_sentiment': codebook_sentiment}
    writer = pd.ExcelWriter('Codebook.xlsx', engine='xlsxwriter')
    for sheet_name in dfs.keys():
        dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()


df = txt_to_dataframe()
set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords = create_dataframes('Assessment_framework.xlsx')
clean_text(df)
df = split_sentences(df)
key_dict = organize_keywords(set_keywords)
df = df.apply(check_groups, axis=1, args=(key_dict,))
#df = df.apply(check_words, word_dict=key_dict, axis=1)
df = initiate_co_occurrences(df, set_co_occurrences, key_dict)
initiate_document_conditionals(df, set_doc_conditionals)
df['vadar_compound'] = df['sentences'].apply(vadar_sentiment_analysis)
initiate_search_strings(set_search_strings, df)
create_codebook(df)