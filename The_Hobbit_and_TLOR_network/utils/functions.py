import pandas as pd
import numpy as np
import os
import spacy
from spacy import displacy
import networkx as nx

import matplotlib.pyplot as plt



def ner_book(os_name):
    """
    Function to process text from a text file (.txt) using Spacy.
    
    Params:
    nlp -- spacy English languague model
    
    os_name -- name of a txt file as string
    
    Returns:
    a processed doc file using Spacy English language model
    
    """
    
    with open(os_name,  'r', encoding='utf-8') as t:
        book_doc=nlp(t.read())
        t.close()
    return book_doc



def entity_tag(spacy_doc):
    """
    Get a list of entites per sentence of a Spacy document and store in a dataframe.
    
    Params:
    spacy_doc -- a Spacy processed document
    
    Returns:
    a dataframe containing the sentences and corresponding list of recognised named entities       in the sentences
    """
    
    sent_entity_df = []

    # Loop through sentences, store named entity list for each sentence
    for sent in spacy_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({"sentence": sent, "entities": entity_list})

    sent_entity_df = pd.DataFrame(sent_entity_df)
    
    return sent_entity_df


def filter_entity(ent_list, characters_df):
    """
    Function to filter out non-character entities.
    (Adapted to include the nickname of Samwise and 
    Isildur)
    
    
    Params:
    ent_list -- list of entities to be filtered
    characters_df -- a dataframe contain characters' names and characters' first names
    
    Returns:
    a list of entities that are characters (matching by names or first names).
    
    """
    return [ent for ent in ent_list 
            if ent in list(characters_df.Name) 
            or ent in list(characters_df.Firstname)
            or ent in ['Sam','Isildur']]

def window_relationships(df, window_size):
    
    """
    Create a dataframe of relationships based on the df dataframe (containing lists of chracters per sentence) and the window size of n sentences.
    
    Params:
    df -- a dataframe containing a column called character_entities with the list of characters for each sentence of a document.
    window_size -- size of the windows (number of sentences) for creating relationships between two adjacent characters in the text.
    
    Returns:
    a relationship dataframe containing 3 columns: source, target, value.
    
    """
    
    relationships = []

    for i in range(df.index[-1]):
        end_i = min(i+window_size, df.index[-1])
        char_list = sum((df.loc[i: end_i].character_entities), [])

        # Remove duplicated characters that are next to each other
        char_unique = [char_list[i] for i in range(len(char_list)) 
                       if (i==0) or char_list[i] != char_list[i-1]]

        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({"source": a, "target": b})
           
    relationship_df = pd.DataFrame(relationships)
    
    # Fixing the Sam, Samwise problem I
    for i in relationship_df[(relationship_df['source']=='Samwise')].index.values:
        relationship_df.iloc[i,0] = 'Sam'


    for i in relationship_df[(relationship_df['target']=='Samwise')].index.values:
        relationship_df.iloc[i,1] = 'Sam'

    relationship_df.drop(relationship_df[(relationship_df['target']==relationship_df['source']) ].index.values, inplace=True)
    
    # Sort the cases with a->b and b->a
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), 
                                   columns = relationship_df.columns)
    relationship_df["value"] = 1
    relationship_df = relationship_df.groupby(["source","target"], 
                                              sort=False, 
                                              as_index=False).sum()
                
    return relationship_df

