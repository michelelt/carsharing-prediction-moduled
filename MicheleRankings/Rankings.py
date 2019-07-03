from functions import feature_importance_mrmr, feature_importance_RF
import pandas as pd
import os
import copy

def DataPreparation(Features, Label_name):
    
    df = pd.read_csv("dataset.csv")
       
    # remove eventually columns with only nan  values
    df = df.dropna(axis=1)
    # remove eventually row with at least one nan value
    df = df.dropna()

        
    if(Features!=None):
        Subset = copy.deepcopy(Features)
        Subset.append(Label_name)
        df = df[Subset]
    
    return df


def main():
    
 
    #Folder Where Rankings will be saved
    #!The script DOES NOT create the folder
    output_folder = "."
    
    #Decide if the task is a classification task or regression task for the ranking
    task = "classification" 
    #task = "regression" 
    
    #Vector with a Subset of the Dataset Features. 
    #The Vector specifies which Features will be used. All other features will be discarded while loading the dataset
    #Used to Run Ranking Aglos only on a subset of all possible features. 
    #Used if some features are removed using the Domain Knowledge but these Features are still in the dataset
    Features = None

    #The Name of the Label to Use for the ranking
    #For each Run only a single label! If the dataset HAS 2 (or more) LABELS REMOVE ALL LABELS EXCEPT THIS ONE BEFORE THE RANKING
    Label_name = "Label"

    #Load the dataset
    df = DataPreparation(Features,Label_name)  
    
    #Compute RF Ranking
    feature_importance_RF(df, Label_name,output_folder,task)

    #Compute MRMR Ranking
    feature_importance_mrmr(df, Label_name,output_folder,task)
    
       
    return
main()
