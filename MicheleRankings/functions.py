import os
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend. useful when save plot on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import imp
mrmr = imp.load_source('mrmr', os.path.abspath('./mrmr.py'))


''' 
MRMR Functions
'''
def feature_importance_mrmr(df, ranking_name,output_folder,task):
    
    if(task!="classification" and task!="regression"):
        print("error: task has to be: classification or regression")
        exit(0)
    
    labels = np.array(df[ranking_name])
    df = df.drop([ranking_name], axis=1)
    features_list = list(df.columns)
    features = np.array(df[features_list])
    
    estimator = mutual_info_classif
    if(task=="regression"): estimator = mutual_info_regression

    
    selected_feature_indices, relevance_per_feature, redundancy_per_feature, mid_per_feature = mrmr.mrmr(
                                                                                            features, 
                                                                                            labels,
                                                                                            estimator=estimator,
                                                                                            num_features_to_select=len(features_list),
                                                                                            k_max_features=len(features_list),
                                                                                            n_jobs=20,
                                                                                            discrete=False,
                                                                                            randomState=42,
                                                                                            feature_list=features_list)
                                                                                    
    with open(output_folder+"/"+ranking_name+'_mrmr_'+task+'.csv', 'wb') as f:
        f.write(',Feature_ranking,relevance, redundancy, MID\n')
        j = 0
        for i in selected_feature_indices:
            f.write(str(i)+ ', ' + str(features_list[i]) + ',' + str(relevance_per_feature[j]) + ','
                    + str(redundancy_per_feature[j]) + ',' + str(mid_per_feature[j]) + '\n')
            j += 1
    plt.plot(redundancy_per_feature, color='red', linewidth=0.7)
    plt.ylabel('Redundancy of last feature selected')
    plt.xlabel('Selected features')
    plt.grid(True)
    plt.savefig(output_folder+"/"+ranking_name+'_Redundancy_'+task+'.png', dpi=300)
    plt.close()
    plt.plot(mid_per_feature, color='red', linewidth=0.7)
    plt.ylabel('MID of last feature selected')
    plt.xlabel('Selected features')
    plt.grid(True)
    plt.savefig(output_folder+"/"+ranking_name+'_MID_'+task+'.png', dpi=300)
    plt.close()
    plt.plot(relevance_per_feature, color='red', linewidth=0.7)
    plt.ylabel('Relevance of last feature selected')
    plt.xlabel('Selected feature')
    plt.grid(True)
    plt.savefig(output_folder+"/"+ranking_name+'_Relevance_'+task+'.png', dpi=300)
    plt.close()


''' 
RF Function
'''
def feature_importance_RF(df, ranking_name,output_folder,task):
       
    if(task!="classification" and task!="regression"):
        print("error: task has to be: classification or regression")
        exit(0)

    labels = np.array(df[ranking_name])
    df = df.drop([ranking_name], axis=1)
    features_list = list(df.columns)
    features = np.array(df[features_list])


    # create the model instance for feature importance
    rf_feature_importance = RandomForestClassifier(random_state=42, n_estimators=500, n_jobs=10, 
                                      max_features=len(features_list))

    if(task=="regression"): 
        rf_feature_importance = RandomForestRegressor(random_state=42, n_estimators=500, n_jobs=10, 
                                  max_features=len(features_list))

    
    rf_feature_importance.fit(features, labels)

    feature_importances_df = pd.DataFrame({'Feature_ranking': features_list,
                                           'importance': np.around(rf_feature_importance.feature_importances_, decimals=5)})
    feature_importances_sorted_df = feature_importances_df.sort_values('importance', ascending=False)

    with open(output_folder+"/"+ranking_name+'_rf_'+task+'.csv', 'wb') as f:
        feature_importances_sorted_df.to_csv(f)

    y = [v for v in feature_importances_sorted_df['importance']]

    plt.plot([i for i in range(len(y))],y, marker="+", linewidth=0.7)
    plt.ylim(0,0.1)
    plt.ylabel('Feature importance')
    plt.xlabel('Selected feature')
    plt.grid(True)
    plt.savefig(output_folder+"/"+ranking_name+'_rf_'+task+'.png', dpi=300)
    plt.close()

    return
