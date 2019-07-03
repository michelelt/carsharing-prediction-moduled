import datetime
import os
import shutil
from tempfile import mkdtemp

from sklearn.externals.joblib import Parallel, delayed
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def generic_combined_scorer(feature_redundance_vec, k_values, i_values, i, randomstate):
    Z = np.stack((i_values, k_values), axis=-1)
    s2 = mutual_info_regression(Z, k_values, discrete_features=False, random_state=randomstate)
    feature_redundance_vec[i] = s2[0]


def relevance_scorer(estimator, feature_relevance_vec, k_values, i_values, i, randomstate):
    Z = np.stack((i_values, k_values), axis=-1)
    s2 = estimator(Z, k_values, discrete_features=False, random_state=randomstate)
    feature_relevance_vec[i] = s2[0]


def mrmr(X, y, estimator, feature_list, num_features_to_select=None, k_max_features=1000, discrete=False,
         randomState=None, n_jobs=1):
    num_dim = X.shape[1]
    if num_features_to_select is not None:
        num_selected_features = min(num_dim, num_features_to_select)
    else:
        num_selected_features = num_dim

    initial_scores = np.zeros(len(feature_list))
    for i in range(len(feature_list)):
        relevance_scorer(estimator, initial_scores, y, X[:, i], i, randomState)
    print 'relevance computed'
    # rank the scores in descending order
    sorted_scores_index = np.flipud(np.argsort(initial_scores))

    k_max_features = min(k_max_features, len(initial_scores))
    num_selected_features = min(num_selected_features, k_max_features)

    X_subset = X[:, sorted_scores_index[0:k_max_features]]

    selected_features_index = np.zeros(num_selected_features, dtype=int)
    remaining_candidate_index = range(1, k_max_features)

    # memory map this for parallelization speed
    tmp_folder = mkdtemp()

    relevance_vec_fname = os.path.join(tmp_folder, 'relevance_vec')
    feature_redundance_vec_fname = os.path.join(tmp_folder, 'feature_redundance_vec')
    mi_matrix_fname = os.path.join(tmp_folder, 'mi_matrix')
    relevance_vec = np.memmap(relevance_vec_fname, dtype=float,
                              shape=(k_max_features,), mode='w+')
    feature_redundance_vec = np.memmap(feature_redundance_vec_fname, dtype=float,
                                       shape=(k_max_features,), mode='w+')
    mi_matrix = np.memmap(mi_matrix_fname, dtype=float,
                          shape=(k_max_features, num_selected_features - 1), mode='w+')

    print feature_list[sorted_scores_index[0]]
    relevance_per_feature_selected = [initial_scores[sorted_scores_index[0]]]
    redundancy_per_feature_selected = [0]
    mid_per_feature_selected = [initial_scores[sorted_scores_index[0]]]
    mi_matrix[:] = np.nan
    print 'Start redundancy phase', datetime.datetime.now().strftime("%H:%M:%S")
    for k in range(1, num_selected_features):
        last_selected_feature = k - 1

        # compute the redundancy with the last feature selected
        Parallel(n_jobs=n_jobs) \
            (delayed(generic_combined_scorer)(feature_redundance_vec,
                                              X_subset[:, selected_features_index[last_selected_feature]],
                                              X_subset[:, i], i, randomState)
             for i in remaining_candidate_index)

        # copy the redundance into the mi_matrix, which accumulates our redundance as we compute
        mi_matrix[remaining_candidate_index, last_selected_feature] = feature_redundance_vec[remaining_candidate_index]
        # compute the mean of redundancy of each feature with the feature already selected
        redundance_vec = np.nanmean(mi_matrix[remaining_candidate_index, :], axis=1)

        # get the index of the feature that maximize the difference between relevance and redundancy according to MID
        # criterion (Mutual Information Difference)
        mid = initial_scores[sorted_scores_index[remaining_candidate_index]] - redundance_vec
        tmp_idx = np.argmax(mid)
        # mid of last feature selected
        mid_per_feature_selected.append(mid[tmp_idx])
        # select the feature that maximize the difference
        ind = remaining_candidate_index[tmp_idx]
        selected_features_index[k] = ind
        relevance_per_feature_selected.append(initial_scores[sorted_scores_index[ind]])
        # extract redundancy for the last selected feature
        redundancy_per_feature_selected.append(redundance_vec[tmp_idx])
        # remove the feature from the list of remaining feature
        del remaining_candidate_index[tmp_idx]

    # map the selected features back to the original dimensions
    selected_features_index = sorted_scores_index[selected_features_index]
    print 'End redundancy phase', datetime.datetime.now().strftime("%H:%M:%S")

    # clean up
    try:
        shutil.rmtree(tmp_folder)
    except:
        pass

    # return the index of selected feature
    return selected_features_index, relevance_per_feature_selected, redundancy_per_feature_selected, mid_per_feature_selected
