"""Run Hypothesis Test"""
import json
import pandas as pd
import numpy as np
import utils
from scipy.stats import ttest_rel
import config
import os
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
import json
import sys

"""Compare class"""
class Compare(object):
    def __init__(self, result, compare_method, compare_metric):
        super(Compare, self).__init__()
        """ Compare
        Args:
            result (dict): result dict
            compare_method (fn): function to compare two metrics
            compare_metric (fn): function to specify metrics to be compared
        """
        self.result = result
        self.compare_metric = compare_metric
        self.compare_method = compare_method

        self.four_metrics = {}
        self.compare_result = {}
        for error_type in ["missing_values", "outliers", "mislabel", "inconsistency", "duplicates"]:
            self.compare_result[error_type], self.four_metrics[error_type] = self.compare_error(error_type)
        
        # key order: error/clean_method/dataset/models/scenario/ [compare_keys...]
        self.compare_result = utils.flatten_dict(self.compare_result)

        # rearrange key order: error/dataset/clean_method/models/scenario/ [compare_keys...]
        self.compare_result = utils.rearrange_dict(self.compare_result, [0, 2, 1, 3, 4])

    def get_four_metrics(self, error_type, file_types):
        """Get four metrics (A, B, C, D) for all datasets in a table (pd.DataFrame)

        Args:
            error_type (string): error type
            file_types (list): names of two types of train or test files
        """
        four_metrics = {}
        for (dataset, split_seed, error, train_file, model), value in self.result.items():
            if error == error_type and train_file in file_types:
                for test_file in file_types:
                    metric_name = self.compare_metric(dataset, error_type, test_file)
                    metric = value[metric_name]
                    four_metrics[(dataset, split_seed, train_file, model, test_file)] = metric

        four_metrics = utils.dict_to_df(four_metrics, [0, 2, 1], [3, 4]).sort_index()
        return four_metrics

    def compare_four_metrics(self, four_metrics, file_types):
        """Compute the relative difference between four metrics

        Args:
            four_metrics (pandas.DataFrame): four metrics
            file_types (list): names of two types of train or test files
            compare_method (fn): function to compare two metrics
        """
        A = lambda m: m.loc[file_types[0], file_types[0]]
        B = lambda m: m.loc[file_types[0], file_types[1]]
        C = lambda m: m.loc[file_types[1], file_types[0]]
        D = lambda m: m.loc[file_types[1], file_types[1]]

        CD = lambda m: self.compare_method(C(m), D(m))
        BD = lambda m: self.compare_method(B(m), D(m))
        # AB = lambda m: self.compare_method(A(m), B(m))
        # AC = lambda m: self.compare_method(A(m), C(m))

        comparison = {}
        datasets = list(set(four_metrics.index.get_level_values(0)))
        models = list(set(four_metrics.columns.get_level_values(0)))
        for dataset in datasets:
            for model in models:
                m = four_metrics.loc[dataset, model]
                comparison[(dataset, model, "CD")] = CD(m)
                comparison[(dataset, model, "BD")] = BD(m)
                # comparison[(dataset, model, "AC")] = AC(m)
                # comparison[(dataset, model, "AB")] = AB(m)
                # comparison[(dataset, model, "AD")] = AD(m)
        # comparison = utils.dict_to_df(comparison, [0, 1], [2])
        return comparison

    def compare_error(self, error_type):
        """Compare four metrics based on compared method given error_type

        Args:
            error_type (string): error type

        Return: 
            clean_method/dataset/model/scenario/compare_method:result

        """
        ## each error has two types of files
        # file type 1
        file1 = "delete" if error_type == "missing_values" else "dirty"
        file2 = list(set([k[3] for k in self.result.keys() if k[2] == error_type and k[3] != file1]))
        comparisons = {}
        metrics = {}

        for f2 in file2:
            file_types = [file1, f2]
            four_metrics = self.get_four_metrics(error_type, file_types)
            comparison = self.compare_four_metrics(four_metrics, file_types)
            metrics[f2] = four_metrics
            comparisons[f2] = comparison
        return comparisons, metrics

    def save_four_metrics(self, save_dir):
        for error_type in ["missing_values", "outliers", "mislabel", "inconsistency", "duplicates"]:
            save_path = os.path.join(save_dir, "{}_four_metrics.xlsx".format(error_type))
            utils.dfs_to_xls(self.four_metrics[error_type], save_path)
        flat_metrics = utils.flatten_dict(self.four_metrics)

"""Compare method"""
def t_test(dirty, clean):
    def two_tailed_t_test(dirty, clean):
        n_d = len(dirty)
        n_c = len(clean)
        n = min(n_d, n_c)
        t, p = ttest_rel(clean[:n], dirty[:n])
        if np.isnan(t):
            t, p = 0, 1
        return {"t-stats":t, "p-value":p}

    def one_tailed_t_test(dirty, clean, direction):
        two_tail = two_tailed_t_test(dirty, clean)
        t, p_two = two_tail['t-stats'], two_tail['p-value']
        if direction == 'positive':
            if t > 0 :
                p = p_two * 0.5
            else:
                p = 1 - p_two * 0.5
        else:
            if t < 0:
                p = p_two * 0.5
            else:
                p = 1 - p_two * 0.5
        return {"t-stats":t, "p-value":p}

    result = {}
    result['two_tail'] = two_tailed_t_test(dirty, clean)
    result['one_tail_pos'] = one_tailed_t_test(dirty, clean, 'positive')
    result['one_tail_neg'] = one_tailed_t_test(dirty, clean, 'negative')
    return result

def mean_f1(dirty, clean):
    result = {"dirty_f1": np.mean(dirty), "clean_f1":np.mean(clean)}
    return result

def mean_acc(dirty, clean):
    result = {"dirty_acc": np.mean(dirty), "clean_acc":np.mean(clean)}
    return result

def direct_count(dirty, clean):
    result = {"pos_count": np.sum(dirty < clean), "neg_count": np.sum(dirty > clean), "same_count": np.sum(dirty == clean)}
    return result

"""Compare metric"""
def test_f1(dataset_name, error_type, test_file):
    metric = test_file + "_test_f1"
    return metric

def test_acc(dataset_name, error_type, test_file):
    metric = test_file + "_test_acc"
    return metric

def mixed_f1_acc(dataset_name, error_type, test_file):
    if error_type == 'mislabel':
        dataset_name = dataset_name.split('_')[0]
    dataset = utils.get_dataset(dataset_name)
    if ('class_imbalance' in dataset.keys() and dataset['class_imbalance']):
        metric = test_file + "_test_f1"
    else:
        metric = test_file + "_test_acc"
    return metric

"""Multiple hypothesis test """
def BY_procedure(t_test_results, test_type, alpha=0.05):
    p_vals = t_test_results.loc[:, (test_type, 'p-value')]
    reject, correct_p_vals, _, _ = multipletests(p_vals.values, method='fdr_by', alpha=alpha)
    reject_df = pd.DataFrame(reject, index=p_vals.index, columns=['reject'])
    correct_p_df = pd.DataFrame(correct_p_vals, index=p_vals.index, columns=['p-value'])
    return reject_df, correct_p_df

def hypothesis_test(t_test_results, alpha=0.05):
    # convert to pd.DataFrame
    t_test_results_df = utils.dict_to_df(t_test_results, [0, 1, 2, 3, 4], [5, 6])

    # run BY procedure
    test_types = ['two_tail', 'one_tail_pos','one_tail_neg']
    rejects = {}
    correct_p= {}
    for test_type in test_types:
        r, p = BY_procedure(t_test_results_df, test_type, alpha=alpha)
        if test_type != 'two_tail':
            rejects[test_type] = r
            correct_p[test_type] = p
        # save_path = os.path.join(config.table_dir, 't_test', '{}_reject.pkl'.format(test_type))
        # utils.df_to_pickle(r, save_path)

    hypothesis_result = {}
    for e, d, c, m, s, _, _ in t_test_results.keys():
        hypothesis_result[(e, d, c, m, s, 'pos_pvalue')] = correct_p['one_tail_pos'].loc[(e, d, c, m, s),'p-value']
        hypothesis_result[(e, d, c, m, s, 'neg_pvalue')] = correct_p['one_tail_neg'].loc[(e, d, c, m, s),'p-value']
        pos = rejects['one_tail_pos'].loc[(e, d, c, m, s), 'reject']
        neg = rejects['one_tail_neg'].loc[(e, d, c, m, s), 'reject']
        if pos:
            hypothesis_result[(e, d, c, m, s, 'flag')] = 'P'
        elif neg:
            hypothesis_result[(e, d, c, m, s, 'flag')] = 'N'
        else:
            hypothesis_result[(e, d, c, m, s, 'flag')] = 'S'

    # save_path = os.path.join(config.table_dir, 't_test', 'hypothesis_test.xlsx')
    # save_hypothesis_test(t_test_results, rejects, save_path)
    return hypothesis_result

def split_clean_method(result):
    new_result = {}
    for (error, dataset, clean_method, model, scenario, comp_key), value in result.items():
        if error == 'outliers':
            detect = clean_method.split('_')[1]
            repair = clean_method.replace('_{}'.format(detect), '')
        else:
            detect = 'detect'
            repair = clean_method
        new_result[(error, dataset, detect, repair, model, scenario, comp_key)] = value
    return new_result

def group_by_mean(result):
    # group by training seed and reduce by mean
    result = utils.group(result, 5)
    result = utils.reduce_by_mean(result)
    return result

def group_by_best_model(result):
    # select best model by max val acc
    result = utils.group(result, 5)
    result = utils.reduce_by_max_val(result)
    result = utils.group(result, 4, keepdim=True)
    result = utils.reduce_by_max_val(result, dim=4, dim_name="model")
    return result

def group_by_best_model_clean(result):
    # select best model by max val acc
    result = utils.group_reduce_by_best_clean(result)
    return result    

def analyze(result, save_dir, name, alpha=0.05, split_detect=True):
    # compare by t-test and do multiple hypothesis test
    t_test_comp = Compare(result, t_test, mixed_f1_acc)
    metric_dir = utils.makedirs([save_dir, 'four_metrics'])
    t_test_comp.save_four_metrics(metric_dir)
    hypothesis_result = hypothesis_test(t_test_comp.compare_result, alpha)

    # show mean of acc
    mean_acc_comp = Compare(result, mean_acc, test_acc)

    # show mean of f1
    mean_f1_comp = Compare(result, mean_f1, test_f1)

    # count directly
    direct_count_comp = Compare(result, direct_count, mixed_f1_acc)
    
    # combine results
    analysis = {**mean_acc_comp.compare_result, **mean_f1_comp.compare_result, **hypothesis_result, **direct_count_comp.compare_result}
    if split_detect:
        analysis = split_clean_method(analysis)
        analysis_df = utils.dict_to_df(analysis, [0, 1, 2, 3, 4, 5], [6])
    else:
        analysis_df = utils.dict_to_df(analysis, [0, 1, 2, 3, 4], [5])

    save_path = os.path.join(save_dir, '{}_analysis.csv'.format(name))
    # analysis_df.to_csv(save_path, index_label=['error_type', 'dataset', 'clean_method', 'model', 'scenario'])
    analysis_df.to_csv(save_path)

if __name__ == '__main__':
    # save training result 
    result = utils.load_result(parse_key=True)
    # save_dir = os.path.join(config.analysis_dir, "training_result")
    # utils.result_to_table(result, save_dir)

    #anaylze by mean performance
    result_mean = group_by_mean(result)
    save_dir = os.path.join(config.analysis_dir, "mean_analysis")
    analyze(result_mean, save_dir, "mean")

    # anaylze by best model, selected by max val acc
    # result_best_model = group_by_best_model(result)
    # save_dir = os.path.join(config.analysis_dir, "best_model_analysis")
    # analyze(result_best_model, save_dir, "best_model") 

    # analyze by best model and best clean method
    # result_best_model_clean = group_by_best_model_clean(result_best_model)
    # save_dir = os.path.join(config.analysis_dir, "best_model_clean_analysis")
    # analyze(result_best_model_clean, save_dir, "best_model_clean", split_detect=False) 
