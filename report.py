import pickle
import os
import numpy as np
import pandas as pd
from utils import *
from tabulate import tabulate

def toExcel(args, base_path):
    with open(base_path+'/seed'+str(args.seed)+'history.pkl', 'rb') as file:
        history = pickle.load(file)
    df = pd.DataFrame.from_dict(history, orient='columns')
    df = df.drop(['args'], axis=1)
    df = df.drop(['group_acc_list'], axis=1)
    columns = df.columns.tolist()
    for column in columns:
        df[column] = df[column].apply(lambda x: round(float(x), 4))
    df.to_excel(base_path+'/seed'+str(args.seed)+'AllHistory.xlsx', index=False)

def evaluate(args, base_path, groups_acc_list):
    with open(base_path+'/seed'+str(args.seed)+'history.pkl', 'rb') as file:
        history = pickle.load(file)

    best_fair = 100
    best_epoch = -1
    start_flag = 0
    max_val_acc = max(history['val_acc'])
    max_val_roc = max(history['val_roc'])
    max_val_f1 = max(history['val_f1'])
    for ratio in [0.95,0.94,0.93,0.92,0.91,0.9]:
        threshold_acc = max_val_acc * ratio
        threshold_roc = max_val_roc * ratio
        threshold_f1 = max_val_f1 * ratio

        for epoch, val_acc, val_roc, val_f1, val_parity, val_equality in zip(range(args.epochs+1), history['val_acc'], history['val_roc'], history['val_f1'], history['val_parity'], history['val_equality']):
            if val_acc >= threshold_acc and val_roc >= threshold_roc and val_f1 >= threshold_f1:
                if start_flag == 0:
                    start_flag = 1
                    start_epoch = epoch
                if val_parity+val_equality < best_fair:
                    best_fair = val_parity+val_equality
                    best_epoch = epoch
        if start_flag == 1:
            break
        else:
            print("choose a smaller ratio!")
            start_epoch = 0
            best_epoch = -1
    epoch_list = [start_epoch, best_epoch]

    validation_metrics = {
        'Accuracy': history['val_acc'][best_epoch] * 100,
        'AUC-ROC': history['val_roc'][best_epoch] * 100,
        'F1-Score': history['val_f1'][best_epoch] * 100,
        'Parity': history['val_parity'][best_epoch] * 100,
        'Equality': history['val_equality'][best_epoch] * 100,
    }

    test_metrics = {
        'Accuracy': history['test_acc'][best_epoch] * 100,
        'AUC-ROC': history['test_roc'][best_epoch] * 100,
        'F1-Score': history['test_f1'][best_epoch] * 100,
        'Parity': history['test_parity'][best_epoch] * 100,
        'Equality': history['test_equality'][best_epoch] * 100,
    }

    table_data = [
        ['Metric'] + list(validation_metrics.keys()) + list(test_metrics.keys()),
        ['Validation Set'] + list(validation_metrics.values()),
        ['Test Set'] + list(test_metrics.values())
    ]

    table = tabulate(table_data, headers='firstrow', tablefmt='plain')
    table = table.replace('|', ' ')
    print(table)

    groups_metric = {
        's0y0': groups_acc_list[best_epoch][0] * 100, 
        's0y1': groups_acc_list[best_epoch][1] * 100, 
        's1y0': groups_acc_list[best_epoch][2] * 100, 
        's1y1': groups_acc_list[best_epoch][3] * 100, 
    }
    table2_data = [
        ['groups'] + list(groups_metric.keys()), 
        ['Accuracy'] + list(groups_metric.values()), 
    ]
    table2 = tabulate(table2_data, headers='firstrow', tablefmt='plain').replace('|', ' ')
    print(table2)

def report(args, base_path, excel=True, draw=False):
    """
    select a more adaptive threshold
    """
    report_path = base_path

    create_directory_safely(report_path) 
    with open(base_path+'/seed'+str(args.seed)+'history.pkl', 'rb') as file:
        history = pickle.load(file)

    best_fair = 100
    best_epoch = -1
    start_flag = 0
    max_val_acc = max(history['val_acc'])
    max_val_roc = max(history['val_roc'])
    max_val_f1 = max(history['val_f1'])
    for ratio in [0.95,0.94,0.93,0.92,0.91,0.9]:
        threshold_acc = max_val_acc * ratio
        threshold_roc = max_val_roc * ratio
        threshold_f1 = max_val_f1 * ratio

        for epoch, val_acc, val_roc, val_f1, val_parity, val_equality in zip(range(args.epochs+1), history['val_acc'], history['val_roc'], history['val_f1'], history['val_parity'], history['val_equality']):
            if val_acc >= threshold_acc and val_roc >= threshold_roc and val_f1 >= threshold_f1:
                if start_flag == 0:
                    start_flag = 1
                    start_epoch = epoch
                if val_parity+val_equality < best_fair:
                    best_fair = val_parity+val_equality
                    best_epoch = epoch
        if start_flag == 1:
            break
        else:
            print("choose a smaller ratio!")
            start_epoch = 0
            best_epoch = -1
    epoch_list = [start_epoch, best_epoch]

    if excel:
        excel_name = args.dataset+'_'+args.model+'_accrocf1.xlsx'
        merge_columns = ['Seed']
        val_metrics = ['valACC', 'valROC', 'valF1', 'valSP', 'valEO']
        test_metrics = ['ACC', 'ROC', 'F1', 'SP', 'EO', 'S0Y0','S0Y1','S1Y0','S1Y1']
        max_columns = ['start_epoch', 'stop_epoch', 'max_val_acc', 'max_val_roc', 'max_val_f1']
        df = pd.DataFrame(columns=merge_columns + val_metrics + test_metrics + max_columns)
        data = {'Seed': [args.seed],
                'start_epoch': [start_epoch],
                'stop_epoch': [best_epoch],
                'max_val_acc': [max_val_acc * 100],
                'max_val_roc': [max_val_roc * 100],
                'max_val_f1': [max_val_f1 * 100],
                'valACC': [history['val_acc'][best_epoch] * 100],
                'valROC': [history['val_roc'][best_epoch] * 100],
                'valF1': [history['val_f1'][best_epoch] * 100],
                'valSP': [history['val_parity'][best_epoch] * 100],
                'valEO': [history['val_equality'][best_epoch] * 100],
                'ACC': [history['test_acc'][best_epoch] * 100],
                'ROC': [history['test_roc'][best_epoch] * 100],
                'F1': [history['test_f1'][best_epoch] * 100],
                'SP': [history['test_parity'][best_epoch] * 100],
                'EO': [history['test_equality'][best_epoch] * 100], 
                'S0Y0': [history['group_acc_list'][best_epoch][0] * 100], 
                'S0Y1': [history['group_acc_list'][best_epoch][1] * 100], 
                'S1Y0': [history['group_acc_list'][best_epoch][2] * 100], 
                'S1Y1': [history['group_acc_list'][best_epoch][3] * 100], }
        for column_name in df.columns:
            df[column_name] = data.get(column_name, [None])
        for column in val_metrics:
            df[column] = df[column].apply(lambda x: round(float(x), 4))
        for column in test_metrics:
            df[column] = df[column].apply(lambda x: round(float(x), 4))
        excel_path = os.path.join(report_path, excel_name)
        if os.path.exists(excel_path):
            existing_df = pd.read_excel(excel_path)
            existing_df.set_index(merge_columns, inplace=True)
            df.set_index(merge_columns, inplace=True)
            existing_df.update(df)
            existing_df = existing_df.combine_first(df)
            existing_df.reset_index(inplace=True)  # reset index
            df = existing_df
        df.to_excel(excel_path, index=False)

        if args.seed == 5:
            if args.model == 'ssf':
                numeric_columns = ['lr', 'wd', 'drop', 'coeff']
                new_df = pd.DataFrame(columns=numeric_columns)
                row = {'lr': args.lr, 'wd': args.weight_decay, 'drop': args.dropout, 'coeff': args.sim_coeff}
                earlystop_path = os.path.join('../', args.dataset, excel_name)
            elif args.model in ['fairgcn','fairsage']:
                numeric_columns = ['lr', 'wd', 'hidden', 'drop', 'alpha', 'beta']
                new_df = pd.DataFrame(columns=numeric_columns)
                row = {'lr': args.lr, 'wd': args.weight_decay, 'hidden': args.num_hidden, 'drop': args.dropout, 'alpha': args.alpha, 'beta': args.beta}
                earlystop_path = os.path.join('../', args.dataset, excel_name)
            elif args.model in ['mlp', 'gcn']:
                numeric_columns = ['layers', 'lr', 'wd', 'drop']
                new_df = pd.DataFrame(columns=numeric_columns)
                row = {'layers': args.num_layers, 'lr': args.lr, 'wd': args.weight_decay, 'drop': args.dropout}
                earlystop_path = os.path.join('../'+args.dataset, excel_name)
                if args.dataset in ['synthetic', 'syn-1', 'syn-2']:
                    numeric_columns = ['n','dy','ds','yscale','sscale','covy','covs','layers', 'lr', 'wd', 'drop']
                    new_df = pd.DataFrame(columns=numeric_columns)
                    row = {'n':args.n,'dy':args.dy,'ds':args.ds,'yscale':args.yscale,'sscale':args.sscale,'covy':args.covy,'covs':args.covs,
                           'layers': args.num_layers, 'lr': args.lr, 'wd': args.weight_decay, 'drop': args.dropout}
                    earlystop_path = os.path.join('../'+args.dataset, 'ES_'+excel_name)            

            for metric in val_metrics+test_metrics:
                metric_values = df[metric].values
                mean = np.mean(metric_values)
                std = np.std(metric_values)
                row[metric] = round(mean,2).astype(str) + ' ± ' + round(std,2).astype(str)
            new_df = new_df.append(row, ignore_index=True)
            if os.path.exists(earlystop_path):
                existing_df = pd.read_excel(earlystop_path)
                existing_df.set_index(numeric_columns, inplace=True)
                new_df.set_index(numeric_columns, inplace=True)
                existing_df.update(new_df)
                existing_df = existing_df.combine_first(new_df)
                existing_df.reset_index(inplace=True)  # reset index
                new_df = existing_df
            if args.model == 'ssf':  
                new_df = new_df[['lr', 'wd', 'drop', 'coeff', 'valACC', 'valROC', 'valF1', 'valSP', 'valEO', 'ACC', 'ROC', 'F1', 'SP', 'EO', 'S0Y0','S0Y1','S1Y0','S1Y1']]
            elif args.model in ['fairgcn','fairsage']:
                new_df = new_df[['lr', 'wd', 'hidden', 'drop', 'alpha', 'beta', 'valACC', 'valROC', 'valF1', 'valSP', 'valEO', 'ACC', 'ROC', 'F1', 'SP', 'EO', 'S0Y0','S0Y1','S1Y0','S1Y1']]
            elif args.model in ['mlp', 'gcn']:
                if args.dataset in ['synthetic', 'syn-1', 'syn-2']:
                    new_df = new_df[['n','dy','ds','yscale','sscale','covy','covs','layers', 'lr', 'wd', 'drop', 'valACC', 'valROC', 'valF1', 'valSP', 'valEO', 'ACC', 'ROC', 'F1', 'SP', 'EO', 'S0Y0','S0Y1','S1Y0','S1Y1']]
                else:
                    new_df = new_df[['layers', 'lr', 'wd', 'drop', 'valACC', 'valROC', 'valF1', 'valSP', 'valEO', 'ACC', 'ROC', 'F1', 'SP', 'EO', 'S0Y0','S0Y1','S1Y0','S1Y1']]
            new_df.to_excel(earlystop_path, index=False)
