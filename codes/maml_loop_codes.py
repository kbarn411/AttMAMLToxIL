import deepchem as dc
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU, Sigmoid
from rdkit import RDLogger
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from utils_codes import *
from learners import *
from attmaml import *

def execute_experiment(config_dict):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    RDLogger.logger().setLevel(RDLogger.ERROR)

    torch.serialization.add_safe_globals([F.relu, F.leaky_relu, ReLU, torch.nn.modules.activation.ReLU,
                                      torch.nn.modules.activation.Sigmoid, Sigmoid,
                                      dc.data.NumpyDataset, np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                      np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.ObjectDType, np.dtypes.Int64DType,
                                      torch.nn.modules.container.ModuleList, torch.nn.modules.linear.Linear, np.core.multiarray.scalar,
                                      torch.clamp, CensoredRegressionLoss, CensoredHybridLoss, IFMLayer])


    seed_value = config_dict["seed_values"][0]

    if config_dict["flag_test_on_new_tasks"]:
        tasks_names_to_test_list = [["S. vacuolatus#EC50", "S. marcescens#MIC", 
                                        "M. catarrhalis#MIC", "P. subcapitata#EC50", 
                                        "L. minor#EC50", "M. luteus#MIC", 
                                        "S. epidermidis#MBC", "P. vulgaris#MIC", 
                                        "A. hydrophila#EC50", "C. albicans#MIC", 
                                        "E. faecalis#MIC", "D. rerio#LC50"]]
    else:
        tasks_names_to_test_list = [['E.coli'], ['AChE'], ['IPC-81'], ['vibrio fischeri']]

    for tasks_names_to_test in tasks_names_to_test_list:

        flag_remove_testing_ils = False
        seed_everything(seed_value)

        df = prepare_df("tox", flag_subseta_only=config_dict["flag_subseta_only"])
        df = normalize_smiles_in_df(df)
        df = drop_duplicates_in_df(df)

        if config_dict["flag_use_same_units"]:
            if tasks_names_to_test == ['E.coli']:
                allowed_values_of_units = "#MIC"
            elif tasks_names_to_test in [['IPC-81'], ['AChE'], ['vibrio fischeri']]:
                allowed_values_of_units = "#EC50"
            
            subset_a_tasks =  ['E.coli', 'AChE', 'IPC-81', 'vibrio fischeri']
            subset_a_tasks.remove(tasks_names_to_test[0])
            subset_a_units = {}
            for el in subset_a_tasks:
                if el == 'E.coli':
                    subset_a_units[el] = f"{el}#MIC"
                else:
                    subset_a_units[el] = f"{el}#EC50"
            df['task_target'] = df['task_target'].replace(subset_a_units)
            
            mask_contains = df['task_target'].str.contains(allowed_values_of_units, case=False, na=False)
            mask_exact = df['task_target'] == tasks_names_to_test[0]
            combined_mask = mask_contains | mask_exact
            df = df[combined_mask]
            df.reset_index(drop=True, inplace=True)

        if config_dict["flag_use_similar_tasks"]:
            if tasks_names_to_test == ['IPC-81']:
                allowed_values_of_tasks = ["CaCo-2#EC50", "HT-29#EC50", "HeLa#EC50", "CaCo-2#IC50", "HeLa#IC50", "HepG2#IC50", "IPC-81"]
            elif tasks_names_to_test in [['E.coli'], ['vibrio fischeri']]:
                allowed_values_of_tasks = ["S. aureus#EC50", "S. aureus#MIC", "S. aureus#MBC", "E.coli", "vibrio fischeri"]
            elif tasks_names_to_test == ['AChE']:
                allowed_values_of_tasks = ['E.coli', 'AChE', 'IPC-81', 'vibrio fischeri']
            df = df[df["task_target"].isin(allowed_values_of_tasks)]
            df.reset_index(drop=True, inplace=True)

        if config_dict["flag_subseta_only"] and not config_dict["flag_test_on_new_tasks"]:
            allowed_values_of_tasks = ['E.coli', 'AChE', 'IPC-81', 'vibrio fischeri']
            df = df[df["task_target"].isin(allowed_values_of_tasks)]
            df.reset_index(drop=True, inplace=True)

        if config_dict["flag_test_on_new_tasks"]:
            exclude_list = ['E.coli', 'AChE', 'IPC-81', 'vibrio fischeri']

            if config_dict["flag_subseta_only"]:
                list_of_tasks_final = [*exclude_list, *tasks_names_to_test]
                df = df[df["task_target"].isin(list_of_tasks_final)]
                df.reset_index(drop=True, inplace=True)
            else:
                pass
        
        with open(os.path.join(config_dict["root_dir"], 'log.txt'), 'a') as f:
            f.write(f'LOG | SEED {seed_value} | TARGET {tasks_names_to_test[0]} | REM TEST ILs {flag_remove_testing_ils} \n')
        
        if flag_remove_testing_ils: df = remove_testing_ils(df, tasks_names_to_test[0])

        summary_of_results = []
        for shot_size in config_dict["shot_sizes_to_test"]:
            metaparams = None
            metrics_to_save = create_metrics_to_save(tasks_names_to_test)
            folds_to_use = [0] if config_dict["flag_use_validation"] == 0 else list(range(5))
            for fold_to_use in folds_to_use:

                df_fold, tasks, y, tasks_dict, tasks_names_to_test, use_validation = prepare_splits(df, fold_to_use, shot_size, config_dict["validation_schema_to_test"], tasks_names_to_test, config_dict["flag_use_validation"], seed_value)
                censorslist = df_fold['censorship'].values if 'censorship' in df_fold.columns else None

                if not metaparams: ecfp, columns_names_ecfp = calculate_descriptors_x(df_fold, config_dict["descriptors_type"])

                if config_dict["descriptors_type"] != 'tr' and not metaparams:
                    ecfp, columns_names_ecfp = drop_all_none_features(ecfp, columns_names_ecfp)
                    ecfp, columns_names_ecfp = drop_zero_variance_features(ecfp, columns_names_ecfp)
                    ecfp, columns_names_ecfp = drop_highly_correlated_features(ecfp, columns_names_ecfp, threshold=0.95, descriptors_type=config_dict["descriptors_type"])

                if not metaparams:
                    scaler_ecfp = MinMaxScaler()
                    scaler_ecfp = scaler_ecfp.fit(ecfp)
                    ecfp = scaler_ecfp.transform(ecfp)

                nn_layer_sizes = [ecfp.shape[1], 16, 1]

                if not metaparams:
                    start_params = None

                if config_dict["transform_y"]:
                    if config_dict["flag_per_task_transform"]:
                        y, task_scalers = transform_y_func(y, df=df_fold, tasks_column_name='task_target', flags=config_dict)
                    else:
                        y, pt_target, min_max_scaler_y = transform_y_func(y, df=df_fold, tasks_column_name='task_target', flags=config_dict)
                        task_scalers = [pt_target, min_max_scaler_y]

                dataset = dc.data.NumpyDataset(X=ecfp, y=y, ids=tasks, w=censorslist)

                full_tasks_names_to_test = get_full_tasks_names_to_test(tasks_names_to_test, tasks_dict)
                tasks_to_test = [tasks_dict[task] for task in full_tasks_names_to_test]

                if config_dict["flag_use_ifm"]:
                    if config_dict["flag_attmaml"]:
                        learner = MetaIFMMFLearner(
                            layer_sizes=nn_layer_sizes,
                            dataset=dataset, batch_size=shot_size,
                            tasks_dict=tasks_dict, test_tasks=tasks_to_test,
                            activation=F.leaky_relu,
                        )
                    else:
                        learner = MetaIFMMFLearner_classic(
                            layer_sizes=nn_layer_sizes,
                            dataset=dataset, batch_size=shot_size,
                            tasks_dict=tasks_dict, test_tasks=tasks_to_test,
                            activation=F.leaky_relu, lossfn=CensoredHybridLoss
                        )
                    outer_lr = 5e-4 
                    inner_lr = 1e-4
                else:
                    learner = MetaMFLearner(
                        layer_sizes=nn_layer_sizes,
                        dataset=dataset, batch_size=shot_size,
                        tasks_dict=tasks_dict, test_tasks=tasks_to_test,
                        activation=F.leaky_relu
                        )
                    outer_lr = 5e-5
                    inner_lr = 1e-4 

                if start_params and not metaparams:
                    learner.set_params_to_layers_from_list(start_params)

                if metaparams:
                    learner.set_params_to_layers_from_list(metaparams)

                optimizer = dc.models.optimizers.Adam(learning_rate=outer_lr)
                if config_dict["flag_attmaml"]:
                    maml = AttMAML(learner, optimization_steps=config_dict["optimization_steps"], learning_rate=inner_lr,
                                meta_batch_size=config_dict["meta_batch_size"], optimizer=optimizer, device=device)
                else:
                    maml = MAML(learner, optimization_steps=config_dict["optimization_steps"], learning_rate=inner_lr,
                                meta_batch_size=config_dict["meta_batch_size"], optimizer=optimizer, device=device)
                if not metaparams:
                    n_episodes = config_dict["base_n_of_episodes"]
                else:
                    n_episodes = 1
                maml.fit(n_episodes)
                metaparams = learner.variables

                if config_dict["flag_two_losses"]:
                    learner.switch_loss()
                    
                metrics_to_save = finetune_and_eval_maml_on_test_tasks(
                    learner, maml, tasks_dict, metrics_to_save, config_dict["list_with_ft_episodes"],
                    tasks_names_to_test, df_fold, y, ecfp, censorslist, config_dict, task_scalers, use_validation)

                for compound in tasks_names_to_test:
                    metrics_to_save_mean = copy.deepcopy(metrics_to_save[compound])

                    print(f"COMPOUND {compound} | SHOT SIZE {shot_size} | VALIDATION SCHEMA {config_dict['validation_schema_to_test']}")
                    for key in metrics_to_save[compound].keys():
                        for metric in metrics_to_save[compound][key].keys():
                            metrics_to_save_mean[key][metric] = np.mean(metrics_to_save[compound][key][metric])
                            metric_std = np.std(metrics_to_save[compound][key][metric])
                            metric_value = metrics_to_save_mean[key][metric]
                            if not np.isnan(metric_value) and config_dict["flag_verbose"]:
                                print(f"RESULTS | MEAN | {key:<13} {metric:<6} = {metric_value:12.3f} | STD = {metric_std:10.4f}")
                    summary_of_results.append([shot_size, config_dict["validation_schema_to_test"], metrics_to_save])
