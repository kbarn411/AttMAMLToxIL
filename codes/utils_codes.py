import deepchem as dc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.activation import ReLU, Sigmoid
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, RobertaTokenizer, RobertaModel
from deepchem.metalearning.torch_maml import MetaLearner, MAML
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, PolynomialFeatures, PowerTransformer, QuantileTransformer
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import math
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, SDWriter, Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, AddHs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import DataStructs
from rdkit import RDLogger
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import copy
from typing import List, Optional
from collections import defaultdict
import re
import os
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RegressionMetric:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true).ravel()
        self.y_pred = np.array(y_pred).ravel()

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def rmse(self):
        return root_mean_squared_error(self.y_true, self.y_pred)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def mape(self):
        mask = self.y_true != 0
        return np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100

    def get_metric_by_name(self, metric_name: str):
        if metric_name == 'R2':
            return {metric_name: self.r2()}
        elif metric_name == 'RMSE':
            return {metric_name: self.rmse()}
        elif metric_name == 'MAE':
            return {metric_name: self.mae()}
        elif metric_name == 'MAPE':
            return {metric_name: self.mape()}
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def expand_smiles_to_cation_and_anion(df):

    df['smiles_cation'] = df['smiles'].apply(
        lambda x: next((part for part in x.split('.') if '+' in part), None)
    )

    df['smiles_anion'] = df['smiles'].apply(
        lambda x: next((part for part in x.split('.') if '-' in part), None)
    )

    return df

def smiles_to_mol_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.MolWt(mol)

def normalize_units_in_toxdb(row):
    if pd.isna(row['y']):
        return np.nan

    mass_units = [
        "mg/l", "mg per l", "mg/L",
        "g/l", "g per l", "g/L", "mg/mL", "mg/ml",
        "kg/l", "kg per l", "g/mL",
        "ug/L"
    ]
        
    if row['unit'] in ('μM', 'µM', 'μmol/L', 'μmolL'):
        return np.log(row['y'])
    elif row['unit'] in ('mM', 'mmol/L'):
        return np.log(row['y'] * 1e3)
    elif row['unit'] in ('M', 'mol/L'):
        return np.log(row['y'] * 1e6)
    elif row['unit'] in ('nM'):
        return np.log(row['y'] * 1e-3) 
    elif row['unit'] in ('log(mM)'):
        return row['y'] + 3
    elif row['unit'] in ('p(mol/L)'):
        return -1 * row['y'] + 6
    elif row['unit'] in ('log(M)'):
        return row['y'] + 6
    elif row['unit'] in ('log(μmol/L)', 'log(μM)'):
        return row['y']
    
    elif row['unit'] in mass_units:
        if row['unit'] in mass_units[0:3]:
            grams_per_l = row['y'] * 1e-3
        elif row['unit'] in mass_units[3:8]:
            grams_per_l = row['y'] * 1.0
        elif row['unit'] in mass_units[8:11]:
            grams_per_l = row['y'] * 1e3
        elif row['unit'] in mass_units[11:]:
            grams_per_l = row['y'] * 1e-6

        molar_mass = smiles_to_mol_weight(row['SMILES'])
        mol_per_l = grams_per_l / molar_mass
        return np.log(mol_per_l * 1e6)
    
    else:
        return np.nan
    
def clean_y_and_censorship(row):
    raw_value = str(row['y']).strip()
    censorship = 0
    
    value = raw_value

    if '±' in value:
        value = value.split('±')[0].strip()

    if value.startswith(">=") or value.startswith(">"):
        censorship = 1
        value = re.sub(r'^[>=\s]+', '', value).strip()
    elif value.startswith("<=") or value.startswith("<"):
        censorship = -1
        value = re.sub(r'^[<=\s]+', '', value).strip()

    if value.startswith("~") or value.startswith("≈"):
        value = value[1:].strip()

    if re.match(r'^\d+(\.\d+)?\s*-\s*\d+(\.\d+)?$', value):
        nums = [float(x) for x in re.split(r'-', value)]
        value = str(sum(nums)/len(nums))
        censorship = 0

    if value.upper() in ["ND", "N/A", "NA", "NONE", "NULL", ""]:
        return pd.Series([pd.NA, pd.NA], index=['y_clean', 'censorship'])

    value = re.sub(r'[<>≈~]', '', value).strip()

    try:
        value = float(value)
    except ValueError:
        return pd.Series([pd.NA, pd.NA], index=['y_clean', 'censorship'])

    return pd.Series([value, censorship], index=['y_clean', 'censorship'])

def prepare_tox_df(flag_subseta_only):
    xls = pd.ExcelFile('../data/Data.xlsx')
    
    dfs = []
    for sheet in xls.sheet_names:
        df_part = pd.read_excel(xls, sheet)
        df_part['task_target'] = sheet
        
        y_col = [col for col in df_part.columns if 'μM' in col][0]
        df_part.rename(columns={y_col: 'y'}, inplace=True)
        df_part['y'] = df_part['y'].astype(float)
        df_part['censorship'] = 0
        
        dfs.append(df_part)
       
    list_of_omittable_xls2 = ["AChE inhibition#EC50", "AChE inhibition#IC50", 
                              "V. fischeri#EC50", "E. coli#EC50", 
                              "E. coli#IC50", "E. coli#MIC", 
                              "E. coli#MBC", "IPC-81#EC50"]
    
    if not flag_subseta_only:
      xls2 = pd.ExcelFile('../data/Iltox-full.xlsx')
      for sheet in xls2.sheet_names:
          if sheet in list_of_omittable_xls2:
              continue
          
          df_part = pd.read_excel(xls2, sheet)
          df_part['task_target'] = sheet 
          df_part['Split'] = np.where(np.random.rand(len(df_part)) < 0.2, 'Test', 'Training')
      
          unit_col = [col for col in df_part.columns if 'unit' in col][0]
          df_part.rename(columns={
              "Value": 'y', unit_col: 'unit'
          }, inplace=True)
      
          df_part['censorship'] = 0
          df_part[['y', 'censorship']] = df_part.apply(clean_y_and_censorship, axis=1)
          df_part = df_part[df_part['y'] > 0]
          df_part['y'] = df_part.apply(normalize_units_in_toxdb, axis=1)
          df_part = df_part[df_part['y'].notna()]
          
          dfs.append(df_part)
      
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    df.drop(["NO.", "Standard Value"], axis=1, inplace=True)
    if "unit" in df.columns: df.drop(["unit"], axis=1, inplace=True)
    df.rename(columns={"SMILES": "smiles"}, inplace=True)
    df = expand_smiles_to_cation_and_anion(df)
    df['smiles'] = df['smiles_cation'] + '.' + df['smiles_anion']

    cols_to_float = ['y', 'censorship']
    df[cols_to_float] = df[cols_to_float].astype(float)

    return df

def prepare_df(ds_name, flag_subseta_only):
    if ds_name == 'tox':
      df = prepare_tox_df(flag_subseta_only)
      return df
    else:
      print("Unknown dataset")
      return None
    
def remove_testing_ils(df, test_task):
  condition = (df['task_target'] == test_task) & (df['Split'] == 'Test')
  testing_ils = df[condition]['smiles'].unique()
  condition_to_remove = (df['task_target'] != test_task) & (df['smiles'].isin(testing_ils))
  df = df[~condition_to_remove]
  df = df.reset_index(drop=True)
  return df

def normalize_smiles(smile, nrm):
  mol = Chem.MolFromSmiles(smile)
  mol_norm = nrm.normalize(mol)
  smile_norm = Chem.MolToSmiles(mol_norm, True)
  return smile_norm

def normalize_smiles_in_df(df):
  nrm = rdMolStandardize.Normalizer()
  smiles_cation_unique = df['smiles_cation'].unique()
  smiles_anion_unique = df['smiles_anion'].unique()

  smiles_cation_norm_dict = {}
  smiles_anion_norm_dict = {}

  for smile in smiles_cation_unique:
    smiles_cation_norm_dict[smile] = normalize_smiles(smile, nrm)
  for smile in smiles_anion_unique:
    smiles_anion_norm_dict[smile] = normalize_smiles(smile, nrm)

  df['smiles_cation'] = df['smiles_cation'].map(smiles_cation_norm_dict)
  df['smiles_anion'] = df['smiles_anion'].map(smiles_anion_norm_dict)
  return df

def drop_duplicates_in_df(df):
  df.drop_duplicates(['smiles_cation', 'smiles_anion', 'y', 'task_target'], inplace=True)
  df = df.reset_index(drop=True)
  return df

def prepare_splits(df, fold_to_use, shot_size, validation_schema, tasks_names_to_test, flag_use_validation, seed_value=42):
  df['task_target'] = df['task_target'].apply(lambda x:
                                                      x.replace('_test', '').replace('_valid', '').replace('_adjustval', ''))

  if flag_use_validation == None:
    use_validation = {}
  elif flag_use_validation == 0:
    use_validation = {}
    for comp in tasks_names_to_test:
      use_validation[comp] = False
  elif flag_use_validation == 1:
    use_validation = {}
    for comp in tasks_names_to_test:
      use_validation[comp] = True

  for task in tasks_names_to_test:
    if flag_use_validation == None: use_validation[task] = True
    df_task_test = df[(df['task_target'] == task) & (df['censorship'] == 0) & (df['Split'] == 'Test')].copy()
    df.loc[df_task_test.index, 'task_target'] = task + f'_test'

    if use_validation[task]:
      df_task = df[(df['task_target'] == task) & (df['censorship'] == 0) & (df['Split'] == 'Training')].copy()
      
      if validation_schema == 'random':
        splitter_to_valid = KFold(n_splits=5, shuffle=True, random_state=seed_value)
        splits_to_enum = splitter_to_valid.split(df_task)
      elif validation_schema == 'smiles':
        splitter_to_valid = GroupKFold(n_splits=5, shuffle=True, random_state=seed_value)
        splits_to_enum = splitter_to_valid.split(df_task, df_task['y'], df_task['smiles'])

      for fold_index, (train_index, valid_index) in enumerate(splits_to_enum):
        if fold_index == fold_to_use:
          df.loc[df_task.iloc[valid_index].index, 'task_target'] = task + f'_valid'

    df_task = df[df['task_target'] == task].copy()
    excess_samples = df_task[df_task['task_target'] == task].shape[0] - shot_size
    if excess_samples > 0:
      df.loc[df_task[df_task['task_target'] == task].sample(n=excess_samples, random_state=seed_value).index, 'task_target'] = task + f'_adjustval'

  tasks = df['task_target'].values
  tasks_dict = {task: i for i, task in enumerate(np.unique(tasks))}
  tasks = np.array([tasks_dict[task] for task in tasks])

  y = df['y'].values.copy()

  return df, tasks, y, tasks_dict, tasks_names_to_test, use_validation

def create_metrics_to_save(compounds_names_to_test):
  metrics_to_save = {}
  for compound in compounds_names_to_test:
    metrics_to_save[compound] = {
      'train': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
      'cv' : {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
      'cv_classic' : {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
      'test': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
      'test_classic': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
    }
  return metrics_to_save

def calculate_descriptors_x_twod(df):
  unique_smiles_cations = df["smiles_cation"].unique()
  unique_smiles_anions = df["smiles_anion"].unique()

  lookup_cations = {}
  lookup_anions = {}

  for smiles_str in unique_smiles_cations:
    mol = Chem.MolFromSmiles(smiles_str)
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol)
    if result != 0:
      print(f"Failed to embed molecule for SMILES: {smiles_str}")
    lookup_cations[smiles_str] = Descriptors.CalcMolDescriptors(mol)
  for smiles_str in unique_smiles_anions:
    mol = Chem.MolFromSmiles(smiles_str)
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol)
    if result != 0:
      print(f"Failed to embed molecule for SMILES: {smiles_str}")
    lookup_anions[smiles_str] = Descriptors.CalcMolDescriptors(mol)
  
  X_cations = pd.DataFrame.from_dict(lookup_cations, orient='index')
  X_anions = pd.DataFrame.from_dict(lookup_anions, orient='index')

  X_cations.index=unique_smiles_cations
  X_anions.index=unique_smiles_anions

  ecfp_features = []

  for index, row in tqdm(df.iterrows(), total=len(df)):
      cation_smiles = row['smiles_cation']
      anion_smiles = row['smiles_anion']

      cation_features = X_cations.loc[cation_smiles].values if cation_smiles in X_cations.index else np.zeros(X_cations.shape[1])
      anion_features = X_anions.loc[anion_smiles].values if anion_smiles in X_anions.index else np.zeros(X_anions.shape[1])

      combined_features = np.concatenate([cation_features, anion_features])

      ecfp_features.append(combined_features)

  ecfp = np.array(ecfp_features)


  cations_cols = [f'cat_{col}' for col in X_cations.columns]
  anions_cols = [f'an_{col}' for col in X_anions.columns]
  columns_names_ecfp = cations_cols + anions_cols

  return ecfp, np.array(columns_names_ecfp)

def calculate_descriptors_tr(df):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_chemberta = "DeepChem/ChemBERTa-77M-MTR" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_chemberta)
    model = AutoModel.from_pretrained(model_name_chemberta).to(device)
    model.eval()

    embeddings_cache = {}  

    def get_embedding(smi):
        if smi not in embeddings_cache:
            inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            embeddings_cache[smi] = embeddings
        return embeddings_cache[smi]

    embeddings_c, embeddings_a = [], []

    for smi in df['smiles_cation'].values:
        embeddings_c.append(get_embedding(smi))

    for smi in df['smiles_anion'].values:
        embeddings_a.append(get_embedding(smi))

    embeddings_c = np.array(embeddings_c, dtype=float)
    embeddings_a = np.array(embeddings_a, dtype=float)

    chemberta_data = np.concatenate((embeddings_c, embeddings_a), axis=1)

    return chemberta_data, np.array(['chemberta_' + str(i) for i in range(chemberta_data.shape[1])])

def calculate_descriptors_x(df, descriptors_type):
    if descriptors_type == 'twod':
      ecfp, columns_names_ecfp = calculate_descriptors_x_twod(df)
    if descriptors_type == 'tr':
      ecfp, columns_names_ecfp = calculate_descriptors_tr(df)
    return ecfp, columns_names_ecfp

def drop_all_none_features(ecfp, columns_names_ecfp):

    all_none_cols = np.all(np.isnan(ecfp), axis=0)

    if not np.any(all_none_cols):
        return ecfp, columns_names_ecfp

    keep_cols = np.where(~all_none_cols)[0]
    filtered_ecfp = ecfp[:, keep_cols]
    filtered_columns_names_ecfp = columns_names_ecfp[keep_cols]

    return filtered_ecfp, filtered_columns_names_ecfp

def drop_zero_variance_features(ecfp, columns_names_ecfp):
  variances = np.var(ecfp, axis=0)
  zero_variance_cols = np.where(variances == 0)[0]

  if len(zero_variance_cols) == 0:
    return ecfp, columns_names_ecfp

  keep_cols = np.where(variances != 0)[0]
  filtered_ecfp = ecfp[:, keep_cols]
  filtered_columns_names_ecfp = columns_names_ecfp[keep_cols]

  return filtered_ecfp, filtered_columns_names_ecfp

def drop_highly_correlated_features(ecfp, columns_names_ecfp, threshold, descriptors_type):

  if descriptors_type == 'twod':
    ecfp = np.nan_to_num(ecfp, nan=np.nanmean(ecfp))
    ecfp = (ecfp - np.mean(ecfp, axis=0)) / np.std(ecfp, axis=0)

  corr_matrix = np.corrcoef(ecfp, rowvar=False)
  upper_tri = np.triu_indices(corr_matrix.shape[0], k=1)
  highly_correlated_cols = np.where(np.abs(corr_matrix[upper_tri]) > threshold)[0]

  if len(highly_correlated_cols) == 0 :
    return ecfp, columns_names_ecfp

  drop_cols = []
  for i in highly_correlated_cols:
      col_idx = upper_tri[1][i]
      drop_cols.append(col_idx)

  columns_names_ecfp = np.array(columns_names_ecfp)
  keep_cols = np.array([i for i in range(len(columns_names_ecfp)) if i not in drop_cols])

  filtered_ecfp = ecfp[:, keep_cols]
  filtered_columns_names_ecfp = columns_names_ecfp[keep_cols]

  return filtered_ecfp, filtered_columns_names_ecfp

def transform_y_func(y, df=None, tasks_column_name=None, flags=None):
    if flags["transform_y"] and not flags["flag_per_task_transform"]:
      return transform_y_func_pertask(y, df, flags)
    else:
      return transform_y_func_nontask(y, df, tasks_column_name, flags)

def reverse_transform_y(y_transformed, scalers, flags=None, task=None):
    if flags["transform_y"] and not flags["flag_per_task_transform"]:
      return reverse_transform_y_pertask(y_transformed, scalers[0], scalers[1], flags)
    else:
      return reverse_transform_y_nontask(y_transformed, task, scalers, flags)

def transform_y_func_pertask(y, df, flags=None):

    pattern = r"(?:_test|_valid|_adjustvalid|_cosmo)$"
    mask = ~df['task_target'].str.contains(pattern, regex=True)
    y_base = y[mask]

    if flags["apply_log_flag"]: y = -np.log10(y)

    if flags["apply_power_flag"]:

      if flags["power_transform_type"] == 'robust':
        pt_target = RobustScaler()
      elif flags["power_transform_type"] == 'box-cox':
        pt_target = PowerTransformer(method='box-cox')
      elif flags["power_transform_type"] == 'yeo-johnson':
        pt_target = PowerTransformer(method='yeo-johnson')
      elif flags["power_transform_type"] == 'quantile':
        pt_target = QuantileTransformer(output_distribution='normal')
      elif 'quantile_' in flags["power_transform_type"]:
        n_quantiles = int(flags["power_transform_type"].split('_')[-1])
        pt_target = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles)

      pt_target = pt_target.fit(y_base.reshape(-1, 1))
      y = pt_target.transform(y.reshape(-1, 1))
    else:
      pt_target = None
    if flags["apply_minmax_flag"]:
      min_max_scaler_y = MinMaxScaler()
      min_max_scaler_y = min_max_scaler_y.fit(y_base.reshape(-1, 1))
      y = min_max_scaler_y.transform(y.reshape(-1, 1))
    else:
      min_max_scaler_y = None
    return y, pt_target, min_max_scaler_y

def reverse_transform_y_pertask(y_transformed, pt_target, min_max_scaler_y, flags=None):
    if flags["apply_minmax_flag"]: y_transformed = min_max_scaler_y.inverse_transform(y_transformed.reshape(-1, 1))
    if flags["apply_power_flag"]: y_transformed = pt_target.inverse_transform(y_transformed.reshape(-1, 1))
    if flags["apply_log_flag"]: y_transformed = np.power(10, (-1) * y_transformed.reshape(-1, 1))
    return y_transformed

def get_base_task_name(task_name):
      return re.split(r"_test|_valid|_adjustvalid|_", task_name)[0]

def transform_y_func_nontask(y, df=None, tasks_column_name=None, flags=None):
      if df is None or tasks_column_name is None:
          raise ValueError("Both df and tasks_column_name must be provided for per-task transformation.")

      y_transformed = np.empty_like(y, dtype=float)
      task_scalers_dict = {}

      unique_tasks = df[tasks_column_name].unique()

      for task in unique_tasks:
          base_task = get_base_task_name(task)
          is_same_base = df[tasks_column_name].apply(lambda x: get_base_task_name(x) == base_task)
          is_base_task = df[tasks_column_name] == base_task
          is_current_task = df[tasks_column_name] == task

          if base_task not in task_scalers_dict:
              y_base = y[is_base_task.to_numpy()]
              if flags["apply_log_flag"]:
                  y_base = -np.log10(y_base)
              if flags["apply_power_flag"]:
                  if flags["power_transform_type"] == 'robust':
                      pt_target = RobustScaler()
                  elif flags["power_transform_type"] == 'box-cox':
                      pt_target = PowerTransformer(method='box-cox')
                  elif flags["power_transform_type"] == 'yeo-johnson':
                      pt_target = PowerTransformer(method='yeo-johnson')
                  elif flags["power_transform_type"] == 'quantile':
                      pt_target = QuantileTransformer(output_distribution='normal')
                  elif 'quantile_' in flags["power_transform_type"]:
                      n_quantiles = int(flags["power_transform_type"].split("_")[-1])
                      pt_target = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles)
                  pt_target.fit(y_base.reshape(-1, 1))
              else:
                  pt_target = None

              if flags["apply_minmax_flag"]:
                  min_max_scaler_y = MinMaxScaler().fit(
                      pt_target.transform(y_base.reshape(-1, 1)) if pt_target else y_base.reshape(-1, 1)
                  )
              else:
                  min_max_scaler_y = None

              task_scalers_dict[base_task] = (pt_target, min_max_scaler_y)

          y_current = y[is_current_task.to_numpy()]
          if flags["apply_log_flag"]:
              y_current = -np.log10(y_current)

          pt_target, min_max_scaler_y = task_scalers_dict[base_task]
          if pt_target:
              y_current = pt_target.transform(y_current.reshape(-1, 1))
          if min_max_scaler_y:
              y_current = min_max_scaler_y.transform(y_current.reshape(-1, 1))

          y_transformed[is_current_task.to_numpy()] = y_current.ravel()

      return y_transformed, task_scalers_dict

def reverse_transform_y_nontask(y_transformed, task, task_scalers, flags):
      base_task = get_base_task_name(task)
      pt_target, min_max_scaler_y = task_scalers[base_task]

      if flags["apply_minmax_flag"]:
          y_transformed = min_max_scaler_y.inverse_transform(y_transformed.reshape(-1, 1))
      if pt_target:
          y_transformed = pt_target.inverse_transform(y_transformed.reshape(-1, 1))
      if flags["apply_log_flag"]:
          y_transformed = np.power(10, (-1) * y_transformed.reshape(-1, 1))

      return y_transformed

def get_full_tasks_names_to_test(tasks_names_to_test, tasks_dict):
  full_tasks_names_to_test = []
  for el in tasks_names_to_test:
    full_tasks_names_to_test.append(el)
    if f"{el}_val" in tasks_dict.keys():
      full_tasks_names_to_test.append(f"{el}_valid")
    full_tasks_names_to_test.append(f"{el}_test")
    if f"{el}_adjustval" in tasks_dict.keys():
      full_tasks_names_to_test.append(f"{el}_adjustval")
  return full_tasks_names_to_test

def get_batch_from_data_by_indices(indexes, y, ecfp, censorslist, flag_for_screening=False):
  if not flag_for_screening:
    y_true_tra = torch.tensor(y[indexes], dtype=torch.float32).view(-1, 1)
    x_test_tra = torch.tensor(ecfp[indexes, :], dtype=torch.float32)
    censorsval = torch.tensor(censorslist[indexes], dtype=torch.float32).view(-1, 1)
  else:
    y_true_tra = torch.zeros(ecfp.shape[0], dtype=torch.float32).view(-1, 1)
    x_test_tra = torch.tensor(ecfp, dtype=torch.float32)
    censorsval = torch.zeros(ecfp.shape[0], dtype=torch.float32).view(-1, 1)
  return [x_test_tra, y_true_tra, censorsval]

def finetune_and_eval_maml_on_test_tasks(learner, maml, tasks_dict, metrics_to_save, num_iterations_list, tasks_names_to_test, df, y, ecfp, censorslist, config_dict, task_scalers, use_validation):

    for val_task in tasks_names_to_test:
      compound = val_task
      iters_metrics_to_save = {
        'train': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'cv' : {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'cv_classic' : {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'test': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'test_classic': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
      }

      list_metrics = ["R2", "RMSE", "MAE", "MAPE"]

      for num_iterations in tqdm(num_iterations_list):

        maml.restore()
        learner.select_task_by_name(val_task)
        batch = learner.get_batch()
        loss, outputs = maml.predict_on_batch(batch)
        maml.train_on_current_task(num_iterations, restore=False)
        loss, outputs = maml.predict_on_batch(batch)
        current_task_name = get_key_by_value(tasks_dict, learner.task_index)

        indexes = df[df['task_target'] == f"{val_task}"].index
        batch_tra = get_batch_from_data_by_indices(indexes, y, ecfp, censorslist)
        loss, outputs = maml.predict_on_batch(batch_tra)

        y_pred_tra = outputs[0].cpu().detach().numpy()
        y_true_tra = batch_tra[1].cpu().detach().numpy()

        if config_dict["transform_y"]:
          y_pred_tra = reverse_transform_y(y_pred_tra, task_scalers, config_dict, val_task)
          y_true_tra = reverse_transform_y(y_true_tra, task_scalers, config_dict, val_task)

        evaluator = RegressionMetric(y_true_tra, y_pred_tra)
        for metric in list_metrics:
            temp_metric_value = evaluator.get_metric_by_name(metric)[metric]
            iters_metrics_to_save['train'][metric].append(temp_metric_value)

        if use_validation[compound]:
          indexes = df[df['task_target'] == f"{val_task}_valid"].index
          batch_val = get_batch_from_data_by_indices(indexes, y, ecfp, censorslist)
          loss, outputs = maml.predict_on_batch(batch_val)

          y_pred_val = outputs[0].cpu().detach().numpy()
          y_true_val = batch_val[1].cpu().detach().numpy()

          if config_dict["transform_y"]:
            y_pred_val = reverse_transform_y(y_pred_val, task_scalers, config_dict, val_task)
            y_true_val = reverse_transform_y(y_true_val, task_scalers, config_dict, val_task)

          evaluator = RegressionMetric(y_true_val, y_pred_val)
          for metric in list_metrics:
            temp_metric_value = evaluator.get_metric_by_name(metric)[metric]
            iters_metrics_to_save['cv'][metric].append(temp_metric_value)

          if num_iterations == num_iterations_list[-1]:
            if config_dict["flag_compare_to_onetask_ml"]:
              indexes_classical = df[df['task_target'].isin([f"{val_task}"])].index
            else:
              indexes_classical = df[~df['task_target'].isin([f"{val_task}_valid", f"{val_task}_test", f"{val_task}_adjustval"])].index
            X_train_class = ecfp[indexes_classical, :]
            y_train_class = y[indexes_classical].ravel()
            model_lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, force_col_wise="true", verbose=-1)
            model_lgbm.fit(X_train_class, y_train_class)
            indexes_class_test = df[df['task_target'] == f"{val_task}_valid"].index
            X_test_class = ecfp[indexes_class_test, :]
            y_test_class = y[indexes_class_test].ravel()
            y_pred_class = model_lgbm.predict(X_test_class)

            if config_dict["transform_y"]:
              y_pred_class = reverse_transform_y(y_pred_class, task_scalers, config_dict, val_task)
              y_test_class = reverse_transform_y(y_test_class, task_scalers, config_dict, val_task)
            evaluator = RegressionMetric(y_test_class, y_pred_class)
            for metric in list_metrics:
              temp_metric_value = evaluator.get_metric_by_name(metric)[metric]
              iters_metrics_to_save['cv_classic'][metric].append(temp_metric_value)

        indexes = df[df['task_target'] == f"{val_task}_test"].index
        batch_test = get_batch_from_data_by_indices(indexes, y, ecfp, censorslist)
        loss, outputs = maml.predict_on_batch(batch_test)

        y_pred = outputs[0].cpu().detach().numpy()
        y_true = batch_test[1].cpu().detach().numpy()

        if config_dict["transform_y"]:
          y_pred = reverse_transform_y(y_pred, task_scalers, config_dict, val_task)
          y_true = reverse_transform_y(y_true, task_scalers, config_dict, val_task)

        evaluator = RegressionMetric(y_true, y_pred)
        for metric in list_metrics:
          temp_metric_value = evaluator.get_metric_by_name(metric)[metric]
          iters_metrics_to_save['test'][metric].append(temp_metric_value)

        if num_iterations == num_iterations_list[-1]:
          if not use_validation[compound]:
            if config_dict["flag_compare_to_onetask_ml"] and config_dict["flag_use_whole_ds_for_classical"]:
              indexes_classical = df[df['task_target'].isin([f"{val_task}", f"{val_task}_valid", f"{val_task}_adjustval"])].index
            elif config_dict["flag_compare_to_onetask_ml"]:
              indexes_classical = df[df['task_target'].isin([f"{val_task}"])].index
            else:
              indexes_classical = df[~df['task_target'].isin([f"{val_task}_valid", f"{val_task}_test", f"{val_task}_adjustval"])].index
            X_train_class = ecfp[indexes_classical, :]
            y_train_class = y[indexes_classical].ravel()
            model_lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, force_col_wise="true", verbose=-1)
            model_lgbm.fit(X_train_class, y_train_class)

          indexes_class_test = df[df['task_target'] == f"{val_task}_test"].index
          X_test_class = ecfp[indexes_class_test, :]
          y_test_class = y[indexes_class_test].ravel()
          y_pred_class = model_lgbm.predict(X_test_class)

          if config_dict["transform_y"]:
            y_pred_class = reverse_transform_y(y_pred_class, task_scalers, config_dict, val_task)
            y_test_class = reverse_transform_y(y_test_class, task_scalers, config_dict, val_task)
          evaluator = RegressionMetric(y_test_class, y_pred_class)
          for metric in list_metrics:
            temp_metric_value = evaluator.get_metric_by_name(metric)[metric]
            iters_metrics_to_save['test_classic'][metric].append(temp_metric_value)

      if use_validation[compound]:
        max_pos = iters_metrics_to_save['cv']['R2'].index(max(iters_metrics_to_save['cv']['R2']))

      else: max_pos = iters_metrics_to_save['test']['RMSE'].index(min(iters_metrics_to_save['test']['RMSE']))
      print(f"LOG | COMPOUND {compound} | max_pos = {num_iterations_list[max_pos]}")

      if use_validation[compound]:
        splits_to_iterate = ['train', 'cv', 'test']
      else:
        splits_to_iterate = ['train', 'test']
      for split_to_save in splits_to_iterate:
        for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
          metrics_to_save[compound][split_to_save][metric].append(iters_metrics_to_save[split_to_save][metric][max_pos])
      if use_validation[compound]:
        metrics_to_save[compound]['cv_classic']['R2'].append(iters_metrics_to_save['cv_classic']['R2'][-1])
        metrics_to_save[compound]['cv_classic']['RMSE'].append(iters_metrics_to_save['cv_classic']['RMSE'][-1])
        metrics_to_save[compound]['cv_classic']['MAE'].append(iters_metrics_to_save['cv_classic']['MAE'][-1])
        metrics_to_save[compound]['cv_classic']['MAPE'].append(iters_metrics_to_save['cv_classic']['MAPE'][-1])
      metrics_to_save[compound]['test_classic']['R2'].append(iters_metrics_to_save['test_classic']['R2'][-1])
      metrics_to_save[compound]['test_classic']['RMSE'].append(iters_metrics_to_save['test_classic']['RMSE'][-1])
      metrics_to_save[compound]['test_classic']['MAE'].append(iters_metrics_to_save['test_classic']['MAE'][-1])
      metrics_to_save[compound]['test_classic']['MAPE'].append(iters_metrics_to_save['test_classic']['MAPE'][-1])

    with open(f"{config_dict['root_dir']}/metrics_vals.txt", "a") as f:
      f.write(metrics_to_save.__str__() + "\n")
    print(metrics_to_save)
    return metrics_to_save
