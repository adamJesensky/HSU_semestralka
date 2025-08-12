import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

def convert_column_types(df):
    """
    Konvertuje stĺpce DataFrame na vhodné dátové typy.
    """
    # Iterácia cez všetky stĺpce v DataFrame
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except (ValueError, TypeError):
            pass  # or handle the column differently
    return df

def safe_remove_columns(df, columns_to_remove):
    """
    Bezpečne odstráni stĺpce z DataFrame a vráti informácie o odstránených stĺpcoch.
    """
    # Inicializácia zoznamov na sledovanie odstránených a nenájdených stĺpcov
    removed_columns = []
    not_found_columns = []
    # Vytvorenie kópie DataFrame, aby sa predišlo SettingWithCopyWarning
    df = df.copy()
    # Iterácia cez zoznam stĺpcov na odstránenie
    for column in columns_to_remove:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
            removed_columns.append(column)
        else:
            not_found_columns.append(column)
    
    # Voliteľne, vypísanie zoznamov odstránených a nenájdených stĺpcov
    if removed_columns:
        # print(f"Odstránené stĺpce: {removed_columns}")
        pass
    if not_found_columns:
        print(f"Stĺpce nenájdené a preto neodstránené: {not_found_columns}")
    
    return df

def clean_segment_data_with_columns(segments, columns_to_keep, min_price=5, min_floor_size=5, max_floor_size=100000, required_columns=None, verbose=True):
    """
    Vyčistí segmenty dát ponechaním len vybraných stĺpcov a odstránením neplatných záznamov.
    """
    if not segments:
        return {}
    
    if required_columns is None:
        required_columns = ['price', 'floor_size']
    
    segments_cleaned = {}
    
    for segment_name, segment_df in segments.items():
        if verbose:
            print(f"Čistenie segmentu: {segment_name} ({len(segment_df)})")
        
        # Vytvorenie kópie DataFrame, aby sa predišlo SettingWithCopyWarning
        segment_df = segment_df.copy()
        
        # Odstránenie stĺpcov, ktoré nie sú v zozname columns_to_keep
        columns_to_remove = [col for col in segment_df.columns if col not in columns_to_keep]
        segment_df = segment_df.drop(columns=columns_to_remove, errors='ignore')
        
        if verbose:
            print(f"Odstránené stĺpce zo segmentu {segment_name}: {len(columns_to_remove)}")
        
        # Konverzia dátových typov
        segment_df = convert_column_types(segment_df)
        
        # Odstránenie riadkov s neplatnými cenami a veľkosťami
        segment_df = segment_df[(segment_df['price'] > min_price) & 
                               (segment_df['floor_size'] > min_floor_size) & 
                               (segment_df['floor_size'] < max_floor_size)]
        
        # Odstránenie riadkov s chýbajúcimi hodnotami v požadovaných stĺpcoch
        valid_required_columns = [col for col in required_columns if col in segment_df.columns]
        if valid_required_columns:
            segment_df = segment_df.dropna(subset=valid_required_columns)
        
        segments_cleaned[segment_name] = segment_df
        
        if verbose:
            print(f"Segment {segment_name} vyčistený, počet riadkov: {len(segment_df)}")
    
    return segments_cleaned

def extract_and_create_binary_features(df, column_name, prefix):
    """
    Vytvorí binárne príznaky pre unikátne hodnoty v danom stĺpci.
    """
    if column_name in df.columns:
            all_values = set()
            # Rozdelenie záznamov čiarkami a aktualizácia množiny unikátnych atribútov
            for values in df[column_name].dropna().apply(lambda x: x.split(', ')):
                all_values.update(values)
            # Vytvorenie binárnych stĺpcov pre každý unikátny atribút
            for value in all_values:
                # Použitie metódy `apply` na vytvorenie binárnych stĺpcov
                df[f'{prefix}_{value}'] = df[column_name].apply(lambda x: 1 if value in str(x).split(', ') else 0)
    return df

def detect_and_handle_outliers(df, columns=None, method='z-score', z_score_threshold=3, iqr_threshold=1.5, action='remove'):
    """
    Detekuje a spracuje outliers v DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame s dátami na spracovanie
    columns : list, optional
        Zoznam stĺpcov na kontrolu outlierov. Ak None, skontrolujú sa všetky numerické stĺpce.
    method : str, optional
        Metóda na detekciu outlierov: 'z-score', 'iqr', 'percentile'
    z_score_threshold : float, optional
        Prah pre z-score metódu
    iqr_threshold : float, optional
        Prah pre IQR metódu
    action : str, optional
        Akcia na spracovanie outlierov: 'remove', 'cap', 'log', 'report'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame s ošetrenými outliermi
    dict
        Slovník s informáciami o detekovaných outlieroch
    """
    # Vytvorenie kópie DataFrame, aby sa predišlo SettingWithCopyWarning
    df = df.copy()
    outlier_info = {}
    
    # Ak nie sú zadané stĺpce, použijeme všetky numerické stĺpce
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        # Filtrovanie len existujúcich numerických stĺpcov
        columns = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    # Iterácia cez všetky stĺpce
    for column in columns:
        # Preskočenie stĺpcov s chýbajúcimi hodnotami
        if df[column].isnull().all():
            print(f"Stĺpec '{column}' obsahuje iba hodnoty NaN, preskakujem.")
            continue
        
        # Detekcia outlierov podľa zvolenej metódy
        if method == 'z-score':
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            outliers_mask = abs(z_scores) > z_score_threshold
            outliers = df[outliers_mask]
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_threshold * IQR
            upper_bound = Q3 + iqr_threshold * IQR
            outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers = df[outliers_mask]
        elif method == 'percentile':
            lower_bound = df[column].quantile(0.01)  # 1. percentil
            upper_bound = df[column].quantile(0.99)  # 99. percentil
            outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers = df[outliers_mask]
        
        # Uloženie informácií o outlieroch
        outlier_count = len(outliers)
        outlier_percent = outlier_count / len(df) * 100
        outlier_info[column] = {
            'count': outlier_count,
            'percent': outlier_percent,
            'min': outliers[column].min() if outlier_count > 0 else None,
            'max': outliers[column].max() if outlier_count > 0 else None
        }
        
        # Výpis informácií o outlieroch
        print(f"Stĺpec '{column}': {outlier_count} outlierov ({outlier_percent:.2f}%)")
        
        # Spracovanie outlierov podľa zvolenej akcie
        if action == 'remove' and outlier_count > 0:
            df = df[~outliers_mask]
            print(f"  - Odstránených {outlier_count} outlierov zo stĺpca '{column}'")
        elif action == 'cap' and outlier_count > 0:
            if method == 'z-score':
                # Pre z-score použijeme mean +/- threshold * std
                mean = df[column].mean()
                std = df[column].std()
                lower_cap = mean - z_score_threshold * std
                upper_cap = mean + z_score_threshold * std
            elif method == 'iqr':
                # Pre IQR použijeme Q1 - threshold * IQR a Q3 + threshold * IQR
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_cap = Q1 - iqr_threshold * IQR
                upper_cap = Q3 + iqr_threshold * IQR
            elif method == 'percentile':
                # Pre percentile použijeme 1. a 99. percentil
                lower_cap = df[column].quantile(0.01)
                upper_cap = df[column].quantile(0.99)
            
            # Aplikácia capping
            df.loc[df[column] < lower_cap, column] = lower_cap
            df.loc[df[column] > upper_cap, column] = upper_cap
            print(f"  - Outliers v stĺpci '{column}' ohraničené na [{lower_cap:.2f}, {upper_cap:.2f}]")
        elif action == 'log' and outlier_count > 0:
            # Logaritmická transformácia (len pre pozitívne hodnoty)
            if df[column].min() > 0:
                df[f"{column}_log"] = np.log1p(df[column])
                print(f"  - Vytvorený nový stĺpec '{column}_log' s logaritmickou transformáciou")
            else:
                print(f"  - Logaritmická transformácia nie je možná pre stĺpec '{column}' (obsahuje nulové alebo záporné hodnoty)")
    
    return df, outlier_info

def normalize_data(df, columns=None, method='min-max', new_range=(0, 1)):
    """
    Normalizuje numerické stĺpce v DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame s dátami na normalizáciu
    columns : list, optional
        Zoznam stĺpcov na normalizáciu. Ak None, normalizujú sa všetky numerické stĺpce.
    method : str, optional
        Metóda normalizácie: 'min-max', 'z-score', 'robust'
    new_range : tuple, optional
        Nový rozsah pre min-max normalizáciu
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame s normalizovanými dátami
    dict
        Slovník s parametrami normalizácie (pre neskoršie použitie pri predikciách)
    """
    # Vytvorenie kópie DataFrame, aby sa predišlo SettingWithCopyWarning
    df = df.copy()
    normalization_params = {}
    
    # Ak nie sú zadané stĺpce, použijeme všetky numerické stĺpce
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        # Filtrovanie len existujúcich numerických stĺpcov
        columns = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    # Iterácia cez všetky stĺpce
    for column in columns:
        # Preskočenie stĺpcov s chýbajúcimi hodnotami
        if df[column].isnull().all():
            print(f"Stĺpec '{column}' obsahuje iba hodnoty NaN, preskakujem normalizáciu.")
            continue
        
        # Normalizácia podľa zvolenej metódy
        if method == 'min-max':
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val > min_val:  # Zabránenie deleniu nulou
                df[f"{column}_norm"] = (df[column] - min_val) / (max_val - min_val) * (new_range[1] - new_range[0]) + new_range[0]
                normalization_params[column] = {'method': 'min-max', 'min': min_val, 'max': max_val, 'new_range': new_range}
                print(f"Stĺpec '{column}' normalizovaný metódou min-max na rozsah {new_range}")
            else:
                print(f"Stĺpec '{column}' má rovnaké min a max hodnoty, preskakujem normalizáciu.")
        
        elif method == 'z-score':
            mean = df[column].mean()
            std = df[column].std()
            if std > 0:  # Zabránenie deleniu nulou
                df[f"{column}_norm"] = (df[column] - mean) / std
                normalization_params[column] = {'method': 'z-score', 'mean': mean, 'std': std}
                print(f"Stĺpec '{column}' normalizovaný metódou z-score")
            else:
                print(f"Stĺpec '{column}' má nulovú štandardnú odchýlku, preskakujem normalizáciu.")
        
        elif method == 'robust':
            median = df[column].median()
            iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
            if iqr > 0:  # Zabránenie deleniu nulou
                df[f"{column}_norm"] = (df[column] - median) / iqr
                normalization_params[column] = {'method': 'robust', 'median': median, 'iqr': iqr}
                print(f"Stĺpec '{column}' normalizovaný robustnou metódou")
            else:
                print(f"Stĺpec '{column}' má nulové IQR, preskakujem normalizáciu.")
    
    return df, normalization_params

def merge_segments(segments, create_combined=True, create_super=True):
    """
    Zlúči segmenty dát do combined segmentov a/alebo super segmentu so zachovaním relevantných stĺpcov.
    
    Parameters:
    -----------
    segments : dict
        Slovník so segmentami dát, kde kľúč je názov segmentu a hodnota je DataFrame
    create_combined : bool, optional
        Či sa majú vytvoriť combined segmenty pre každý typ nehnuteľnosti
    create_super : bool, optional
        Či sa má vytvoriť super segment obsahujúci všetky dáta
    
    Returns:
    --------
    dict
        Slovník obsahujúci pôvodné segmenty a nové combined/super segmenty
    """
    if not segments:
        return segments
    
    # Vytvorenie kópie slovníka segmentov
    updated_segments = segments.copy()
    
    # Identifikácia typov nehnuteľností a transakcií
    property_types = set()
    transaction_types = set()
    
    for segment_name in segments.keys():
        # Extrakcia typu nehnuteľnosti a transakcie z názvu segmentu
        parts = segment_name.split('_')
        if len(parts) >= 2 and 'Combined' not in parts[0]:
            property_types.add(parts[0])
        if len(parts) >= 2 and 'Combined' not in parts[1]:
            transaction_types.add(parts[1])
    
    # Vytvorenie combined segmentov pre každý typ nehnuteľnosti
    if create_combined:
        for property_type in property_types:
            # Nájdenie všetkých segmentov pre daný typ nehnuteľnosti
            property_segments = {name: df for name, df in segments.items() 
                               if name.startswith(property_type + '_') and 'Combined' not in name}
            
            # if len(property_segments) > 1:  # Vytvoríme combined segment len ak existuje viac ako jeden segment
            #     # Spojenie všetkých dataframov do jedného
            #     combined_df = pd.concat(property_segments.values(), ignore_index=True)
                
            #     # Pridanie combined segmentu do slovníka
            #     combined_name = f"{property_type}_Combined"
            #     updated_segments[combined_name] = combined_df
            #     print(f"Vytvorený combined segment '{combined_name}' s {len(combined_df)} záznamami")
                
              
            if len(property_segments) > 1:
                combined_dfs = []
                for name, df in property_segments.items():
                    tx_type = name.split('_')[1] if len(name.split('_')) > 1 else 'Unknown'
                    df_copy = df.copy()
                    df_copy['transaction_type'] = tx_type
                    combined_dfs.append(df_copy)

                combined_df = pd.concat(combined_dfs, ignore_index=True)
                combined_name = f"{property_type}_Combined"
                updated_segments[combined_name] = combined_df
                print(f"Vytvorený combined segment '{combined_name}' s {len(combined_df)} záznamami")

    
    # Vytvorenie super segmentu obsahujúceho všetky dáta
    if create_super:    
        # Vylúčenie už existujúcich combined segmentov a super segmentu
        original_segments = {name: df for name, df in segments.items() 
                           if 'Combined' not in name and name != 'Super_Segment'}

      
        # if original_segments:  # Vytvoríme super segment len ak existujú nejaké pôvodné segmenty
        #     # Spojenie všetkých dataframov do jedného
        #     super_df = pd.concat(original_segments.values(), ignore_index=True)

        #     # Pridanie super segmentu do slovníka
        #     updated_segments["Super_Segment"] = super_df
        #     print(f"Vytvorený Super_Segment s {len(super_df)} záznamami")
        if original_segments:
            super_dfs = []
            for name, df in original_segments.items():
                tx_type = name.split('_')[1] if len(name.split('_')) > 1 else 'Unknown'
                df_copy = df.copy()
                df_copy['transaction_type'] = tx_type
                super_dfs.append(df_copy)

            super_df = pd.concat(super_dfs, ignore_index=True)
            updated_segments["Super_Segment"] = super_df
            print(f"Vytvorený Super_Segment s {len(super_df)} záznamami")
    return updated_segments

def save_segments_to_csv(segments, output_dir='./processed_data', prefix=''):
    """
    Uloží segmenty dát do CSV súborov.
    
    Parameters:
    -----------
    segments : dict
        Slovník segmentov dát na uloženie
    output_dir : str, optional
        Adresár, do ktorého sa majú uložiť CSV súbory (default: './processed_data')
    prefix : str, optional
        Prefix pre názvy súborov (default: '')
    
    Returns:
    --------
    dict
        Slovník s cestami k uloženým súborom
    """
    if not segments:
        print("Žiadne segmenty na uloženie.")
        return {}
    
    # Vytvorenie adresára, ak neexistuje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Vytvorený adresár: {output_dir}")
    
    saved_files = {}
    
    # Uloženie každého segmentu do samostatného CSV súboru
    for segment_name, segment_df in segments.items():
        file_name = f"{prefix}{segment_name}.csv" if prefix else f"{segment_name}.csv"
        file_path = os.path.join(output_dir, file_name)
        
        # Uloženie do CSV
        segment_df.to_csv(file_path, index=False)
        saved_files[segment_name] = file_path
        print(f"Segment {segment_name} uložený do {file_path} ({len(segment_df)} záznamov)")
    
    return saved_files
