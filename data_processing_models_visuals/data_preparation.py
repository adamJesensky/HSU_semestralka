import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Funkcie na prípravu dát pre model

def segment_data(df, allowed_property_types=None, allowed_transaction_types=None):
    """
    Segmentuje dáta podľa typu nehnuteľnosti a typu transakcie, s možnosťou filtrovať podľa allowed_property_types a allowed_transaction_types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame s dátami na segmentáciu
    allowed_property_types : str, list, optional
        None, hodnota alebo zoznam hodnôt druh_objektu, ktoré sa majú segmentovať
    allowed_transaction_types : str, list, optional
        None, hodnota alebo zoznam hodnôt typ transakcie, ktoré sa majú segmentovať
    
    Returns:
    --------
    dict
        Slovník so segmentami dát
    """
    segments = {}
    
    # Filtrovanie podľa typu nehnuteľnosti
    if allowed_property_types is not None:
        if isinstance(allowed_property_types, str):
            allowed_property_types = [allowed_property_types]
        df = df[df['druh_objektu'].isin(allowed_property_types)]
    
    # Filtrovanie podľa typu transakcie
    if allowed_transaction_types is not None:
        if isinstance(allowed_transaction_types, str):
            allowed_transaction_types = [allowed_transaction_types]
        df = df[df['typ'].isin(allowed_transaction_types)]
    
    # Vytvorenie segmentov pre každú kombináciu typu nehnuteľnosti a transakcie
    for property_type in df['druh_objektu'].unique():
        for transaction_type in df['typ'].unique():
            segment_name = f"{property_type}_{transaction_type}"
            segment_data = df[(df['druh_objektu'] == property_type) & (df['typ'] == transaction_type)]
            if len(segment_data) > 0:
                segments[segment_name] = segment_data
                print(f"{segment_name}: {len(segment_data)} záznamov")
    
    # Vytvorenie Combined segmentov len ak existuje viac ako jeden typ transakcie pre daný typ nehnuteľnosti
    unique_transaction_types = df['typ'].unique()
    if len(unique_transaction_types) > 1:
        for property_type in df['druh_objektu'].unique():
            combined_data = df[df['druh_objektu'] == property_type]
            if len(combined_data) > 0:
                segments[f"{property_type}_Combined"] = combined_data
                print(f"{property_type}_Combined: {len(combined_data)} záznamov")
    return segments


def normalize_data_for_nn(df, feature_columns=None, target_column=None, test_size=0.2, random_state=42):
    """
    Normalizuje a pripraví dáta pre trénovanie neurónovej siete bez použitia sklearn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame s dátami na prípravu
    feature_columns : list, optional
        Zoznam stĺpcov, ktoré sa majú použiť ako vstupné príznaky. Ak None, použijú sa všetky numerické stĺpce okrem cieľového.
    target_column : str, optional
        Názov cieľového stĺpca. Ak None, automaticky sa určí na základe názvu segmentu.
    test_size : float, optional
        Podiel testovacích dát (default: 0.2)
    random_state : int, optional
        Seed pre náhodné rozdelenie dát (default: 42)
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_info)
        kde feature_info obsahuje informácie o normalizácii pre každý stĺpec
    """
    df = df.copy()
    
    # Automatické určenie cieľového stĺpca, ak nie je zadaný
    if target_column is None:
        target_column = 'price' # Predvolene používame 'price'
    
    # Kontrola, či cieľový stĺpec existuje
    if target_column not in df.columns:
        raise ValueError(f"Cieľový stĺpec '{target_column}' sa nenachádza v DataFrame.")
    
    # Poznámka: Logaritmická transformácia ceny už nie je potrebná, používame priamo hodnoty 'price'
    
    # Identifikácia numerických stĺpcov, ak nie sú zadané
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_columns = [col for col in feature_columns if col != target_column and col != 'price']
    
    # Kontrola, či všetky zadané stĺpce existujú
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Stĺpec '{col}' sa nenachádza v DataFrame.")
    
    # Rozdelenie stĺpcov na numerické a kategorické
    numeric_features = [col for col in feature_columns if df[col].dtype in [np.int64, np.float64]]
    categorical_features = [col for col in feature_columns if col not in numeric_features]
    
    # Spracovanie kategorických premenných pomocou one-hot encoding
    df_encoded = df.copy()
    categorical_mapping = {}
    
    for col in categorical_features:
        # Kontrola, či stĺpec obsahuje hodnoty NaN a ich nahradenie
        if df_encoded[col].isna().any():
            df_encoded[col] = df_encoded[col].fillna('Neznáme')
        
        # Konverzia na string pre zabezpečenie správneho spracovania
        df_encoded[col] = df_encoded[col].astype(str)
        
        # Obmedzenie počtu unikátnych hodnôt pre veľké kategorické stĺpce
        if len(df_encoded[col].unique()) > 100:
            # Ponechanie len top 100 najčastejších hodnôt
            top_values = df_encoded[col].value_counts().nlargest(100).index.tolist()
            df_encoded[col] = df_encoded[col].apply(lambda x: x if x in top_values else 'Ostatné')
            print(f"Stĺpec '{col}' má príliš veľa unikátnych hodnôt. Obmedzené na top 100 + 'Ostatné'.")
        
        try:
            # Vytvorenie one-hot encoding pre kategorické stĺpce
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            categorical_mapping[col] = list(dummies.columns)
            
            # Odstránenie pôvodného kategorického stĺpca po vytvorení one-hot encoded stĺpcov
            df_encoded = df_encoded.drop(columns=[col])
        except Exception as e:
            print(f"Chyba pri spracovaní stĺpca '{col}': {str(e)}")
            # Odstránenie problematického stĺpca
            df_encoded = df_encoded.drop(columns=[col])
            print(f"Stĺpec '{col}' bol odstránený z dátovej sady.")
    
    # Aktualizácia zoznamu príznakov po one-hot encodingu
    encoded_features = numeric_features.copy()
    for col, encoded_cols in categorical_mapping.items():
        encoded_features.extend(encoded_cols)
        
    # Kontrola, či všetky stĺpce v encoded_features existujú v df_encoded
    encoded_features = [col for col in encoded_features if col in df_encoded.columns]
    
    # Normalizácia numerických stĺpcov
    feature_info = {}
    for col in numeric_features:
        if col in df_encoded.columns:  # Kontrola, či stĺpec stále existuje
            # Kontrola, či stĺpec obsahuje objekty namiesto čísel
            if df_encoded[col].dtype == 'object':
                try:
                    # Pokus o konverziu na numerický typ
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                    # Nahradenie NaN hodnôt mediánom
                    df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                    print(f"Stĺpec '{col}' bol konvertovaný z objektu na numerický typ.")
                except Exception as e:
                    print(f"Chyba pri konverzii stĺpca '{col}': {str(e)}")
                    # Odstránenie problematického stĺpca
                    df_encoded = df_encoded.drop(columns=[col])
                    print(f"Stĺpec '{col}' bol odstránený z dátovej sady.")
                    continue
            
            # Výpočet priemeru a smerodajnej odchýlky pre normalizáciu
            mean_val = df_encoded[col].mean()
            std_val = df_encoded[col].std()
            if std_val == 0:  # Zabránenie deleniu nulou
                std_val = 1
            
            # Normalizácia stĺpca
            df_encoded[col] = (df_encoded[col] - mean_val) / std_val
            
            # Uloženie informácií o normalizácii
            feature_info[col] = {'mean': mean_val, 'std': std_val, 'type': 'numeric'}
    
    # Uloženie informácií o kategorických príznakoch
    for col in categorical_features:
        feature_info[col] = {'mapping': categorical_mapping[col], 'type': 'categorical'}
    
    # Uloženie informácií o cieľovej premennej
    feature_info['target'] = {
        'column': target_column,
        'is_log_transformed': False  # Už nepoužívame logaritmickú transformáciu
    }
    
    # Kontrola, či všetky stĺpce v encoded_features existujú v df_encoded
    valid_encoded_features = [col for col in encoded_features if col in df_encoded.columns]
    
    # Kontrola, či máme dostatok príznakov na trénovanie
    if len(valid_encoded_features) == 0:
        raise ValueError("Po spracovaní nezostali žiadne platné príznaky pre trénovanie modelu.")
    
    # Vytvorenie X a y
    X = df_encoded[valid_encoded_features]
    y = df_encoded[target_column]
    
    # Kontrola, či X obsahuje len numerické hodnoty
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
            except Exception as e:
                print(f"Chyba pri konverzii stĺpca '{col}' v X: {str(e)}")
                # Odstránenie problematického stĺpca
                X = X.drop(columns=[col])
                print(f"Stĺpec '{col}' bol odstránený z X.")
    
    # Kontrola, či y obsahuje len numerické hodnoty
    if y.dtype == 'object':
        try:
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(y.median())
        except Exception as e:
            raise ValueError(f"Cieľový stĺpec '{target_column}' obsahuje nenumerické hodnoty: {str(e)}")
    
    # Rozdelenie na trénovacie a testovacie dáta
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size_idx = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size_idx], indices[test_size_idx:]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Konverzia na numpy array pre kompatibilitu s TensorFlow/Keras
    X_train = X_train.to_numpy().astype('float32')
    X_test = X_test.to_numpy().astype('float32')
    y_train = y_train.to_numpy().astype('float32')
    y_test = y_test.to_numpy().astype('float32')
    
    return X_train, X_test, y_train, y_test, feature_info, test_idx


def prepare_data_for_model(df, target_column='price', test_size=0.2, random_state=42):
    """
    Pripraví dáta pre trénovanie modelu.
    
    Poznámka: Táto funkcia používa sklearn, ktorý nie je podporovaný v Python 3.13.2.
    Pre novšie verzie Pythonu použite funkciu normalize_data_for_nn.
    """
    # Kontrola, či cieľový stĺpec existuje
    if target_column not in df.columns:
        raise ValueError(f"Cieľový stĺpec '{target_column}' sa nenachádza v DataFrame.")
    
    # Identifikácia numerických a kategorických stĺpcov
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col != target_column]
    
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Vytvorenie X a y
    X = df[numeric_features + categorical_features]
    y = df[target_column]
    
    # Rozdelenie na trénovacie a testovacie dáta
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Vytvorenie preprocessingu
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Aplikácia preprocessingu
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, numeric_features, categorical_features

def save_processed_data(segments, output_dir='./processed_data'):
    """
    Uloží spracované segmenty dát do CSV súborov.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for segment_name, segment_data in segments.items():
        output_path = f"{output_dir}/{segment_name}_processed.csv"
        segment_data.to_csv(output_path, index=False)
        print(f"Segment {segment_name} uložený do {output_path}")

# Príklad použitia
if __name__ == "__main__":
    # Načítanie vyčistených dát (príklad)
    # from data_cleaning import clean_data
    # df = pd.read_csv('path_to_your_data.csv')
    # df = clean_data(df)
    # 
    # # Segmentácia dát
    # segments = segment_data(df)
    # 
    # # Feature engineering pre každý segment
    # for segment_name, segment_df in segments.items():
    #     segments[segment_name] = create_feature_engineering(segment_df)
    #     
    #     # Príklad použitia normalize_data_for_nn
    #     # Vybrané stĺpce pre model
    #     # selected_features = ['floor_size', 'room_count', 'is_new']
    #     # 
    #     # # Automatické určenie cieľového stĺpca na základe názvu segmentu
    #     # target = None  # Automaticky zvolí 'price' alebo 'log_price' podľa názvu segmentu
    #     # 
    #     # # Alebo explicitné určenie cieľového stĺpca
    #     # # target = 'price' if 'Combined' not in segment_name else 'log_price'
    #     # 
    #     # X_train, X_test, y_train, y_test, feature_info = normalize_data_for_nn(
    #     #     segment_df, 
    #     #     feature_columns=selected_features,
    #     #     target_column=target
    #     # )
    #     # 
    #     # # feature_info obsahuje informácie o normalizácii pre každý stĺpec
    #     # # Tieto informácie môžu byť použité neskôr pri predikciách na nových dátach
    # 
    # # Uloženie spracovaných dát
    # save_processed_data(segments)
    print("Modul pre prípravu dát je pripravený na použitie.")