import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from datetime import datetime

# Import vlasntnych modulov
from data_loader import process_data, visualize_data_distribution, load_processed_data
from data_cleaning import merge_segments, convert_column_types, clean_segment_data_with_columns, detect_and_handle_outliers, safe_remove_columns, extract_and_create_binary_features, save_segments_to_csv
from data_preparation import normalize_data_for_nn, segment_data, save_processed_data
from neural_network import NeuralNetwork, train_model_for_segment, visualize_predictions

# Create new features based on categorical analysis
def create_features_from_categories(df):
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # 1. Create room count feature from druh_objektu_text
    room_mapping = {
        'Garsónka': 0,
        '1 izbový byt': 1,
        '2 izbový byt': 2,
        '3 izbový byt': 3,
        '4 izbový byt': 4,
        '5 a viac izbový byt': 5,
        'Mezonet': 3,  # Assumption: average mezonet has 3 rooms
        'Apartmán': 2,  # Assumption: average apartment has 2 rooms
        'Dvojgarsónka': 1,      # Assumption: slightly larger than garzónka
        'Iný byt': 2,           # Assumption: neutral default
        'Loft': 2               # Assumption: open space with 1–2 rooms
    }
    
    df_new['room_count'] = df_new['druh_objektu_text'].map(room_mapping)
    
    # 2. Create property condition score (higher = better condition)
    condition_mapping = {
        'Pôvodný stav': 1,
        'Čiastočná rekonštrukcia': 2,
        'Kompletná rekonštrukcia': 3,
        'Vo výstavbe': 4,
        'Developerský projekt': 4,
        'Novostavba': 5
    }
    
    df_new['condition_score'] = df_new['Stav nehnuteľnosti'].map(condition_mapping)
    
    # 3. Create binary features for property condition
    df_new['is_new'] = df_new['Stav nehnuteľnosti'].apply(
        lambda x: 1 if x in ['Novostavba', 'Vo výstavbe', 'Developerský projekt'] else 0
    )
    df_new['is_renovated'] = df_new['Stav nehnuteľnosti'].apply(
        lambda x: 1 if x in ['Kompletná rekonštrukcia', 'Čiastočná rekonštrukcia'] else 0
    )
    return df_new

# Funkcia na spracovanie a čistenie dát
def process_and_clean_data():
    """Spracuje a vyčistí dáta z pôvodných zdrojov a uloží ich do CSV súborov."""
    
    # segments = process_data(regions_list=[], 
        #                     segment_filter={'property_types': ['Byty', 'Domy'],'transaction_types': ['Predaj', 'Prenájom']},
        #                     clean_outliers = False)
    segments = process_data(regions_list=[], 
                            segment_filter={'property_types': ['Byty'],'transaction_types': ['Predaj', 'Prenájom']},
                            clean_outliers = False)
    
    # vytvorenie super segmentu (combined segment)
    segments = merge_segments(segments, True, False)
    print(segments.keys())
    print("--------------------------------------------------")

    
    # Vizualizácia distribúcie dát pre všetky segmenty
    # print("\nVytváranie vizualizácií distribúcie dát...")
    # visualize_data_distribution(segments)               
    # print("\nVizualizácia dát bola úspešne dokončená!")
    # print("--------------------------------------------------")
    
    
    # Výpis informácií o chýbajúcich hodnotách v stlpcoch
    # missing_info = segments['Byty_Combined'].isnull().sum()
    # missing_info = missing_info[missing_info >= 0].sort_values()
    # if len(missing_info) >= 0:
    #     print("Chýbajúce hodnoty v stĺpcoch:")
    #     for col, count in missing_info.items():
    #         percent = count / len(segments['Byty_Combined']) * 100
    #         print(f"{col}: {count} ({percent:.2f}%)")
    
    
    # Cistenie dat
    print("\nCistenie dát...")
    # Identifikácia stĺpcov, ktoré sa majú ponechať
    columns_to_keep = ['price','transaction_type', 'floor_size', 'latitude', 'longitude', 'Cena vrátane energií', 'addressRegion',
                       'streetAddress', 'druh_objektu_text', 'updated_date', 'publication_date',
                        'Stav nehnuteľnosti', 'Vlastníctvo', 'Počet izieb / miestností', 
                        'Podlažie', 'Počet nadzemných podlaží', 'Vybavenie', 'Typ konštrukcie', 'Rok výstavby']
    
    # Požadované stĺpce, ktoré musia mať platné hodnoty
    required_columns = ['price', 'floor_size', 'Cena vrátane energií', 'addressRegion', 'streetAddress', 'Stav nehnuteľnosti']
    # Použitie novej funkcie na čistenie segmentov
    segments_cleaned = clean_segment_data_with_columns(
        segments=segments,
        columns_to_keep=columns_to_keep,
        min_price=5,
        min_floor_size=5,
        max_floor_size=100000,
        required_columns=required_columns,
        verbose=True
    )
    for segment_name, segment_df in segments_cleaned.items():
        print(f"\nPríprava segmentu: {segment_name}")
        # Správne spracovanie výsledku z detect_and_handle_outliers, ktorý vracia tuple (DataFrame, outlier_info)
        cleaned_df, outlier_info = detect_and_handle_outliers(df=segment_df, columns=['price', 'floor_size'],  
                                                           method='z-score', action='remove')
        # Uloženie vyčisteného DataFrame späť do slovníka
        segments_cleaned[segment_name] = cleaned_df
    print("--------------------------------------------------")


    # Adding Features and preprocess data
    print("\nPríprava dát...")
    # Vytvorenie nových príznakov
    for segment_name, segment_df in segments_cleaned.items():
        # print(f"\nPríprava segmentu: {segment_name}")
        # Vytvorenie kópie DataFrame, aby sa predišlo SettingWithCopyWarning
        df_copy = segment_df.copy()
        
        # Vytvorenie nových stĺpcov
        if 'publication_date' in df_copy.columns:
            df_copy['publication_date'] = pd.to_datetime(df_copy['publication_date'], errors='coerce')
            df_copy['days_since_publication'] = (datetime.now() - df_copy['publication_date']).dt.days
        if 'updated_date' in df_copy.columns:
            df_copy['updated_date'] = pd.to_datetime(df_copy['updated_date'], errors='coerce')
            df_copy['update_publication_delta'] = (df_copy['updated_date'] - df_copy['publication_date']).dt.days
        if 'Vybavenie' in df_copy.columns:
            df_copy = extract_and_create_binary_features(df_copy, 'Vybavenie', 'Vybavenie')
            
        # Odstránenie pôvodných dátumových stĺpcov
        columns_to_drop = ['publication_date', 'updated_date', 'Vybavenie']
        if columns_to_drop:
            df_copy = safe_remove_columns(df_copy, columns_to_drop)
            
        df_copy = create_features_from_categories(df_copy)

        # Vytvorenie binárneho príznaku is_for_sale z transaction_type
        if 'transaction_type' in df_copy.columns:
            df_copy['is_for_sale'] = df_copy['transaction_type'].apply(lambda x: 1 if x == 'Predaj' else 0)
            # Odstránenie pôvodného stĺpca transaction_type, ak už nie je potrebný
            df_copy = safe_remove_columns(df_copy, ['transaction_type'])
        
        # Uloženie upraveného DataFrame späť do slovníka segments_cleaned
        segments_cleaned[segment_name] = df_copy
        
        # print("New features created:")
        # new_features = ['room_count', 'condition_score', 'is_new', 'is_renovated']
        # print(segments_cleaned[segment_name][new_features].head())
    print("--------------------------------------------------")

    print(segments_cleaned['Byty_Combined'].columns)
    print("--------------------------------------------------")

    # Uloženie vyčistených segmentov do CSV súborov
    print("\nUkladanie vyčistených segmentov do CSV súborov...")
    output_dir = os.path.join(os.path.dirname(__file__), 'processed_data')
    saved_files = save_segments_to_csv(segments_cleaned, output_dir=output_dir)
    print("\nVyčistené segmenty boli úspešne uložené do CSV súborov!")
    print("--------------------------------------------------")
    
    return segments_cleaned

# Hlavná funkcia
def main(segment_names=None, target_column='price', feature_columns=None, model_params=None): # Changed log_price to price
    """Hlavná funkcia programu - načíta vyčistené dáta alebo ich spracuje, ak neexistujú.
    
    Parameters:
    -----------
    segment_names : list, optional
        Zoznam názvov segmentov na trénovanie. Ak None, použijú sa všetky segmenty.
    target_column : str, optional
        Názov cieľového stĺpca ('price'). Predvolene 'price'. # Removed log_price
    feature_columns : list, optional
        Zoznam stĺpcov, ktoré sa majú použiť ako vstupné príznaky. Ak None, použije sa ['floor_size'].
    apply_feature_engineering : bool, optional
        Ak True, aplikuje sa feature engineering na dáta pred tréningom.
    model_params : dict, optional
        Parametre pre trénovanie modelu (hidden_layers, learning_rate, activation, epochs, batch_size).
        Ak None, použijú sa predvolené hodnoty.
    
    Returns:
    --------
    tuple
        (segments_cleaned, models, metrics) - upravené segmenty, natrénované modely a ich metriky
    """
    # Skúsime načítať už vyčistené dáta
    print("Pokus o načítanie vyčistených dát...")
    segments_cleaned = load_processed_data()

    # Kontrola, či načítané dáta obsahujú všetky potrebné stĺpce (napr. 'is_for_sale')
    # Ak nie, vynútime opätovné spracovanie.
    if segments_cleaned is not None and feature_columns is not None and 'is_for_sale' in feature_columns:
        force_reprocess = False
        for segment_name, segment_df in segments_cleaned.items():
            # Kontrolujeme len pre segmenty, ktore budu trenovane, alebo pre vsetky ak segment_names is None
            if segment_names is None or segment_name in segment_names:
                if 'is_for_sale' not in segment_df.columns:
                    print(f"Stĺpec 'is_for_sale' chýba v načítanom segmente {segment_name}. Dáta budú pregenerované.")
                    force_reprocess = True
                    break
        if force_reprocess:
            segments_cleaned = None # Vynúti volanie process_and_clean_data()
    
    # Ak vyčistené dáta neexistujú alebo bolo vynútené pregenerovanie, spracujeme a vyčistíme pôvodné dáta
    if segments_cleaned is None:
        print("Vyčistené dáta neboli nájdené. Spúšťam proces spracovania a čistenia dát...")
        segments_cleaned = process_and_clean_data()
    else:
        print("Vyčistené dáta boli úspešne načítané!")
        print("--------------------------------------------------")
        
        # Výpis informácií o načítaných segmentoch
        for segment_name, segment_df in segments_cleaned.items():
            print(f"Segment {segment_name}: {len(segment_df)} záznamov, {len(segment_df.columns)} stĺpcov")
        print("--------------------------------------------------")
    
    # Vizualizácia vyčistených dát
    # print("\nVytváranie vizualizácií distribúcie dát...")
    # visualize_data_distribution(segments_cleaned)
    # print("\nVizualizácia dát bola úspešne dokončená!")
    # print("--------------------------------------------------")
    
    # Trénovanie modelov pre vybrané segmenty
    segments_cleaned, models, metrics = train_selected_segments(
        segments_cleaned,
        segment_names=segment_names,
        target_column=target_column,
        feature_columns=feature_columns,
        model_params=model_params
    )
    
    return segments_cleaned, models, metrics

# Funkcia na trénovanie modelov pre vybrané segmenty
def train_selected_segments(segments, segment_names=None, target_column=None, feature_columns=None, model_params=None):
    """
    Trénuje neurónové siete pre vybrané segmenty s možnosťou konfigurácie parametrov.
    
    Parameters:
    -----------
    segments : dict
        Slovník segmentov dát (DataFrame) na trénovanie
    segment_names : list, optional
        Zoznam názvov segmentov na trénovanie. Ak None, použijú sa všetky segmenty.
    target_column : str, optional
        Názov cieľového stĺpca ('price'). Ak None, automaticky sa určí. # Removed log_price
    feature_columns : list, optional
        Zoznam stĺpcov, ktoré sa majú použiť ako vstupné príznaky. Ak None, použije sa ['floor_size'].
    model_params : dict, optional
        Parametre pre trénovanie modelu (hidden_layers, learning_rate, activation, epochs, batch_size).
        Ak None, použijú sa predvolené hodnoty.
    
    Returns:
    --------
    tuple
        (segments, models, metrics) - upravené segmenty, natrénované modely a ich metriky
    """
    # Inicializácia predvolených hodnôt
    if segment_names is None:
        segment_names = list(segments.keys())
    if feature_columns is None:
        feature_columns = ['floor_size']
    if model_params is None:
        model_params = {
            'hidden_layers': [128, 64, 32, 16],
            'learning_rate': 0.001,
            'activation': 'relu',
            'epochs': 100,
            'batch_size': 32,
            'dropout_rate': 0.2,
            'l2_reg': 0.001,
            'use_batch_norm': True,
            'patience': 15
        }
    
    # Kontrola, či zadané segmenty existujú
    valid_segments = {name: segments[name] for name in segment_names if name in segments}
    if not valid_segments:
        print("Neboli nájdené žiadne platné segmenty na trénovanie.")
        return segments, {}, {}
    
    # Poznámka: Logaritmická transformácia ceny už nie je potrebná, používame priamo hodnoty 'price'
    
    # Inicializácia slovníkov pre modely a metriky
    models = {}
    metrics = {}
    
    # Normalizácia dát a trénovanie modelov
    print("\nTrénovanie neurónových sietí pre vybrané segmenty...")
    for segment_name in valid_segments.keys():
        print(f"\nSpracovanie segmentu: {segment_name}")
        
        # Získanie DataFrame pre aktuálny segment
        segment_df = segments[segment_name]
        
        # Automatické určenie cieľového stĺpca, ak nie je zadaný
        current_target = target_column
        if current_target is None:
            current_target = 'price' # Default to 'price'
        
        # Kontrola, či cieľový stĺpec existuje
        if current_target not in segment_df.columns:
            print(f"Varovanie: Cieľový stĺpec '{current_target}' sa nenachádza v segmente {segment_name}. Preskakujem.")
            continue
        
        print(f"Cieľový stĺpec: {current_target}")
        # Dynamická úprava príznakov pre kombinované segmenty
        current_features = feature_columns.copy() if feature_columns else ['floor_size']
        if segment_name.endswith('_Combined'):
            if 'is_for_sale' not in current_features:
                current_features.append('is_for_sale')
        else:
            if 'is_for_sale' in current_features:
                current_features.remove('is_for_sale')
        # Ak sa rozhodneme ponechať transaction_type pre one-hot encoding aj popri is_for_sale:
        # if segment_name.endswith('_Combined') and 'transaction_type' not in current_features:
        #     current_features.append('transaction_type')
        print(f"Vstupné príznaky: {', '.join(current_features)}")
 
        # Normalizácia dát
        try:
            # Store original 'is_for_sale' for combined segments before normalization
            is_for_sale_test_series = None
            if 'Combined' in segment_name and 'is_for_sale' in segment_df.columns:
                # We need to get the test indices first to correctly select the 'is_for_sale' for the test set
                # This will be done after splitting in normalize_data_for_nn
                pass # Placeholder, will be handled after normalize_data_for_nn call

            X_train, X_test, y_train, y_test, feature_info, test_idx = normalize_data_for_nn(
                segment_df, 
                feature_columns=current_features,
                target_column=current_target
            )

            if 'Combined' in segment_name and 'is_for_sale' in segment_df.columns:
                is_for_sale_test_series = segment_df.iloc[test_idx]['is_for_sale']
            
            print(f"Trénovacia sada: {X_train.shape[0]} vzoriek, {X_train.shape[1]} príznakov")
            print(f"Testovacia sada: {X_test.shape[0]} vzoriek, {X_test.shape[1]} príznakov")
            print(f"Normalizované príznaky: {', '.join(feature_info.keys())}")
            
            # Rozdelenie trénovacích dát na tréningovú a validačnú množinu
            from sklearn.model_selection import train_test_split
            if X_train.shape[0] > 1: # Potrebujeme aspoň 2 vzorky na rozdelenie
                X_train_actual, X_val, y_train_actual, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42 # 20% pre validáciu
                )
            else:
                # Ak máme len jednu vzorku, nemôžeme vytvoriť validačnú sadu
                X_train_actual, y_train_actual = X_train, y_train
                X_val, y_val = None, None

            # Trénovanie modelu
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            
            # Príprava konfiguračného slovníka pre train_model_for_segment
            config = {
                'model_params': {
                    'hidden_layers': model_params['hidden_layers'],
                    'learning_rate': model_params['learning_rate'],
                    'activation': model_params['activation'],
                    'dropout_rate': model_params.get('dropout_rate', 0.2),
                    'l2_reg': model_params.get('l2_reg', 0.001),
                    'use_batch_norm': model_params.get('use_batch_norm', True)
                },
                'train_params': {
                    'epochs': model_params['epochs'],
                    'batch_size': model_params['batch_size'],
                    'patience': model_params.get('patience', 15),
                    'X_val': X_val,
                    'y_val': y_val
                },
                'show_feature_importance': True, # alebo False, podľa potreby
                'feature_names': list(feature_info.keys()) # Pridanie feature_names do configu
            }
            
            model, model_metrics = train_model_for_segment(
                X_train_actual, y_train_actual, X_test, y_test, 
                segment_name, 
                config,
                base_save_path=model_dir,
                X_test_original_is_for_sale=is_for_sale_test_series # Pass the original 'is_for_sale' series
            )

            
            # Uloženie modelu a metrík
            models[segment_name] = model
            metrics[segment_name] = model_metrics
            
            # Vizualizácia predikcií
            visualize_dir = os.path.join(os.path.dirname(__file__), 'visualizations', 'predictions')
            if not os.path.exists(visualize_dir):
                os.makedirs(visualize_dir)
            
            # Určenie indexu a názvu príznaku pre vizualizáciu
            feature_index = 0  # Predvolený index
            feature_name = feature_columns[0] if feature_columns else 'feature'
            
            # visualize_predictions is now called within train_model_for_segment with X_test_original_is_for_sale
            # visualize_predictions(
            #     model, X_test, y_test,
            #     feature_name=feature_name, feature_index=feature_index,
            #     save_path=os.path.join(visualize_dir, f"{segment_name}_predictions.png")
            # )
            
        except Exception as e:
            print(f"Chyba pri trénovaní modelu pre segment {segment_name}: {str(e)}")
            continue
    
    # Výpis výsledkov pre všetky segmenty
    if metrics:
        print("\nVýsledky modelov pre všetky segmenty:")
        for segment_name, segment_metrics in metrics.items():
            print(f"\nSegment: {segment_name}")
            print(f"MSE: {segment_metrics['mse']:.4f}")
            print(f"RMSE: {segment_metrics['rmse']:.4f}")
            print(f"MAE: {segment_metrics['mae']:.4f}")
            print(f"R^2: {segment_metrics['r2']:.4f}")
    else:
        print("\nNeboli natrénované žiadne modely.")
    
    return segments, models, metrics
    
    # # Vizualizácia čistených dát
    # print("\nVizualizácia čistených dát...")
    # visualize_data_distribution(segments_cleaned)
    
    # # Porovnanie pôvodných a vyčistených dát
    # print("\nPorovnanie pôvodných a vyčistených dát...")
    # for segment_name in segments_with_super.keys():
    #     if segment_name in segments_cleaned:
    #         print(f"Porovnávam segment: {segment_name}")
    #         visualize_cleaning_results(
    #             segments_with_super[segment_name],
    #             segments_cleaned[segment_name],
    #             output_dir=f"./visualizations/cleaning_results/{segment_name}",
    #             columns=['price', 'floor_size']
    #         )


# Spustenie hlavnej funkcie
if __name__ == "__main__":
    segment_names = None  # Všetky segmenty

    feature_columns = ['floor_size', 'room_count', 'condition_score', 'is_new', 'is_renovated', 'streetAddress', 'addressRegion', 'latitude','longitude',
                        'Cena vrátane energií', 'Vybavenie_Výťah','Vybavenie_Balkón','Vybavenie_Vyhradené parkovanie',
                        'days_since_publication','update_publication_delta', 'is_for_sale'] # Pridaný 'is_for_sale' 
    # feature_columns = ['floor_size',]
    target_column = 'price' 
    
    # Parametre modelu
    model_params = {
        'hidden_layers': [128, 64, 32, 16],
        'learning_rate': 0.015,
        'activation': 'relu',
        'epochs': 200,
        'batch_size': 32,
        'dropout_rate': 0.15,
        'l2_reg': 0.0005,
        'use_batch_norm': True,
        'patience': 20
    }
    
    # Spustenie hlavnej funkcie s parametrami
    result = main(
        segment_names=segment_names,
        target_column=target_column,
        feature_columns=feature_columns,
        model_params=model_params
    )
    
    if isinstance(result, tuple) and len(result) == 3:
        segments_cleaned, models, metrics = result
        print("\nProgram bol úspešne dokončený s natrénovanými modelmi.")
    else:
        segments_cleaned = result
        print("\nProgram bol úspešne dokončený bez trénovania modelov.")
        


