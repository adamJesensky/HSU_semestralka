import pandas as pd
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_csv_data(file_path, encoding='utf-8', low_memory=False):
    """
    Načíta dáta z CSV súboru.
    
    Parameters:
    -----------
    file_path : str
        Cesta k CSV súboru
    encoding : str, optional
        Kódovanie súboru (default: 'utf-8')
    low_memory : bool, optional
        Parameter pre pandas read_csv (default: False)
    
    Returns:
    --------
    pandas.DataFrame
        Načítané dáta
    """
    try:
        df = pd.read_csv(file_path, encoding=encoding, low_memory=low_memory)
        print(f"Dáta úspešne načítané zo súboru {file_path}. Počet záznamov: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Súbor {file_path} nebol nájdený. Prosím, zadajte správnu cestu k dátam.")
        return None
    except Exception as e:
        print(f"Chyba pri načítaní dát: {e}")
        return None

def find_data_files(data_dir, region_name=None):
    """
    Nájde CSV súbory s dátami o nehnuteľnostiach podľa zadaných kritérií.
    
    Parameters:
    -----------
    data_dir : str
        Adresár s dátami
    region_name : str, optional
        Názov kraja (napr. 'zilinsky', 'bratislavsky'). Ak None, hľadajú sa všetky kraje.
    
    Returns:
    --------
    list
        Zoznam ciest k nájdeným súborom
    """
    if region_name:
        pattern = os.path.join(data_dir, f"Nehnutelnosti_{region_name}*kraj*.csv")
    else:
        pattern = os.path.join(data_dir, "Nehnutelnosti_*kraj*.csv")
    
    files = glob.glob(pattern)
    
    if not files:
        if region_name:
            print(f"Neboli nájdené žiadne súbory pre kraj: {region_name}")
        else:
            print(f"Neboli nájdené žiadne súbory v adresári: {data_dir}")
    
    return files

def extract_region_name(file_path):
    """
    Extrahuje názov kraja z cesty k súboru.
    
    Parameters:
    -----------
    file_path : str
        Cesta k súboru
    
    Returns:
    --------
    str
        Názov kraja
    """
    return os.path.basename(file_path).split('_')[1]

def load_data(region_name=None, regions_list=None, data_dir=os.path.join(os.path.dirname(__file__), '../zber_dat/nehnutelnosti_data'), combine_regions=True):
    """
    Univerzálna funkcia na načítanie dát z CSV súborov. Môže načítať dáta pre jeden kraj,
    viacero krajov alebo všetky kraje.
    
    Parameters:
    -----------
    region_name : str, optional
        Názov kraja (napr. 'zilinsky', 'bratislavsky'). Ignorované, ak je zadaný regions_list.
    regions_list : list, optional
        Zoznam názvov krajov na načítanie. Má prednosť pred region_name.
    data_dir : str, optional
        Adresár s dátami (default: '../zberDat_TrenovanieModelov_Testovanie/nehnutelnosti_data')
    combine_regions : bool, optional
        Ak True, vráti spojený DataFrame pre všetky kraje. Ak False, vráti slovník s DataFrame pre každý kraj.
    
    Returns:
    --------
    pandas.DataFrame alebo dict
        Načítané dáta ako DataFrame (ak combine_regions=True) alebo slovník s DataFrame pre každý kraj
    """
    all_data = []
    regions_data = {}
    
    # Spracovanie podľa zadaných parametrov
    if regions_list:
        # Načítanie dát pre zoznam krajov
        for region in regions_list:
            files = find_data_files(data_dir, region)
            if not files:
                continue
            
            df = load_csv_data(files[0])
            if df is not None:
                df['region_source'] = region
                all_data.append(df)
                regions_data[region] = df
    elif region_name:
        # Načítanie dát pre jeden konkrétny kraj
        files = find_data_files(data_dir, region_name)
        if files:
            df = load_csv_data(files[0])
            if df is not None:
                df['region_source'] = region_name
                all_data.append(df)
                regions_data[region_name] = df
    else:
        # Načítanie dát pre všetky kraje
        files = find_data_files(data_dir)
        if files:
            for file_path in tqdm(files, desc="Načítavanie dát z krajov"):
                df = load_csv_data(file_path)
                if df is not None:
                    region_name = extract_region_name(file_path)
                    df['region_source'] = region_name
                    all_data.append(df)
                    regions_data[region_name] = df
    
    # Kontrola, či boli načítané nejaké dáta
    if not all_data:
        print("Neboli načítané žiadne dáta.")
        return None
    
    # Vrátenie výsledku podľa parametra combine_regions
    if combine_regions:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Celkový počet načítaných záznamov: {len(combined_df)}")
        return combined_df
    else:
        return regions_data

def analyze_csv_structure(df):
    """
    Analyzuje štruktúru CSV dát a poskytuje základné informácie.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame na analýzu
    
    Returns:
    --------
    dict
        Slovník s informáciami o štruktúre dát
    """
    if df is None or df.empty:
        print("DataFrame je prázdny alebo None.")
        return None
    
    # Základné informácie o dátach
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'property_types': df['druh_objektu'].value_counts().to_dict() if 'druh_objektu' in df.columns else {},
        'transaction_types': df['typ'].value_counts().to_dict() if 'typ' in df.columns else {}
    }
    
    # Výpis základných informácií
    print(f"Počet riadkov: {info['rows']}")
    print(f"Počet stĺpcov: {info['columns']}")
    
    if 'druh_objektu' in df.columns:
        print("\nRozdelenie podľa druhu objektu:")
        for obj_type, count in info['property_types'].items():
            print(f"  {obj_type}: {count} ({count/info['rows']*100:.2f}%)")
    
    if 'typ' in df.columns:
        print("\nRozdelenie podľa typu transakcie:")
        for trans_type, count in info['transaction_types'].items():
            print(f"  {trans_type}: {count} ({count/info['rows']*100:.2f}%)")
    
    return info

def segment_data(df, allowed_property_types=None, allowed_transaction_types=None, clean_outliers=False, outlier_method='iqr', outlier_action='cap'):
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
    clean_outliers : bool, optional
        Či sa majú odstrániť outliers špecifické pre každý segment
    outlier_method : str, optional
        Metóda na detekciu outlierov: 'iqr', 'z-score', 'percentile'
    outlier_action : str, optional
        Akcia na spracovanie outlierov: 'remove', 'cap', 'log'
    
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
                # Ak je požadované čistenie outlierov, použijeme funkciu z data_cleaning
                if clean_outliers:
                    from data_cleaning import remove_segment_specific_outliers, detect_and_handle_outliers
                    # segment_data, _ = remove_segment_specific_outliers(
                    #     segment_data, 
                    #     segment_type=segment_name,
                    #     method=outlier_method,
                    #     action=outlier_action
                    # )
                    segment_data, _ = detect_and_handle_outliers(segment_data, {'price', 'floor_size'}, outlier_method, 3, outlier_action )

                segments[segment_name] = segment_data
                print(f"{segment_name}: {len(segment_data)} záznamov")
    
    return segments

# Funkcia na vytvorenie super segmentu obsahujúceho všetky dáta
def create_super_segment(segments):
    """
    Vytvorí super segment obsahujúci všetky dáta bez filtrovania.
    
    Parameters:
    -----------
    segments : dict
        Slovník so segmentami dát, kde kľúč je názov segmentu a hodnota je DataFrame
        
    Returns:
    --------
    dict
        Slovník obsahujúci pôvodné segmenty a nový super segment
    """
    # Importujeme funkciu z data_cleaning modulu
    from data_cleaning import merge_segments
    
    # Použijeme pokročilejšiu funkciu na zlúčenie segmentov
    return merge_segments(segments, create_combined=True, create_super=True)

# Funkcia na spracovanie dát - načítanie, segmentácia a čistenie
def process_data(region_name=None, regions_list=None, segment_filter=None, clean_outliers=False, outlier_method='iqr', outlier_action='cap'):
    """
    Komplexná funkcia na spracovanie dát - načítanie, segmentácia a voliteľné čistenie outlierov.
    
    Parameters:
    -----------
    region_name : str, optional
        Názov kraja na načítanie
    regions_list : list, optional
        Zoznam krajov na načítanie
    segment_filter : dict, optional
        Slovník s filtrami pre segmentáciu {'property_types': [...], 'transaction_types': [...]}
    clean_outliers : bool, optional
        Či sa majú odstrániť outliers špecifické pre každý segment
    outlier_method : str, optional
        Metóda na detekciu outlierov: 'iqr', 'z-score', 'percentile'
    outlier_action : str, optional
        Akcia na spracovanie outlierov: 'remove', 'cap', 'log'
    
    Returns:
    --------
    dict
        Slovník so segmentami dát
    """
    # Načítanie dát
    print("Načítavam dáta...")
    df = load_data(region_name=region_name, regions_list=regions_list)
    if df is None:
        print("Neboli načítané žiadne dáta.")
        return None
    
    # Analýza štruktúry dát
    print("\nAnalyzujem štruktúru dát...")
    analyze_csv_structure(df)
    
    # Segmentácia dát
    print("\nSegmentujem dáta...")
    allowed_property_types = None
    allowed_transaction_types = None
    
    if segment_filter:
        allowed_property_types = segment_filter.get('property_types')
        allowed_transaction_types = segment_filter.get('transaction_types')
    
    segments = segment_data(
        df, 
        allowed_property_types=allowed_property_types, 
        allowed_transaction_types=allowed_transaction_types,
        clean_outliers=clean_outliers,
        outlier_method=outlier_method,
        outlier_action=outlier_action
    )
    
    print(f"\nVytvorených {len(segments)} segmentov dát.")
    return segments

# Funkcia na vizualizáciu distribúcie dát
def visualize_data_distribution(segments):
    """
    Vytvorí vizualizácie distribúcie dát pre všetky segmenty.
    
    Parameters:
    -----------
    segments : dict
        Slovník so segmentami dát, kde kľúč je názov segmentu a hodnota je DataFrame
    """
    if not segments:
        print("Žiadne segmenty na vizualizáciu.")
        return
    
    # Vytvorenie adresára pre vizualizácie
    os.makedirs('./visualizations', exist_ok=True)
    
    # Pre každý segment vytvoríme vizualizácie
    for segment_name, segment_data in segments.items():
        print(f"\nVytváranie vizualizácií pre segment: {segment_name}")
        
        # Kontrola, či ide o combined segment alebo super segment
        is_combined_segment = "Combined" in segment_name or segment_name == "Super_Segment"
        
        # Vytvorenie adresára pre segment
        segment_dir = os.path.join("visualizations", segment_name)
        os.makedirs(segment_dir, exist_ok=True)
        
        # Základné informácie o segmente
        print(f"Počet záznamov: {len(segment_data)}")
        print(f"Počet stĺpcov: {len(segment_data.columns)}")
        # print(f"Dostupné stĺpce: {', '.join(segment_data.columns.tolist())}")
        
        # Hľadáme stĺpec s cenou (môže mať rôzne názvy)
        price_columns = [col for col in segment_data.columns if col.lower() in ['cena', 'price']]
        if price_columns:
            price_col = price_columns[0]
            print(f"Používam stĺpec '{price_col}' pre cenu")
            
            # Vizualizácia distribúcie cien - logaritmická transformácia len pre combined segmenty
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            
            if is_combined_segment:
                # Logaritmická transformácia pre combined segmenty
                sns.histplot(np.log1p(segment_data[price_col].dropna()), kde=True)
                plt.title(f'Distribúcia cien (logaritmická) - {segment_name}')
                plt.xlabel('Log(Cena)')
            else:
                # Pôvodné hodnoty pre ostatné segmenty
                sns.histplot(segment_data[price_col].dropna(), kde=True)
                plt.title(f'Distribúcia cien - {segment_name}')
                plt.xlabel('Cena')
            
            plt.ylabel('Počet')
            
            plt.subplot(2, 1, 2)
            if is_combined_segment:
                sns.boxplot(x=np.log1p(segment_data[price_col].dropna()))
                plt.title(f'Boxplot cien (logaritmický) - {segment_name}')
                plt.xlabel('Log(Cena)')
            else:
                sns.boxplot(x=segment_data[price_col].dropna())
                plt.title(f'Boxplot cien - {segment_name}')
                plt.xlabel('Cena')
            
            plt.tight_layout()
            plt.savefig(os.path.join(segment_dir, 'cena_distribúcia.png'))
            plt.close()
        else:
            print("Stĺpec s cenou nebol nájdený.")
        
        # Hľadáme stĺpec s rozlohou (môže mať rôzne názvy)
        area_columns = [col for col in segment_data.columns if col.lower() in ['floor_size']]
        if area_columns:
            area_col = area_columns[0]
            print(f"Používam stĺpec '{area_col}' pre rozlohu")
            
            # Vizualizácia distribúcie rozlohy - vždy bez logaritmickej transformácie
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            sns.histplot(segment_data[area_col].dropna(), kde=True)
            plt.title(f'Distribúcia rozlohy - {segment_name}')
            plt.xlabel('Rozloha (m²)')
            plt.ylabel('Počet')
            
            plt.subplot(2, 1, 2)
            sns.boxplot(x=segment_data[area_col].dropna())
            plt.title(f'Boxplot rozlohy - {segment_name}')
            plt.xlabel('Rozloha (m²)')
            plt.tight_layout()
            plt.savefig(os.path.join(segment_dir, 'rozloha_distribúcia.png'))
            plt.close()
            
            # Vzťah medzi cenou a rozlohou - logaritmická transformácia len pre combined segmenty
            if price_columns:
                plt.figure(figsize=(10, 8))
                
                if is_combined_segment:
                    # Logaritmická transformácia pre combined segmenty
                    plt.scatter(segment_data[area_col], np.log1p(segment_data[price_col]))
                    plt.title(f'Vzťah medzi cenou (logaritmická) a rozlohou - {segment_name}')
                    plt.ylabel('Log(Cena)')
                else:
                    # Pôvodné hodnoty pre ostatné segmenty
                    plt.scatter(segment_data[area_col], segment_data[price_col])
                    plt.title(f'Vzťah medzi cenou a rozlohou - {segment_name}')
                    plt.ylabel('Cena')
                
                plt.xlabel('Rozloha (m²)')
                plt.tight_layout()
                plt.savefig(os.path.join(segment_dir, 'cena_vs_rozloha.png'))
                plt.close()
        else:
            print("Stĺpec s rozlohou nebol nájdený.")
        
        # Distribúcia podľa regiónov - používame stĺpec 'addressRegion' namiesto 'region_source'
        region_column = 'addressRegion' if 'addressRegion' in segment_data.columns else 'region_source'
        
        if region_column in segment_data.columns:
            # Kontrola, či máme viac ako jeden región v dátach
            unique_regions = segment_data[region_column].nunique()
            
            if unique_regions > 1:
                plt.figure(figsize=(14, 8))
                region_counts = segment_data[region_column].value_counts()
                sns.barplot(x=region_counts.index, y=region_counts.values)
                plt.title(f'Počet nehnuteľností podľa regiónov - {segment_name}')
                plt.xlabel('Región')
                plt.ylabel('Počet')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(segment_dir, 'regiony_distribúcia.png'))
                plt.close()
                
                # Priemerná cena podľa regiónov
                if price_columns:
                    plt.figure(figsize=(14, 8))
                    avg_price_by_region = segment_data.groupby(region_column)[price_col].mean().sort_values(ascending=False)
                    sns.barplot(x=avg_price_by_region.index, y=avg_price_by_region.values)
                    plt.title(f'Priemerná cena podľa regiónov - {segment_name}')
                    plt.xlabel('Región')
                    plt.ylabel('Priemerná cena')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(segment_dir, 'regiony_priemerna_cena.png'))
                    plt.close()
                    
                    # Distribučné krivky cien pre jednotlivé regióny
                    plt.figure(figsize=(14, 10))
                    
                    # Vytvorenie spoločnej distribučnej krivky pre všetky regióny
                    if is_combined_segment:
                        # Logaritmická transformácia pre combined segmenty
                        sns.kdeplot(np.log1p(segment_data[price_col].dropna()), label='Všetky regióny', linewidth=3, color='black')
                        plt.title(f'Distribúcia cien (logaritmická) podľa regiónov - {segment_name}')
                        plt.xlabel('Log(Cena)')
                    else:
                        # Pôvodné hodnoty pre ostatné segmenty
                        sns.kdeplot(segment_data[price_col].dropna(), label='Všetky regióny', linewidth=3, color='black')
                        plt.title(f'Distribúcia cien podľa regiónov - {segment_name}')
                        plt.xlabel('Cena')
                    
                    # Vytvorenie distribučných kriviek pre jednotlivé regióny
                    regions = segment_data[region_column].unique()
                    for region in regions:
                        region_data = segment_data[segment_data[region_column] == region]
                        if len(region_data) > 10:  # Vykresľujeme len regióny s dostatočným počtom dát
                            if is_combined_segment:
                                # Logaritmická transformácia pre combined segmenty
                                sns.kdeplot(np.log1p(region_data[price_col].dropna()), label=region)
                            else:
                                # Pôvodné hodnoty pre ostatné segmenty
                                sns.kdeplot(region_data[price_col].dropna(), label=region)
                    
                    plt.ylabel('Hustota')
                    plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    plt.savefig(os.path.join(segment_dir, 'regiony_cena_distribúcia_krivky.png'))
                    plt.close()
            else:
                print(f"Segment {segment_name} obsahuje len jeden región, preskakujem vizualizáciu podľa regiónov.")
        
        # Ďalšie zaujímavé stĺpce pre vizualizáciu
        interesting_columns = ['stav_objektu', 'pocet_izieb', 'poschodie', 'rok_vystavby']
        
        # Pridáme 'druh_objektu' a 'typ' len pre Combined segmenty, keďže ostatné segmenty sú už filtrované
        if is_combined_segment:
            interesting_columns.extend(['druh_objektu', 'typ'])
        
        for col in interesting_columns:
            if col in segment_data.columns and segment_data[col].notna().sum() > 0:
                plt.figure(figsize=(12, 8))
                if segment_data[col].dtype == 'object':
                    value_counts = segment_data[col].value_counts().sort_values(ascending=False).head(15)
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.title(f'Distribúcia {col} - {segment_name}')
                    plt.xlabel(col)
                    plt.ylabel('Počet')
                    plt.xticks(rotation=45)
                else:
                    sns.histplot(segment_data[col].dropna(), kde=True)
                    plt.title(f'Distribúcia {col} - {segment_name}')
                    plt.xlabel(col)
                    plt.ylabel('Počet')
                plt.tight_layout()
                plt.savefig(os.path.join(segment_dir, f'{col}_distribúcia.png'))
                plt.close()
        
        print(f"Vizualizácie pre segment {segment_name} boli uložené do adresára {segment_dir}")

def load_processed_data(processed_data_dir=None):
    """
    Načíta vyčistené a spracované dáta z CSV súborov v zadanom adresári.
    
    Parameters:
    -----------
    processed_data_dir : str, optional
        Adresár s vyčistenými dátami. Ak None, použije sa predvolený adresár 'processed_data'.
    
    Returns:
    --------
    dict
        Slovník segmentov s načítanými dátami
    """
    if processed_data_dir is None:
        processed_data_dir = os.path.join(os.path.dirname(__file__), 'processed_data')
    
    if not os.path.exists(processed_data_dir):
        print(f"Adresár s vyčistenými dátami '{processed_data_dir}' neexistuje.")
        return None
    
    segments = {}
    csv_files = glob.glob(os.path.join(processed_data_dir, '*.csv'))
    
    if not csv_files:
        print(f"V adresári '{processed_data_dir}' neboli nájdené žiadne CSV súbory.")
        return None
    
    for file_path in csv_files:
        segment_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pd.read_csv(file_path)
            segments[segment_name] = df
            print(f"Načítaný segment {segment_name} s {len(df)} záznamami.")
        except Exception as e:
            print(f"Chyba pri načítaní segmentu {segment_name}: {e}")
    
    return segments

# Príklady použitia
if __name__ == "__main__":

    # Príklad : Segmentácia všetkých nehnuteľností na prenájom
    # prenajom_segments = process_data(regions_list=['zilinsky', 'bratislavsky'], 
    #                                  segment_filter={'property_types': ['Byty'],'transaction_types': ['Predaj']})

    # Príklad : Segmentácia všetkých nehnuteľností na prenájom
    # segments = process_data(regions_list=[], 
    #                         segment_filter={'property_types': ['Byty', 'Domy'],'transaction_types': ['Predaj', 'Prenájom']},
    #                         clean_outliers = True, 
    #                         outlier_method ='z-score',
    #                         outlier_action = 'remove')
    
    segments = process_data(regions_list=[], 
                        segment_filter={'property_types': ['Byty', 'Domy'],'transaction_types': ['Predaj', 'Prenájom']},
                        clean_outliers = False)
    
    segments = create_super_segment(segments)
    print(segments.keys())