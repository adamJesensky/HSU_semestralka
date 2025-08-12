import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# Trieda pre neurónovú sieť využívajúcu TensorFlow/Keras
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers=[64, 32], output_size=1, learning_rate=0.001, activation='relu',
                 dropout_rate=0.2, l2_reg=0.001, use_batch_norm=True):
        """
        Inicializácia neurónovej siete.
        
        Parameters:
        -----------
        input_size : int
            Počet vstupných príznakov
        hidden_layers : list
            Zoznam počtu neurónov v skrytých vrstvách
        output_size : int
            Počet výstupných neurónov (1 pre regresiu)
        learning_rate : float
            Rýchlosť učenia
        activation : str
            Aktivačná funkcia ('relu', 'sigmoid', 'tanh', 'elu')
        dropout_rate : float
            Miera dropout-u pre regularizáciu (0.0 až 1.0)
        l2_reg : float
            Koeficient L2 regularizácie
        use_batch_norm : bool
            Použiť batch normalization
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Vytvorenie modelu
        self.model = self._build_model()
        
        # História trénovania pre vizualizáciu
        self.loss_history = []
        self.val_loss_history = []
    
    def _build_model(self):
        """
        Vytvorenie architektúry modelu.
        
        Returns:
        --------
        tensorflow.keras.Model
            Skompilovaný model
        """
        model = models.Sequential()
        
        # Vstupná vrstva
        model.add(layers.Input(shape=(self.input_size,)))
        
        # Skryté vrstvy
        for i, units in enumerate(self.hidden_layers):
            # Pridanie Dense vrstvy s regularizáciou
            model.add(layers.Dense(
                units=units,
                activation=None,  # Aktivácia bude aplikovaná po batch normalizácii
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'dense_{i}'
            ))
            
            # Batch normalizácia (ak je povolená)
            if self.use_batch_norm:
                model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))
            
            # Aktivačná funkcia
            model.add(layers.Activation(self.activation_name, name=f'activation_{i}'))
            
            # Dropout pre regularizáciu
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Výstupná vrstva (lineárna aktivácia pre regresiu)
        model.add(layers.Dense(self.output_size, name='output'))
        
        # Kompilácia modelu
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
              patience=20, min_delta=0.001, verbose=1, callbacks=None): # Pridaný parameter callbacks
        """
        Trénovanie neurónovej siete s early stopping.
        
        Parameters:
        -----------
        X_train : numpy.ndarray alebo pandas.DataFrame
            Trénovacie dáta
        y_train : numpy.ndarray alebo pandas.Series
            Cieľové hodnoty pre trénovacie dáta
        X_val : numpy.ndarray alebo pandas.DataFrame, optional
            Validačné dáta
        y_val : numpy.ndarray alebo pandas.Series, optional
            Cieľové hodnoty pre validačné dáta
        epochs : int
            Maximálny počet epoch
        batch_size : int
            Veľkosť dávky
        patience : int
            Počet epoch bez zlepšenia pre early stopping
        min_delta : float
            Minimálna zmena pre považovanie za zlepšenie
        verbose : int
            Úroveň výpisov počas trénovania (0=žiadne, 1=progress bar, 2=jedna riadka na epochu)
        
        Returns:
        --------
        dict
            História trénovania
        """
        # Konverzia vstupných dát na numpy array, ak sú to pandas DataFrame/Series
        if isinstance(X_train, pd.DataFrame):
            X_train_np = X_train.values
        else:
            X_train_np = X_train
            
        if isinstance(y_train, pd.Series):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
            
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val_np = X_val.values
        else:
            X_val_np = X_val
            
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val_np = y_val.values
        else:
            y_val_np = y_val
        
        # Definícia callbackov
        callbacks_list = []
        
        # Early stopping - zastaví trénovanie, keď sa validačná strata nezlepšuje
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Redukcia learning rate pri stagnácii
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=self.learning_rate / 100,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # Trénovanie modelu
        validation_data = None
        if X_val_np is not None and y_val_np is not None:
            validation_data = (X_val_np, y_val_np)
        
        history = self.model.fit(
            X_train_np, y_train_np,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        # Uloženie histórie trénovania
        self.loss_history = history.history.get('loss', [])
        self.val_loss_history = history.history.get('val_loss', []) # Použitie .get pre bezpečnosť
        
        return history # Vráti celý objekt history
    
    def predict(self, X):
        """
        Predikcia pomocou natrénovaného modelu.
        
        Parameters:
        -----------
        X : numpy.ndarray alebo pandas.DataFrame
            Vstupné dáta
        
        Returns:
        --------
        numpy.ndarray
            Predikované hodnoty
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X).flatten()
    
    def evaluate(self, X, y):
        """
        Vyhodnotenie modelu pomocou vlastných metrík.
        
        Parameters:
        -----------
        X : numpy.ndarray/pandas.DataFrame
            Testovacie dáta
        y : numpy.ndarray/pandas.Series
            Cieľové hodnoty
        
        Returns:
        --------
        dict
            Slovník s metrikami
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Predikcia
        y_pred = self.predict(X)
        
        # Výpočet metrík
        mse = np.mean((y_pred - y) ** 2)
        mae = np.mean(np.abs(y_pred - y))
        rmse = np.sqrt(mse)
        
        # Výpočet R^2 skóre
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def plot_loss(self, save_path=None):
        """
        Vizualizácia priebehu trénovania.
        
        Parameters:
        -----------
        save_path : str, optional
            Cesta pre uloženie grafu
        """
        plt.figure(figsize=(10, 6))
        if hasattr(self, 'loss_history') and self.loss_history:
            plt.plot(self.loss_history, label='Tréningová strata')
        if hasattr(self, 'val_loss_history') and self.val_loss_history:
            plt.plot(self.val_loss_history, label='Validačná strata')
        
        if not (hasattr(self, 'loss_history') and self.loss_history) and not (hasattr(self, 'val_loss_history') and self.val_loss_history):
            plt.text(0.5, 0.5, 'Žiadne dáta o strate na zobrazenie.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Priebeh trénovania (strata)')
            plt.xlabel('Epocha')
            plt.ylabel('Strata')
        else:
            plt.title('Priebeh trénovania (strata)')
            plt.xlabel('Epocha')
            plt.ylabel('Strata (MSE)') # Opravený ylabel
            plt.legend()
            plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_model(self, file_path):
        """
        Uloženie modelu do súboru.
        
        Parameters:
        -----------
        file_path : str
            Cesta k súboru (.keras alebo .h5)
        """
        # Kontrola, či cesta končí na .keras alebo .h5
        if not (file_path.endswith('.keras') or file_path.endswith('.h5')):
            file_path = file_path + '.keras'  # Použitie nového formátu ako predvoleného
            
        self.model.save(file_path)
        print(f"Model uložený do {file_path}")
    
    @classmethod
    def load_model(cls, file_path):
        """
        Načítanie modelu zo súboru.
        
        Parameters:
        -----------
        file_path : str
            Cesta k súboru (.keras alebo .h5)
        
        Returns:
        --------
        NeuralNetwork
            Načítaný model
        """
        # Vytvorenie prázdnej inštancie
        instance = cls(input_size=1)  # Dočasné hodnoty
        
        # Načítanie modelu
        instance.model = models.load_model(file_path)
        
        # Aktualizácia atribútov
        instance.input_size = instance.model.input_shape[1]
        instance.output_size = instance.model.output_shape[1]
        
        # Zistenie hidden_layers z architektúry modelu
        hidden_layers = []
        for layer in instance.model.layers:
            if isinstance(layer, layers.Dense) and layer.name != 'output':
                hidden_layers.append(layer.units)
        
        instance.hidden_layers = hidden_layers
        
        return instance


def train_model_for_segment(X_train, y_train, X_test, y_test, segment_name, config, base_save_path='.', X_test_original_is_for_sale=None):
    """
    Trénuje model pre daný segment dát.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Trénovacie dáta (príznaky)
    y_train : numpy.ndarray
        Trénovacie dáta (cieľová premenná)
    X_test : numpy.ndarray
        Testovacie dáta (príznaky)
    y_test : numpy.ndarray
        Testovacie dáta (cieľová premenná)
    segment_name : str
        Názov segmentu
    config : dict
        Konfigurácia modelu
    base_save_path : str, optional
        Základná cesta pre ukladanie modelov a grafov
    X_test_original_is_for_sale : pd.Series, optional
        Pôvodný stĺpec 'is_for_sale' pre testovaciu sadu (pre kombinované segmenty)
    
    Returns:
    --------
    tuple
        Natrénovaný model a metriky (slovník results, importance)
    """
    print(f"\nTrénovanie modelu pre segment: {segment_name}")
    
    # Inicializácia a trénovanie modelu
    model = NeuralNetwork(input_size=X_train.shape[1], **config.get('model_params', {}))
    train_params = config.get('train_params', {})
    X_val = train_params.pop('X_val', None)
    y_val = train_params.pop('y_val', None)
    # Odovzdanie X_val, y_val do metódy train
    history = model.train(X_train, y_train, X_val=X_val, y_val=y_val, **train_params)
    
    # Vytvorenie ciest pre ukladanie
    segment_path = os.path.join(base_save_path, segment_name)
    os.makedirs(segment_path, exist_ok=True)
    
    # Uloženie modelu
    model_save_path = os.path.join(segment_path, f"{segment_name}_model.h5")
    model.save_model(model_save_path)
    print(f"Model pre segment {segment_name} uložený do {model_save_path}")
    
    # Uloženie grafu priebehu trénovania
    loss_plot_path = os.path.join(segment_path, f"{segment_name}_loss_plot.png")
    model.plot_loss(save_path=loss_plot_path)
    print(f"Graf priebehu trénovania pre segment {segment_name} uložený do {loss_plot_path}")
    
    # Evaluácia a vizualizácia
    print(f"\nEvaluácia modelu pre segment: {segment_name}")
    predictions_plot_path = os.path.join(segment_path, f"{segment_name}_predictions_plot.png")
    
    # Získanie názvov príznakov
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
    elif isinstance(X_test, pd.DataFrame): # Fallback na X_test, ak X_train nie je DataFrame
        feature_names = X_test.columns.tolist()
    else: # Fallback na feature_names z configu, ak ani jedno nie je DataFrame
        feature_names = config.get('feature_names', [f'Feature_{i}' for i in range(X_train.shape[1])])

    metrics_dict, importance = visualize_predictions(
        model,
        X_test,
        y_test,
        feature_names=feature_names, 
        save_path=predictions_plot_path,
        show_feature_importance=config.get('show_feature_importance', True),
        segment_name=segment_name,
        X_test_original_is_for_sale=X_test_original_is_for_sale,
        top_n_features=config.get('top_n_features_to_display', 10) # Pridanie nového parametra
    )
    print(f"Graf predikcií pre segment {segment_name} uložený do {predictions_plot_path}")
    
    # Kombinovanie metrik do jedneho slovnika
    final_metrics = metrics_dict.copy()
    final_metrics['feature_importance'] = importance
    return model, final_metrics


def calculate_feature_importance(model, X, y, feature_names=None, n_repeats=10, random_state=42):
    """
    Vypočíta dôležitosť príznakov pomocou permutačnej metódy.
    
    Parameters:
    -----------
    model : NeuralNetwork
        Natrénovaný model
    X : numpy.ndarray alebo pandas.DataFrame
        Vstupné dáta
    y : numpy.ndarray alebo pandas.Series
        Skutočné hodnoty
    feature_names : list, optional
        Zoznam názvov príznakov
    n_repeats : int, optional
        Počet opakovaní permutácie pre každý príznak
    random_state : int, optional
        Seed pre generátor náhodných čísel
    
    Returns:
    --------
    dict
        Slovník s dôležitosťou príznakov
    """
    # Konverzia vstupných dát na numpy array, ak sú to pandas DataFrame/Series
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X_np = X.values
    else:
        X_np = X
        feature_names = feature_names or [f'Feature_{i}' for i in range(X_np.shape[1])]
        
    if isinstance(y, pd.Series):
        y_np = y.values
    else:
        y_np = y
    
    # Výpočet základnej chyby
    baseline_pred = model.predict(X_np)
    baseline_error = np.mean((baseline_pred - y_np) ** 2)  # MSE
    
    # Inicializácia slovníka pre uloženie dôležitosti príznakov
    importance = {}
    
    # Výpočet dôležitosti pre každý príznak
    np.random.seed(random_state)
    
    for i, feature_name in enumerate(feature_names):
        # Inicializácia zoznamu pre uloženie chýb po permutácii
        errors = []
        
        for _ in range(n_repeats):
            # Vytvorenie kópie dát
            X_permuted = X_np.copy()
            
            # Permutácia hodnôt v stĺpci
            perm_idx = np.random.permutation(len(X_permuted))
            X_permuted[:, i] = X_permuted[perm_idx, i]
            
            # Predikcia s permutovanými dátami
            perm_pred = model.predict(X_permuted)
            
            # Výpočet chyby
            perm_error = np.mean((perm_pred - y_np) ** 2)  # MSE
            
            # Výpočet relatívnej zmeny chyby
            rel_error = (perm_error - baseline_error) / baseline_error
            errors.append(rel_error)
        
        # Uloženie priemernej dôležitosti príznaku
        importance[feature_name] = np.mean(errors)
    
    # Normalizácia dôležitosti príznakov (aby súčet bol 1)
    total_importance = sum(max(0, imp) for imp in importance.values())
    if total_importance > 0:
        normalized_importance = {feat: max(0, imp) / total_importance for feat, imp in importance.items()}
    else:
        normalized_importance = {feat: 1.0 / len(importance) for feat in importance.keys()}
    
    return normalized_importance

def visualize_feature_importance(importance, title="Dôležitosť príznakov", figsize=(10, 6), save_path=None):
    """
    Vizualizuje dôležitosť príznakov.
    
    Parameters:
    -----------
    importance : dict
        Slovník s dôležitosťou príznakov
    title : str, optional
        Názov grafu
    figsize : tuple, optional
        Veľkosť grafu
    save_path : str, optional
        Cesta pre uloženie grafu
    """
    # Zoradenie príznakov podľa dôležitosti
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # Vytvorenie grafu
    plt.figure(figsize=figsize)
    
    # Vykreslenie stĺpcového grafu
    bars = plt.bar(range(len(sorted_importance)), list(sorted_importance.values()), align='center')
    
    # Pridanie popiskov
    plt.xticks(range(len(sorted_importance)), list(sorted_importance.keys()), rotation=45, ha='right')
    plt.xlabel('Príznaky')
    plt.ylabel('Relatívna dôležitosť')
    plt.title(title)
    
    # Pridanie hodnôt nad stĺpce
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_feature_importance(model, X, y, feature_names=None):
    """
    Výpočet dôležitosti príznakov pomocou permutačnej metódy.
    
    Parameters:
    -----------
    model : NeuralNetwork
        Natrénovaný model
    X : numpy.ndarray alebo pandas.DataFrame
        Vstupné dáta
    y : numpy.ndarray alebo pandas.Series
        Skutočné hodnoty
    feature_names : list, optional
        Zoznam názvov príznakov
    
    Returns:
    --------
    dict
        Slovník s dôležitosťou príznakov
    """
    # Konverzia vstupných dát na numpy array, ak sú to pandas DataFrame/Series
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X_np = X.values
    else:
        X_np = X
        feature_names = feature_names or [f'Feature_{i}' for i in range(X_np.shape[1])]
        
    if isinstance(y, pd.Series):
        y_np = y.values
    else:
        y_np = y
    
    # Základná predikcia a chyba
    baseline_pred = model.predict(X_np)
    baseline_error = np.mean((baseline_pred - y_np) ** 2)  # MSE
    
    # Výpočet dôležitosti pre každý príznak
    importance = {}
    for i, feature_name in enumerate(feature_names):
        # Vytvorenie kópie dát
        X_permuted = X_np.copy()
        
        # Permutácia hodnôt príznaku
        np.random.shuffle(X_permuted[:, i])
        
        # Predikcia s permutovanými dátami
        permuted_pred = model.predict(X_permuted)
        permuted_error = np.mean((permuted_pred - y_np) ** 2)  # MSE
        
        # Dôležitosť = nárast chyby po permutácii
        importance[feature_name] = max(0, (permuted_error - baseline_error) / baseline_error)
    
    return importance

def visualize_predictions(model, X_test, y_test, feature_names=None, feature_index=0, save_path=None, show_feature_importance=True, segment_name=None, X_test_original_is_for_sale=None, top_n_features=10):
    """
    Rozšírená vizualizácia predikcií modelu s analýzou dôležitosti príznakov.
    
    Parameters:
    -----------
    model : NeuralNetwork
        Natrénovaný model
    X_test : numpy.ndarray alebo pandas.DataFrame
        Testovacie dáta
    y_test : numpy.ndarray alebo pandas.Series
        Skutočné hodnoty
    feature_names : list, optional
        Zoznam názvov príznakov
    feature_index : int, optional
        Index príznaku pre os x
    save_path : str, optional
        Cesta pre uloženie grafu
    show_feature_importance : bool, optional
        Či sa má zobraziť analýza dôležitosti príznakov
    segment_name : str, optional
        Názov segmentu, pre ktorý sa robí vizualizácia
    X_test_original_is_for_sale : pd.Series, optional
        Pôvodný stĺpec 'is_for_sale' pre testovaciu sadu (pre kombinované segmenty)
    """
    # Konverzia vstupných dát na numpy array, ak sú to pandas DataFrame/Series
    if isinstance(X_test, pd.DataFrame):
        feature_names = feature_names or X_test.columns.tolist()
        X_test_np = X_test.values
    else:
        X_test_np = X_test
        feature_names = feature_names or [f'Feature_{i}' for i in range(X_test_np.shape[1])]
        
    if isinstance(y_test, pd.Series):
        y_test_np = y_test.values
    else:
        y_test_np = y_test
    
    # Predikcia
    y_pred = model.predict(X_test_np).flatten()
    
    # Výpočet metrík
    mse = np.mean((y_pred - y_test_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test_np))
    
    # Výpočet R^2 skóre
    ss_total = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    ss_residual = np.sum((y_test_np - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Výpočet dôležitosti príznakov, ak je požadované
    if show_feature_importance and X_test_np.shape[1] > 1:
        # Uistime sa, ze pouzivame feature_names definovane na zaciatku funkcie
        current_feature_names = feature_names 
        importance = calculate_feature_importance(model, X_test, y_test, current_feature_names)
        
        # Vytvorenie grafu s 2x3 podgrafmi
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Scatter plot predikcie vs. skutočné hodnoty
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_test_np, y_pred, alpha=0.5)
        ax1.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
        ax1.set_xlabel('Skutočné hodnoty')
        ax1.set_ylabel('Predikované hodnoty')
        ax1.set_title('Predikcia vs. Skutočnosť')
        ax1.grid(True)
        
        # Pridanie metrík do grafu
        textstr = f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # 2. Histogram reziduálov
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = y_test_np - y_pred
        
        # Kontrola rozsahu reziduálov
        residual_range = np.max(residuals) - np.min(residuals)
        
        # Ak sú všetky reziduály rovnaké alebo takmer rovnaké, použijeme len 1 bin
        if residual_range < 1e-10 or len(np.unique(residuals)) <= 1:
            ax2.text(0.5, 0.5, 'Všetky reziduály sú rovnaké', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
        else:
            # Dynamické určenie počtu binov na základe rozsahu dát a počtu unikátnych hodnôt
            unique_values = len(np.unique(residuals))
            bins = min(30, max(5, unique_values // 10))
            ax2.hist(residuals, bins=bins, alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--')
            
            # Pridanie krivky normálneho rozdelenia
            from scipy import stats
            x = np.linspace(np.min(residuals), np.max(residuals), 100)
            mean, std = np.mean(residuals), np.std(residuals)
            pdf = stats.norm.pdf(x, mean, std)
            ax2.plot(x, pdf * len(residuals) * (np.max(residuals) - np.min(residuals)) / bins, 'r-', linewidth=2)
            
        ax2.set_xlabel('Reziduály (Skutočné - Predikované)')
        ax2.set_ylabel('Počet')
        ax2.set_title('Distribúcia reziduálov')
        
        # 3. Scatter plot predikcie vs. najdôležitejší príznak
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Výber najdôležitejšieho príznaku, ak je k dispozícii
        if importance:
            most_important_feature = max(importance.items(), key=lambda x: x[1])[0]
            most_important_idx = feature_names.index(most_important_feature)
            
            ax3.scatter(X_test_np[:, most_important_idx], y_pred, alpha=0.5, label='Predikcia')
            ax3.scatter(X_test_np[:, most_important_idx], y_test_np, alpha=0.5, label='Skutočnosť')
            ax3.set_xlabel(most_important_feature)
            ax3.set_title(f'Predikcia vs. Najdôležitejší príznak ({most_important_feature})')
        else:
            # Použitie indexu príznaku, ak dôležitosť nie je k dispozícii
            if X_test_np.shape[1] > feature_index:
                ax3.scatter(X_test_np[:, feature_index], y_pred, alpha=0.5, label='Predikcia')
                ax3.scatter(X_test_np[:, feature_index], y_test_np, alpha=0.5, label='Skutočnosť')
                ax3.set_xlabel(feature_names[feature_index])
                ax3.set_title(f'Predikcia vs. Príznak ({feature_names[feature_index]})')
            else:
                ax3.text(0.5, 0.5, 'Nedostatok príznakov pre vizualizáciu', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes)
        
        ax3.set_ylabel('Hodnota')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Q-Q plot reziduálov (pre kontrolu normality)
        ax4 = fig.add_subplot(gs[1, 0])
        residuals_sorted = np.sort(residuals)
        n = len(residuals_sorted)
        if n > 1:
            # Teoretické kvantily normálneho rozdelenia
            quantiles = np.arange(1, n + 1) / (n + 1)
            theoretical_quantiles = np.sqrt(2) * np.array([np.log(q / (1 - q)) for q in quantiles])
            
            # Vykreslenie Q-Q plotu
            ax4.scatter(theoretical_quantiles, residuals_sorted, alpha=0.5)
            
            # Referenčná čiara
            min_val = min(theoretical_quantiles.min(), residuals_sorted.min())
            max_val = max(theoretical_quantiles.max(), residuals_sorted.max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax4.set_xlabel('Teoretické kvantily')
            ax4.set_ylabel('Kvantily reziduálov')
            ax4.set_title('Q-Q Plot reziduálov')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Nedostatok dát pre Q-Q plot', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
        
        # 5. Dôležitosť príznakov
        ax5 = fig.add_subplot(gs[1, 1:])
        
        # Zoradenie príznakov podľa dôležitosti a výber top N
        if importance:
            sorted_importance_full = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            top_features_keys = list(sorted_importance_full.keys())[:top_n_features]
            sorted_importance = {k: sorted_importance_full[k] for k in top_features_keys if k in sorted_importance_full}
        else:
            sorted_importance = {}

        if sorted_importance:
            # Vykreslenie stĺpcového grafu
            bars = ax5.bar(range(len(sorted_importance)), list(sorted_importance.values()), align='center')
            
            # Pridanie popiskov
            ax5.set_xticks(range(len(sorted_importance)))
            ax5.set_xticklabels(list(sorted_importance.keys()), rotation=45, ha='right')
        else:
            ax5.text(0.5, 0.5, 'Nebola vypočítaná dôležitosť príznakov.', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
        ax5.set_xlabel('Príznaky')
        ax5.set_ylabel('Relatívna dôležitosť')
        ax5.set_title('Dôležitosť príznakov (permutačná metóda)')
        
        # Pridanie hodnôt nad stĺpce, len ak 'bars' existuje
        if 'bars' in locals() and bars:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
    else:
        # Vytvorenie grafu s 2x2 podgrafmi (pôvodná verzia)
        plt.figure(figsize=(12, 10))
        
        # 1. Scatter plot predikcie vs. skutočné hodnoty
        plt.subplot(2, 2, 1)
        plt.scatter(y_test_np, y_pred, alpha=0.5)
        plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
        plt.xlabel('Skutočné hodnoty')
        plt.ylabel('Predikované hodnoty')
        plt.title('Predikcia vs. Skutočnosť')
        plt.grid(True)
        
        # Pridanie metrík do grafu
        textstr = f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        # 2. Histogram reziduálov
        plt.subplot(2, 2, 2)
        residuals = y_test_np - y_pred
        
        # Kontrola rozsahu reziduálov
        residual_range = np.max(residuals) - np.min(residuals)
        
        # Ak sú všetky reziduály rovnaké alebo takmer rovnaké, použijeme len 1 bin
        if residual_range < 1e-10 or len(np.unique(residuals)) <= 1:
            plt.text(0.5, 0.5, 'Všetky reziduály sú rovnaké', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
        else:
            # Dynamické určenie počtu binov na základe rozsahu dát a počtu unikátnych hodnôt
            unique_values = len(np.unique(residuals))
            bins = min(30, max(5, unique_values // 10))
            plt.hist(residuals, bins=bins, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            
            # Pridanie krivky normálneho rozdelenia
            from scipy import stats
            x = np.linspace(np.min(residuals), np.max(residuals), 100)
            mean, std = np.mean(residuals), np.std(residuals)
            pdf = stats.norm.pdf(x, mean, std)
            plt.plot(x, pdf * len(residuals) * (np.max(residuals) - np.min(residuals)) / bins, 'r-', linewidth=2)
            
        plt.xlabel('Reziduály (Skutočné - Predikované)')
        plt.ylabel('Počet')
        plt.title('Distribúcia reziduálov')
        
        # 3. Scatter plot predikcie vs. vybraný príznak
        plt.subplot(2, 2, 3)
        if X_test_np.shape[1] > feature_index:
            plt.scatter(X_test_np[:, feature_index], y_pred, alpha=0.5, label='Predikcia')
            plt.scatter(X_test_np[:, feature_index], y_test_np, alpha=0.5, label='Skutočnosť')
            plt.xlabel(feature_names[feature_index])
        else:
            plt.text(0.5, 0.5, 'Nedostatok príznakov pre vizualizáciu', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
        
        plt.ylabel('Hodnota')
        plt.title('Predikcia a skutočnosť vs. príznak')
        plt.legend()
        plt.grid(True)
        
        # 4. Q-Q plot reziduálov (pre kontrolu normality)
        plt.subplot(2, 2, 4)
        residuals_sorted = np.sort(residuals)
        n = len(residuals_sorted)
        if n > 1:
            # Teoretické kvantily normálneho rozdelenia
            quantiles = np.arange(1, n + 1) / (n + 1)
            theoretical_quantiles = np.sqrt(2) * np.array([np.log(q / (1 - q)) for q in quantiles])
            
            # Vykreslenie Q-Q plotu
            plt.scatter(theoretical_quantiles, residuals_sorted, alpha=0.5)
            
            # Referenčná čiara
            min_val = min(theoretical_quantiles.min(), residuals_sorted.min())
            max_val = max(theoretical_quantiles.max(), residuals_sorted.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Teoretické kvantily')
            plt.ylabel('Kvantily reziduálov')
            plt.title('Q-Q Plot reziduálov')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Nedostatok dát pre Q-Q plot', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    # Výpis metrík
    print(f"Metriky modelu na testovacej sade:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    # Výpis metrík
    print(f"Metriky modelu na testovacej sade ({segment_name if segment_name else 'Celkovo'}):")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    current_importance = None
    if show_feature_importance and X_test_np.shape[1] > 1 and 'importance' in locals():
        current_importance = importance
        print("\nDôležitosť príznakov:")
        for feature, imp in sorted(current_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {imp:.4f}")

    # Výpočet a výpis separátnych metrík pre kombinované segmenty
    if segment_name and 'Combined' in segment_name and X_test_original_is_for_sale is not None and isinstance(X_test_original_is_for_sale, pd.Series):
        print(f"\nSeparátne metriky pre segment: {segment_name}")
        
        # Zabezpečenie, že X_test_original_is_for_sale má rovnaký index ako y_test
        # a že y_test_np a y_pred majú rovnakú dĺžku ako X_test_original_is_for_sale
        if len(X_test_original_is_for_sale) == len(y_test_np):
            is_for_sale_values = X_test_original_is_for_sale.values

            # Metriky pre 'is_for_sale == 1' (Predaj)
            mask_predaj = (is_for_sale_values == 1)
            if np.sum(mask_predaj) > 0:
                y_test_predaj = y_test_np[mask_predaj]
                y_pred_predaj = y_pred[mask_predaj]
                
                mse_predaj = np.mean((y_pred_predaj - y_test_predaj) ** 2)
                rmse_predaj = np.sqrt(mse_predaj)
                mae_predaj = np.mean(np.abs(y_pred_predaj - y_test_predaj))
                ss_total_predaj = np.sum((y_test_predaj - np.mean(y_test_predaj)) ** 2)
                ss_residual_predaj = np.sum((y_test_predaj - y_pred_predaj) ** 2)
                r2_predaj = 1 - (ss_residual_predaj / ss_total_predaj) if ss_total_predaj > 0 else 0
                
                results['MSE_Predaj_Combined'] = mse_predaj
                results['RMSE_Predaj_Combined'] = rmse_predaj
                results['MAE_Predaj_Combined'] = mae_predaj
                results['R2_Predaj_Combined'] = r2_predaj
                
                print("  Metriky pre 'Predaj' (is_for_sale == 1):")
                print(f"    RMSE: {rmse_predaj:.4f}")
                print(f"    MAE: {mae_predaj:.4f}")
                print(f"    R^2: {r2_predaj:.4f}")
            else:
                print("  Pre 'Predaj' (is_for_sale == 1) neboli nájdené žiadne dáta v testovacej sade.")

            # Metriky pre 'is_for_sale == 0' (Prenájom)
            mask_prenajom = (is_for_sale_values == 0)
            if np.sum(mask_prenajom) > 0:
                y_test_prenajom = y_test_np[mask_prenajom]
                y_pred_prenajom = y_pred[mask_prenajom]
                
                mse_prenajom = np.mean((y_pred_prenajom - y_test_prenajom) ** 2)
                rmse_prenajom = np.sqrt(mse_prenajom)
                mae_prenajom = np.mean(np.abs(y_pred_prenajom - y_test_prenajom))
                ss_total_prenajom = np.sum((y_test_prenajom - np.mean(y_test_prenajom)) ** 2)
                ss_residual_prenajom = np.sum((y_test_prenajom - y_pred_prenajom) ** 2)
                r2_prenajom = 1 - (ss_residual_prenajom / ss_total_prenajom) if ss_total_prenajom > 0 else 0
                
                results['MSE_Prenajom_Combined'] = mse_prenajom
                results['RMSE_Prenajom_Combined'] = rmse_prenajom
                results['MAE_Prenajom_Combined'] = mae_prenajom
                results['R2_Prenajom_Combined'] = r2_prenajom
                
                print("  Metriky pre 'Prenájom' (is_for_sale == 0):")
                print(f"    RMSE: {rmse_prenajom:.4f}")
                print(f"    MAE: {mae_prenajom:.4f}")
                print(f"    R^2: {r2_prenajom:.4f}")
            else:
                print("  Pre 'Prenájom' (is_for_sale == 0) neboli nájdené žiadne dáta v testovacej sade.")
        else:
            print("Chyba: Dĺžka X_test_original_is_for_sale sa nezhoduje s dĺžkou y_test_np. Separátne metriky nebudú vypočítané.")
            
    return results, current_importance
