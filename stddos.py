import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn. feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn. metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
from sklearn.inspection import DecisionBoundaryDisplay
import time
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="MQTT DDoS Detection - Analyse Complète",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS minimal
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius:  5px;
    }
    .stMetric label {
        color: #262730 !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #262730 !important;
    }
    .success-box {
        background-color: #d4edda;
        border:  1px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        margin:  10px 0;
        color: #155724 !important;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding:  10px;
        border-radius: 5px;
        margin: 10px 0;
        color: #856404 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

@st.cache_data
def load_data(file):
    """Charger les données"""
    return pd.read_csv(file)

def create_is_ddos(df):
    """Créer la variable cible is_ddos selon les 3 critères"""
    df['requests_per_ip'] = df. groupby('general_ip')['general_ip'].transform('count')
    df['duplicate_rate_per_ip'] = df.groupby('general_ip')['mqtt_duplicate']. transform('mean')
    df['unique_topics_per_ip'] = df. groupby('general_ip')['mqtt_topic'].transform('nunique')
    
    SEUIL_REQUETES = 200
    SEUIL_DUPLICATE = 0.3
    SEUIL_TOPICS = 5
    
    df['is_ddos'] = (
        (df['requests_per_ip'] > SEUIL_REQUETES) &
        ((df['duplicate_rate_per_ip'] > SEUIL_DUPLICATE) | (df['unique_topics_per_ip'] < SEUIL_TOPICS))
    ).astype(int)
    
    return df

def prepare_temporal_features(df):
    """Extraire features temporelles"""
    df['timestamp_parsed'] = pd.to_datetime(df['@timestamp'], errors='coerce')
    df['hour'] = df['timestamp_parsed']. dt.hour
    df['day_of_week'] = df['timestamp_parsed'].dt.dayofweek
    df['day'] = df['timestamp_parsed'].dt.day
    df['month'] = df['timestamp_parsed'].dt.month
    df['day_name'] = df['timestamp_parsed']. dt.day_name()
    return df

def detect_data_leakage(X, y, threshold=0.95):
    """Détection automatique du data leakage"""
    suspicious = []
    for col in X.columns:
        corr = abs(X[col].corr(y))
        if corr > threshold:
            suspicious.append((col, corr))
    return suspicious

def comprehensive_data_leakage_check(X_train, y_train, X_test, y_test):
    """
    Vérification complète du data leakage avec plusieurs méthodes:
    1. Corrélation excessive avec la cible
    2. Valeurs identiques entre train et test (contamination potentielle)
    3. Distribution statistiquement identique
    4. Features avec variance quasi-nulle
    5. Perfect separation (AUC = 1.0)
    """
    results = {
        'high_correlation': [],
        'train_test_overlap': [],
        'zero_variance': [],
        'perfect_predictors': [],
        'suspicious_distributions': []
    }
    
    # 1. Corrélation excessive avec la cible
    for col in X_train.columns:
        corr = abs(X_train[col].corr(y_train))
        if corr > 0.95:
            results['high_correlation'].append({
                'feature': col,
                'correlation': corr,
                'severity': 'CRITICAL' if corr > 0.99 else 'HIGH'
            })
    
    # 2. Variance nulle ou quasi-nulle
    for col in X_train.columns:
        variance = X_train[col].var()
        if variance < 1e-10:
            results['zero_variance'].append({
                'feature': col,
                'variance': variance
            })
    
    # 3. Perfect predictors (utilisant AUC)
    from sklearn.metrics import roc_auc_score
    for col in X_train.columns:
        try:
            if X_train[col].nunique() > 1:  # Skip constant features
                auc = roc_auc_score(y_train, X_train[col])
                # AUC très proche de 0 ou 1 indique un perfect predictor
                if auc > 0.99 or auc < 0.01:
                    results['perfect_predictors'].append({
                        'feature': col,
                        'auc': auc,
                        'severity': 'CRITICAL'
                    })
        except:
            pass
    
    # 4. Train-Test overlap suspicieux
    for col in X_train.columns:
        train_unique = set(X_train[col].unique())
        test_unique = set(X_test[col].unique())
        
        # Si les valeurs de test recouvrent presque exactement le train
        overlap_ratio = len(train_unique.intersection(test_unique)) / max(len(test_unique), 1)
        
        # Check si les statistiques sont suspicieusement identiques
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()
        train_std = X_train[col].std()
        test_std = X_test[col].std()
        
        # Règle plus stricte pour éviter les faux positifs
        if train_std > 0 and test_std > 0:
            mean_diff_ratio = abs(train_mean - test_mean) / train_std
            std_diff_ratio = abs(train_std - test_std) / train_std
            
            # Distributions quasi identiques (tolérance réduite)
            if mean_diff_ratio < 0.005 and std_diff_ratio < 0.005 and overlap_ratio > 0.995:
                results['suspicious_distributions'].append({
                    'feature': col,
                    'mean_diff_ratio': mean_diff_ratio,
                    'std_diff_ratio': std_diff_ratio,
                    'overlap_ratio': overlap_ratio,
                    'train_mean': train_mean,
                    'test_mean': test_mean,
                    'train_std': train_std,
                    'test_std': test_std,
                    'n_unique_train': len(train_unique),
                    'n_unique_test': len(test_unique)
                })
    
    return results

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st. image("https://img.icons8.com/fluency/96/000000/security-shield-green.png", width=80)
    st.title("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("📁 Chargez merged_data.csv", type=['csv'])
    
    if uploaded_file:
        st.success("✅ Fichier chargé!")
    
    st.markdown("---")
    st.subheader("🎯 Paramètres Globaux")
    test_size = st.slider("Taille test set", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", value=42, min_value=0)
    
    st.markdown("---")
    st.info("💡 Explorez chaque onglet pour une analyse complète!")

# ============================================================
# VÉRIFICATION UPLOAD
# ============================================================

if uploaded_file is None:
    st.title("🛡️ MQTT DDoS Detection - Analyse Complète")
    st.warning("⚠️ Veuillez charger le fichier merged_data.csv dans la sidebar")
    st.info("""
    Cette application implémente l'analyse complète en 6 parties: 
    
    1. **Exploration & Feature Engineering** - EDA complet, analyses temporelles, réseau, MQTT
    2. **Analyse Cible & Corrélations** - Distribution is_ddos, corrélations, SelectKBest
    3. **PCA Complète** - Toutes composantes, cercles de corrélation
    4. **PCA 2D** - Contributions détaillées, pie charts
    5. **Neural Network** - Diagnostic split, data leakage, MLP
    6. **Comparaison Modèles** - 11 modèles (SVM, Logistic, Trees, KNN, RF, MLP, Naive Bayes)
    """)
    st.stop()

# ============================================================
# CHARGEMENT ET PRÉPARATION
# ============================================================

with st.spinner("📊 Chargement et préparation des données..."):
    data = load_data(uploaded_file)
    original_shape = data.shape
    
    # Split train/test initial
    datatrainset, datatestset = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Créer is_ddos
    datatrainset = create_is_ddos(datatrainset)
    datatestset = create_is_ddos(datatestset)
    
    # Features temporelles
    datatrainset = prepare_temporal_features(datatrainset)
    datatestset = prepare_temporal_features(datatestset)

st.success(f"✅ Données chargées:  {original_shape[0]: ,} lignes, {original_shape[1]} colonnes")

# ============================================================
# TABS PRINCIPALES
# ============================================================

tabs = st.tabs([
    "📊 Partie 1: Exploration",
    "📈 Partie 2: Cible & Corrélations",
    "🎯 Partie 3: PCA Complète",
    "📉 Partie 4: PCA 2D",
    "🧠 Partie 5: Neural Network",
    "🏆 Partie 6: Comparaison Modèles",
    "📖 Concepts MQTT"
])

# ============================================================
# PARTIE 1: EXPLORATION & FEATURE ENGINEERING
# ============================================================

with tabs[0]: 
    st.header("📊 Partie 1: Exploration & Feature Engineering")
    
    # Section 1.1: Informations générales
    st.markdown("### 1.1 Informations Générales")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Lignes Train", f"{len(datatrainset):,}")
    with col2:
        st. metric("📝 Lignes Test", f"{len(datatestset):,}")
    with col3:
        st.metric("📋 Colonnes", datatrainset.shape[1])
    with col4:
        st.metric("💾 Mémoire", f"{datatrainset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with st.expander("🔍 Aperçu des données (10 premières lignes)"):
        st.dataframe(datatrainset.head(10), use_container_width=True)
    
    with st.expander("📊 Info du DataFrame"):
        buffer = io.StringIO()
        datatrainset.info(buf=buffer)
        st.text(buffer.getvalue())
    
    # Section 1.2: Types de données
    st.subheader("1.2 Types de Données")
    
    col1, col2 = st. columns([1, 2])
    
    with col1:
        type_counts = datatrainset.dtypes.value_counts()
        st.dataframe(type_counts.to_frame('Count'), use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        type_counts.plot. pie(autopct='%1.1f%%', startangle=90, ax=ax)
        ax.set_ylabel("")
        ax.set_title("Répartition des Types de Données")
        st.pyplot(fig)
    
    # Section 1.3: Valeurs manquantes
    st. subheader("1.3 Analyse des Valeurs Manquantes")
    
    missing = datatrainset.isnull().sum()
    missing_percent = (missing / len(datatrainset)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_percent
    }).sort_values('Missing', ascending=False)
    missing_df = missing_df[missing_df['Missing'] > 0]
    
    if len(missing_df) > 0:
        col1, col2 = st. columns(2)
        
        with col1:
            st. dataframe(missing_df, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            missing_df['Missing'].plot(kind='barh', ax=ax, color='coral', edgecolor='black')
            ax.set_xlabel('Nombre de valeurs manquantes')
            ax.set_title('Valeurs Manquantes par Colonne')
            st.pyplot(fig)
    else:
        st.success("✅ Aucune valeur manquante détectée!")
    
    # Section 1.4: Distributions numériques
    st.subheader("1.4 Distributions des Variables Numériques")
    
    num_cols = datatrainset.select_dtypes(include=['float64', 'int64']).columns. tolist()
    
    if st.checkbox("📊 Afficher les distributions numériques", value=False):
        num_per_row = 3
        num_rows = int(np.ceil(len(num_cols) / num_per_row))
        
        fig, axes = plt.subplots(num_rows, num_per_row, figsize=(15, num_rows * 4))
        axes = axes.flatten() if num_rows > 1 else [axes] if num_rows == 1 else axes
        
        for idx, col in enumerate(num_cols):
            if idx < len(axes):
                datatrainset[col].hist(bins=30, ax=axes[idx], color='skyblue', edgecolor='black')
                axes[idx].set_title(col)
                axes[idx].set_xlabel('')
        
        # Hide unused axes
        for idx in range(len(num_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Section 1.5: Analyse catégorielles
    st.subheader("1.5 Analyse des Variables Catégorielles")
    
    cat_cols = datatrainset.select_dtypes(include=['object', 'bool']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['@timestamp', 'timestamp_parsed']]
    
    st.write(f"**{len(cat_cols)} variables catégorielles détectées**")
    
    if st.checkbox("📊 Afficher l'analyse catégorielle", value=False):
        for col in cat_cols:
            with st.expander(f"🔹 {col}"):
                nunique = datatrainset[col].nunique()
                missing = datatrainset[col]. isna().sum()
                
                st.write(f"- Valeurs uniques: {nunique}")
                st.write(f"- Valeurs manquantes:  {missing}")
                
                if nunique <= 20:
                    value_counts = datatrainset[col].value_counts().head(20)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.dataframe(value_counts.to_frame('Count'), use_container_width=True)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        value_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
                        ax.set_xlabel('Fréquence')
                        ax.set_title(f'Distribution:  {col}')
                        ax.invert_yaxis()
                        st.pyplot(fig)
    
    # Section 1.6: Analyses temporelles
    st.subheader("1.6 Analyses Temporelles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par heure
        if 'hour' in datatrainset.columns:
            hour_counts = datatrainset['hour'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            hour_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_xlabel('Heure du jour')
            ax.set_ylabel('Nombre de messages')
            ax.set_title('Distribution par Heure')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
    
    with col2:
        # Distribution par jour de la semaine
        if 'day_name' in datatrainset.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = datatrainset['day_name'].value_counts()
            day_counts = day_counts.reindex([d for d in day_order if d in day_counts.index])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            day_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
            ax.set_xlabel('Jour de la semaine')
            ax.set_ylabel('Nombre de messages')
            ax.set_title('Distribution par Jour')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
    
    # Timeline
    if 'timestamp_parsed' in datatrainset.columns:
        st.write("**Timeline des messages**")
        date_counts = datatrainset['timestamp_parsed'].dt.date.value_counts().sort_index()
        
        fig = px.line(x=date_counts.index, y=date_counts.values,
                     labels={'x': 'Date', 'y': 'Nombre de messages'},
                     title='Volume de Messages dans le Temps')
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap Jour vs Heure
    if 'day_of_week' in datatrainset.columns and 'hour' in datatrainset.columns:
        st.write("**Heatmap:  Activité par Jour et Heure**")
        
        heatmap_data = datatrainset.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Nombre de messages'})
        ax.set_xlabel('Heure')
        ax.set_ylabel('Jour de la semaine')
        ax.set_title('Heatmap: Jour vs Heure')
        st.pyplot(fig)
    
    # Section 1.7: Analyses réseau
    st.subheader("1.7 Analyses Réseau (IPs, MACs, Devices)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'general_ip' in datatrainset.columns:
            st.metric("🌐 IPs Uniques", datatrainset['general_ip']. nunique())
    
    with col2:
        if 'general_mac' in datatrainset.columns:
            st.metric("📡 MACs Uniques", datatrainset['general_mac'].nunique())
    
    with col3:
        if 'general_device_name' in datatrainset.columns:
            st.metric("🖥️ Devices Uniques", datatrainset['general_device_name'].nunique())
    
    col1, col2 = st. columns(2)
    
    with col1:
        # Top IPs
        if 'general_ip' in datatrainset.columns:
            top_ips = datatrainset['general_ip'].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_ips.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
            ax.set_xlabel('Nombre de requêtes')
            ax.set_title('Top 15 IPs')
            ax.invert_yaxis()
            st.pyplot(fig)
    
    with col2:
        # Distribution requêtes par IP
        if 'general_ip' in datatrainset.columns:
            ip_counts = datatrainset['general_ip']. value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(ip_counts. values, bins=30, color='orange', edgecolor='black')
            ax.set_xlabel('Nombre de requêtes')
            ax.set_ylabel("Nombre d'IPs")
            ax.set_title('Distribution: Requêtes par IP')
            ax.set_yscale('log')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
    
    # Top MACs et Devices
    col1, col2 = st. columns(2)
    
    with col1:
        if 'general_mac' in datatrainset.columns:
            top_macs = datatrainset['general_mac'].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_macs. plot(kind='barh', ax=ax, color='purple', edgecolor='black')
            ax.set_xlabel('Nombre de requêtes')
            ax.set_title('Top 15 MACs')
            ax.invert_yaxis()
            st.pyplot(fig)
    
    with col2:
        if 'general_device_name' in datatrainset. columns:
            top_devices = datatrainset['general_device_name'].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_devices.plot(kind='barh', ax=ax, color='teal', edgecolor='black')
            ax.set_xlabel('Nombre de messages')
            ax.set_title('Top 15 Devices')
            ax.invert_yaxis()
            st.pyplot(fig)
    
    # Section 1.8: Analyses MQTT
    st.subheader("1.8 Analyses MQTT")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'mqtt_topic' in datatrainset.columns:
            st.metric("📬 Topics Uniques", datatrainset['mqtt_topic'].nunique())
    
    with col2:
        if 'mqtt_message_type' in datatrainset.columns:
            st.metric("📝 Types de Messages", datatrainset['mqtt_message_type']. nunique())
    
    with col3:
        if 'mqtt_duplicate' in datatrainset.columns:
            dup_rate = (datatrainset['mqtt_duplicate']. sum() / len(datatrainset)) * 100
            st.metric("🔄 Taux Duplication", f"{dup_rate:.2f}%")
    
    with col4:
        if 'mqtt_retained' in datatrainset.columns:
            ret_rate = (datatrainset['mqtt_retained'].sum() / len(datatrainset)) * 100
            st.metric("💾 Taux Retained", f"{ret_rate:.2f}%")
    
    col1, col2 = st. columns(2)
    
    with col1:
        # Types de messages
        if 'mqtt_message_type' in datatrainset.columns:
            type_counts = datatrainset['mqtt_message_type'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            type_counts.plot. pie(autopct='%1.1f%%', ax=ax, startangle=90)
            ax.set_ylabel('')
            ax.set_title('Types de Messages MQTT')
            st.pyplot(fig)
    
    with col2:
        # Top topics
        if 'mqtt_topic' in datatrainset. columns:
            top_topics = datatrainset['mqtt_topic'].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            top_topics.plot(kind='barh', ax=ax, color='salmon', edgecolor='black')
            ax.set_xlabel('Fréquence')
            ax.set_title('Top 20 Topics MQTT')
            ax.invert_yaxis()
            st.pyplot(fig)
    
    col1, col2 = st. columns(2)
    
    with col1:
        # Messages dupliqués
        if 'mqtt_duplicate' in datatrainset.columns:
            dup_counts = datatrainset['mqtt_duplicate'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            dup_counts.plot(kind='bar', ax=ax, color=['lightgreen', 'lightcoral'], edgecolor='black')
            ax.set_xlabel('Duplicated')
            ax.set_ylabel('Count')
            ax.set_title('Messages Dupliqués')
            ax.set_xticklabels([str(x) for x in dup_counts.index], rotation=0)
            for i, v in enumerate(dup_counts.values):
                ax.text(i, v, f'{v}\n({v/len(datatrainset)*100:.1f}%)', ha='center', va='bottom')
            st.pyplot(fig)
    
    with col2:
        # Messages retained
        if 'mqtt_retained' in datatrainset.columns:
            ret_counts = datatrainset['mqtt_retained'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            ret_counts.plot(kind='bar', ax=ax, color=['lightblue', 'lightyellow'], edgecolor='black')
            ax.set_xlabel('Retained')
            ax.set_ylabel('Count')
            ax.set_title('Messages Retained')
            ax.set_xticklabels([str(x) for x in ret_counts.index], rotation=0)
            for i, v in enumerate(ret_counts.values):
                ax.text(i, v, f'{v}\n({v/len(datatrainset)*100:.1f}%)', ha='center', va='bottom')
            st.pyplot(fig)
    
    # Section 1.9: Création is_ddos
    st.subheader("1.9 Création de la Variable Cible:  is_ddos")
    
    st.info("""
    **Critères de détection DDoS:**
    - Requêtes par IP > 200 **ET**
    - (Taux de duplication > 30% **OU** Topics uniques < 5)
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Normal (0)", datatrainset[datatrainset['is_ddos'] == 0]. shape[0])
    
    with col2:
        st. metric("🚨 DDoS (1)", datatrainset[datatrainset['is_ddos'] == 1]. shape[0])
    
    with col3:
        ratio = datatrainset[datatrainset['is_ddos'] == 0].shape[0] / datatrainset[datatrainset['is_ddos'] == 1].shape[0]
        st.metric("⚖️ Ratio Normal: DDoS", f"{ratio:.2f}: 1")
    
    # Distribution is_ddos
    fig, ax = plt.subplots(figsize=(10, 5))
    datatrainset['is_ddos'].value_counts().plot(kind='bar', ax=ax, 
                                                 color=['green', 'red'], edgecolor='black')
    ax.set_xlabel('is_ddos')
    ax.set_ylabel('Nombre d\'observations')
    ax.set_title('Distribution de la Variable Cible')
    ax.set_xticklabels(['Normal (0)', 'DDoS (1)'], rotation=0)
    for i, v in enumerate(datatrainset['is_ddos']. value_counts().values):
        ax.text(i, v, f'{v: ,}', ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)

# ============================================================
# PARTIE 2: ANALYSE CIBLE & CORRÉLATIONS
# ============================================================

with tabs[1]:
    st.header("📈 Partie 2: Analyse Cible & Corrélations")
    
    # Section 2.1: Distribution is_ddos
    st.subheader("2.1 Distribution de is_ddos")
    
    col1, col2 = st. columns([1, 2])
    
    with col1:
        distrib = datatrainset['is_ddos'].value_counts()
        st.dataframe(distrib.to_frame('Count'), use_container_width=True)
        
        desequilibre = distrib.iloc[0] / distrib.iloc[1]
        st.metric("📊 Ratio de déséquilibre", f"{desequilibre:.2f}: 1")
        
        if desequilibre > 5:
            st.error("⚠️ Déséquilibre extrême - SMOTE fortement conseillé")
        elif desequilibre > 2:
            st.warning("⚠️ Bon candidat pour SMOTE")
        else:
            st.success("✅ Équilibre correct")
    
    with col2:
        fig = px.pie(values=distrib.values, names=['Normal (0)', 'DDoS (1)'],
                    title="Répartition Normal vs DDoS",
                    color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Section 2.2: Crosstabs
    st.subheader("2.2 Crosstabs:  Variables Catégorielles vs is_ddos")
    
    cat_cols_small = [c for c in cat_cols if datatrainset[c].nunique() < 10]
    
    if len(cat_cols_small) > 0:
        selected_cat = st.selectbox("Sélectionnez une variable", cat_cols_small)
        
        cross_tab = pd.crosstab(datatrainset[selected_cat], datatrainset['is_ddos'], normalize='index') * 100
        
        col1, col2 = st. columns([1, 1])
        
        with col1:
            st.write("**Crosstab (en %)**")
            st.dataframe(cross_tab, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            datatrainset_grouped = datatrainset.groupby([selected_cat, 'is_ddos']).size().unstack(fill_value=0)
            datatrainset_grouped.plot(kind='bar', stacked=True, ax=ax, 
                                     color=['lightgreen', 'lightcoral'], edgecolor='black')
            ax.set_xlabel(selected_cat)
            ax.set_ylabel('Nombre d\'observations')
            ax.set_title(f'Distribution de {selected_cat} par Classe DDoS')
            ax.legend(title='is_ddos', labels=['Normal (0)', 'DDoS (1)'])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
    
    # Section 2.3: Analyses temporelles vs DDoS
    st.subheader("2.3 Analyses Temporelles vs DDoS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Messages par heure
        if 'hour' in datatrainset.columns:
            hourly = datatrainset.groupby(['hour', 'is_ddos']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            hourly.plot(kind='bar', stacked=True, ax=ax, 
                       color=['lightgreen', 'lightcoral'], edgecolor='black')
            ax.set_xlabel('Heure')
            ax.set_ylabel('Nombre de messages')
            ax.set_title('Messages par Heure (Normal vs DDoS)')
            ax.legend(title='is_ddos', labels=['Normal (0)', 'DDoS (1)'])
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
    
    with col2:
        # Messages par jour
        if 'day_name' in datatrainset.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily = datatrainset.groupby(['day_name', 'is_ddos']).size().unstack(fill_value=0)
            daily = daily.reindex([d for d in day_order if d in daily.index])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            daily.plot(kind='bar', stacked=True, ax=ax,
                      color=['lightgreen', 'lightcoral'], edgecolor='black')
            ax.set_xlabel('Jour de la semaine')
            ax.set_ylabel('Nombre de messages')
            ax.set_title('Messages par Jour (Normal vs DDoS)')
            ax.legend(title='is_ddos', labels=['Normal (0)', 'DDoS (1)'])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
    
    # Section 2.4: Durée des attaques
    st.subheader("2.4 Durée des Attaques DDoS")
    
    ddos_data = datatrainset[datatrainset['is_ddos'] == 1]
    
    if len(ddos_data) > 0 and 'timestamp_parsed' in ddos_data.columns:
        attack_duration = ddos_data. groupby('general_ip')['timestamp_parsed']. agg(['min', 'max'])
        attack_duration['duration_seconds'] = (attack_duration['max'] - attack_duration['min']).dt.total_seconds()
        attack_duration['duration_minutes'] = attack_duration['duration_seconds'] / 60
        attack_duration['duration_hours'] = attack_duration['duration_minutes'] / 60
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Moyenne (min)", f"{attack_duration['duration_minutes'].mean():.2f}")
        with col2:
            st.metric("⏱️ Médiane (min)", f"{attack_duration['duration_minutes'].median():.2f}")
        with col3:
            st.metric("⏱️ Min (sec)", f"{attack_duration['duration_seconds'].min():.2f}")
        with col4:
            st.metric("⏱️ Max (h)", f"{attack_duration['duration_hours'].max():.2f}")
        
        # Histogrammes
        fig, axes = plt. subplots(1, 3, figsize=(18, 5))
        
        attack_duration['duration_seconds'].hist(bins=50, ax=axes[0], color='coral', edgecolor='black')
        axes[0].set_xlabel('Durée (sec)')
        axes[0].set_ylabel('Fréquence')
        axes[0].set_title('Durée des attaques (secondes)')
        axes[0].grid(alpha=0.3)
        
        attack_duration['duration_minutes'].hist(bins=50, ax=axes[1], color='orangered', edgecolor='black')
        axes[1].set_xlabel('Durée (min)')
        axes[1].set_ylabel('Fréquence')
        axes[1].set_title('Durée des attaques (minutes)')
        axes[1].grid(alpha=0.3)
        
        attack_duration['duration_hours'].hist(bins=50, ax=axes[2], color='darkred', edgecolor='black')
        axes[2].set_xlabel('Durée (h)')
        axes[2].set_ylabel('Fréquence')
        axes[2].set_title('Durée des attaques (heures)')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Section 2.5: Boxplots
    st.subheader("2.5 Boxplots des Variables Numériques")
    
    if st.checkbox("📊 Afficher les boxplots", value=False):
        num_cols_clean = [c for c in num_cols if c != 'is_ddos']
        
        num_per_row = 3
        num_rows = int(np.ceil(len(num_cols_clean) / num_per_row))
        
        fig, axes = plt.subplots(num_rows, num_per_row, figsize=(15, num_rows * 4))
        axes = axes.flatten() if num_rows > 1 else [axes]
        
        for idx, col in enumerate(num_cols_clean):
            if idx < len(axes):
                datatrainset. boxplot(column=col, ax=axes[idx])
                axes[idx].set_title(f'Boxplot:  {col}')
        
        for idx in range(len(num_cols_clean), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Section 2.6: Encodage & Matrice de Corrélation
    st.subheader("2.6 Encodage & Matrice de Corrélation")
    
    if st.button("🚀 Calculer Matrice de Corrélation (peut être long)"):
        with st.spinner("⏳ Encodage et calcul de la corrélation..."):
            # Copie pour encodage
            data_encoded = datatrainset. copy()
            
            # Encoder les catégorielles
            le = LabelEncoder()
            cat_to_encode = [c for c in cat_cols if c in data_encoded.columns]
            
            for col in cat_to_encode: 
                try:
                    data_encoded[col] = data_encoded[col].fillna('MISSING')
                    data_encoded[col + '_encoded'] = le.fit_transform(data_encoded[col]. astype(str))
                except:
                    pass
            
            # Sélectionner colonnes numériques
            numeric_cols_for_corr = data_encoded.select_dtypes(include=['float64', 'int64', 'int32', 'bool']).columns
            data_for_corr = data_encoded[numeric_cols_for_corr]
            
            # Supprimer colonnes problématiques
            cols_to_drop = ['timestamp_parsed'] if 'timestamp_parsed' in data_for_corr.columns else []
            if len(cols_to_drop) > 0:
                data_for_corr = data_for_corr.drop(columns=cols_to_drop)
            
            st.info(f"📊 Calcul de la corrélation pour {data_for_corr.shape[1]} variables...")
            
            # Calculer corrélation
            corr_matrix = data_for_corr.corr()
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(20, 16))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       linewidths=0.1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Matrice de Corrélation Complète", fontsize=16, fontweight='bold')
            st.pyplot(fig)
            
            # Corrélation avec is_ddos
            if 'is_ddos' in corr_matrix.columns:
                st.subheader("📊 Corrélation avec is_ddos")
                
                ddos_corr = corr_matrix['is_ddos']. sort_values(ascending=False)
                
                col1, col2 = st. columns(2)
                
                with col1:
                    st.write("**Top 15 Corrélations Positives**")
                    st.dataframe(ddos_corr.head(15).to_frame('Correlation'), use_container_width=True)
                
                with col2:
                    st.write("**Top 15 Corrélations Négatives**")
                    st.dataframe(ddos_corr.tail(15).to_frame('Correlation'), use_container_width=True)
                
                # Visualisation
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                top_pos = ddos_corr[ddos_corr > 0].sort_values(ascending=False).head(15)
                axes[0].barh(range(len(top_pos)), top_pos.values, color='red', alpha=0.7, edgecolor='black')
                axes[0].set_yticks(range(len(top_pos)))
                axes[0]. set_yticklabels(top_pos.index, fontsize=9)
                axes[0].set_xlabel('Corrélation')
                axes[0].set_title('Top 15 Corrélations Positives avec is_ddos')
                axes[0].invert_yaxis()
                axes[0].grid(axis='x', alpha=0.3)
                
                top_neg = ddos_corr[ddos_corr < 0].sort_values(ascending=True).head(15)
                axes[1].barh(range(len(top_neg)), top_neg.values, color='blue', alpha=0.7, edgecolor='black')
                axes[1].set_yticks(range(len(top_neg)))
                axes[1].set_yticklabels(top_neg.index, fontsize=9)
                axes[1].set_xlabel('Corrélation')
                axes[1].set_title('Top 15 Corrélations Négatives avec is_ddos')
                axes[1].invert_yaxis()
                axes[1].grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Section 2.7: Feature Selection (SelectKBest)
    st.subheader("2.7 Sélection de Features (SelectKBest)")
    
    if st.button("🚀 Lancer Feature Selection"):
        with st.spinner("⏳ Sélection des features en cours..."):
            # Préparer X et y
            features_for_selection = [c for c in num_cols if c != 'is_ddos']
            X = datatrainset[features_for_selection]. fillna(datatrainset[features_for_selection].median())
            y = datatrainset['is_ddos']
            
            # Supprimer colonnes constantes
            X = X.loc[:, X.nunique() > 1]
            
            st.info(f"📊 {X.shape[1]} features analysées")
            
            # Méthode du coude
            scores = []
            k_range = range(1, min(X.shape[1] + 1, 31))
            
            progress = st.progress(0)
            for idx, k in enumerate(k_range):
                selector = SelectKBest(f_classif, k=k)
                X_selected = selector.fit_transform(X, y)
                
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                
                cv_scores = cross_val_score(
                    LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
                    X_selected, y, cv=5, scoring='accuracy', n_jobs=-1
                )
                scores.append(cv_scores.mean())
                progress.progress((idx + 1) / len(k_range))
            
            # Trouver optimal k
            optimal_k = k_range[scores.index(max(scores))]
            
            st.success(f"✅ Nombre optimal de features: **{optimal_k}** (Accuracy: {max(scores):.4f})")
            
            # Graphique Elbow
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(k_range, scores, marker='o', linestyle='--', color='b', linewidth=2, markersize=6)
            ax.axvline(optimal_k, color='red', linestyle=':', linewidth=2, label=f'Optimal k = {optimal_k}')
            ax.axhline(max(scores), color='green', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_xlabel('Nombre de Features (k)')
            ax.set_ylabel('Cross-Validation Accuracy')
            ax.set_title('Méthode du Coude pour la Sélection de Features')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Sélection finale
            selector = SelectKBest(f_classif, k=optimal_k)
            selector.fit(X, y)
            
            selected_features = X.columns[selector.get_support()]
            feature_scores = selector.scores_[selector.get_support()]
            
            # DataFrame résultats
            results_df = pd.DataFrame({
                'Feature': selected_features,
                'F-Score': feature_scores
            }).sort_values('F-Score', ascending=False)
            
            col1, col2 = st. columns([1, 1])
            
            with col1:
                st. write(f"**Top {optimal_k} Features Sélectionnées**")
                st.dataframe(results_df, use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(results_df['Feature'], results_df['F-Score'], color='skyblue', edgecolor='black')
                ax.set_xlabel('ANOVA F-value')
                ax.set_ylabel('Features')
                ax.set_title(f'Top {optimal_k} Features (F-Scores)')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)# ============================================================
# PARTIE 3: PCA COMPLÈTE
# ============================================================

with tabs[2]:
    st.header("🎯 Partie 3: PCA Complète")
    
    st.info("Cette section applique la PCA avec toutes les composantes principales")
    
    if st.button("🚀 Lancer PCA Complète"):
        with st.spinner("⏳ Calcul de la PCA complète..."):
            # Préparer les données
            features_for_pca = [c for c in num_cols if c != 'is_ddos']
            X = datatrainset[features_for_pca]. fillna(datatrainset[features_for_pca].median())
            y = datatrainset['is_ddos']
            
            # Supprimer colonnes constantes
            X = X.loc[:, X.nunique() > 1]
            
            # Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            st.success(f"✅ {X.shape[1]} features standardisées")
            
            # PCA complète
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            # Section 3.1: Variance expliquée
            st.subheader("3.1 Variance Expliquée")
            
            variance_ratio = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratio)
            
            # Nombre de composantes pour différents seuils
            n_80 = np.argmax(cumulative_variance >= 0.80) + 1
            n_90 = np.argmax(cumulative_variance >= 0.90) + 1
            n_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Pour 80% variance", f"{n_80} composantes")
            with col2:
                st.metric("📊 Pour 90% variance", f"{n_90} composantes")
            with col3:
                st.metric("📊 Pour 95% variance", f"{n_95} composantes")
            
            # Afficher les 10 premières
            st.write("**Variance par Composante (10 premières)**")
            var_df = pd.DataFrame({
                'Composante': [f'PC{i+1}' for i in range(min(10, len(variance_ratio)))],
                'Variance (%)': variance_ratio[:10] * 100,
                'Variance Cumulée (%)': cumulative_variance[:10] * 100
            })
            st.dataframe(var_df, use_container_width=True)
            
            # Section 3.2: Scree Plot
            st.subheader("3.2 Scree Plot & Variance Cumulée")
            
            col1, col2 = st. columns(2)
            
            with col1:
                # Variance par composante
                n_show = min(30, len(variance_ratio))
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(1, n_show + 1), variance_ratio[:n_show] * 100,
                      color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_xlabel('Composante Principale')
                ax.set_ylabel('Variance Expliquée (%)')
                ax.set_title('Scree Plot - Variance par Composante')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Variance cumulée
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, n_show + 1), cumulative_variance[:n_show] * 100,
                       marker='o', linewidth=2, markersize=4, color='darkred')
                ax.axhline(80, color='green', linestyle='--', linewidth=1.5, label='80%')
                ax.axhline(90, color='orange', linestyle='--', linewidth=1.5, label='90%')
                ax.axhline(95, color='red', linestyle='--', linewidth=1.5, label='95%')
                ax.set_xlabel('Nombre de Composantes')
                ax.set_ylabel('Variance Cumulée (%)')
                ax.set_title('Variance Expliquée Cumulée')
                ax. legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            # Éboulis des valeurs propres
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.bar(np.arange(n_show) + 1, variance_ratio[:n_show],
                  color='steelblue', edgecolor='black', alpha=0.7, label='Variance individuelle')
            ax.plot(np.arange(n_show) + 1, cumulative_variance[:n_show],
                   c="red", marker='o', linewidth=2, markersize=6, label='Variance cumulée')
            ax.set_xlabel("Rang de l'axe d'inertie", fontsize=12, fontweight='bold')
            ax.set_ylabel("Pourcentage d'inertie", fontsize=12, fontweight='bold')
            ax.set_title("Éboulis des valeurs propres", fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            # Section 3.3: Projection PC1 vs PC2
            st. subheader("3.3 Projection sur PC1 et PC2")
            
            # Transformation
            X_pca = pca_full.transform(X_scaled)
            
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'is_ddos': y. map({0: 'Normal', 1: 'DDoS'})
            })
            
            col1, col2 = st. columns(2)
            
            with col1:
                # Sans coloration
                fig, ax = plt. subplots(figsize=(10, 8))
                ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, s=20, color='steelblue')
                ax. set_xlabel(f'PC1 ({variance_ratio[0]*100:.2f}%)', fontweight='bold')
                ax.set_ylabel(f'PC2 ({variance_ratio[1]*100:.2f}%)', fontweight='bold')
                ax.set_title('Projection sur PC1 et PC2')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Avec coloration is_ddos
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['green' if x == 'Normal' else 'red' for x in pca_df['is_ddos']]
                ax. scatter(pca_df['PC1'], pca_df['PC2'], c=colors, alpha=0.5, s=20)
                ax.set_xlabel(f'PC1 ({variance_ratio[0]*100:.2f}%)', fontweight='bold')
                ax.set_ylabel(f'PC2 ({variance_ratio[1]*100:.2f}%)', fontweight='bold')
                ax.set_title('Projection colorée par is_ddos')
                
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='green', label='Normal (0)'),
                                  Patch(facecolor='red', label='DDoS (1)')]
                ax.legend(handles=legend_elements)
                ax.grid(alpha=0.3)
                st. pyplot(fig)
            
            # Version Plotly interactive
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='is_ddos',
                           color_discrete_map={'Normal': 'green', 'DDoS': 'red'},
                           title=f'Projection PCA Interactive (PC1: {variance_ratio[0]*100:.1f}%, PC2: {variance_ratio[1]*100:.1f}%)',
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            
            # Section 3.4: Biplot avec vecteurs
            st.subheader("3.4 Biplot (Observations + Variables)")
            
            # Loadings
            loadings = pca_full.components_. T * np.sqrt(pca_full.explained_variance_)
            loading_df = pd.DataFrame(
                loadings[: , :2],
                columns=['PC1', 'PC2'],
                index=X.columns
            )
            loading_df['contribution'] = np.sqrt(loading_df['PC1']**2 + loading_df['PC2']**2)
            loading_df = loading_df.sort_values('contribution', ascending=False)
            
            # Top 10 variables
            top_10_vars = loading_df.head(10)
            
            st.write("**Top 10 Variables Contribuant le Plus**")
            st.dataframe(top_10_vars, use_container_width=True)
            
            # Biplot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Scatter des observations
            colors = ['green' if x == 'Normal' else 'red' for x in pca_df['is_ddos']]
            ax.scatter(pca_df['PC1'], pca_df['PC2'], c=colors, alpha=0.3, s=10)
            
            # Vecteurs des top 10 variables
            for var in top_10_vars.index:
                ax.arrow(0, 0, loading_df. loc[var, 'PC1']*3, loading_df.loc[var, 'PC2']*3,
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
                ax.text(loading_df.loc[var, 'PC1']*3.2, loading_df.loc[var, 'PC2']*3.2,
                       var, fontsize=8, color='darkblue', fontweight='bold')
            
            ax.set_xlabel(f'PC1 ({variance_ratio[0]*100:.2f}%)', fontweight='bold')
            ax.set_ylabel(f'PC2 ({variance_ratio[1]*100:.2f}%)', fontweight='bold')
            ax.set_title('Biplot:  Observations et Variables', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            st.pyplot(fig)
            
            # Section 3.5: Cercles de Corrélation
            st.subheader("3.5 Cercles de Corrélation")
            
            # Fonction cercle de corrélation
            def plot_correlation_circle(components, feature_names, pc1=0, pc2=1, variance_ratio=None):
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Cercle
                circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', linewidth=2)
                ax.add_artist(circle)
                
                # Flèches (limiter à 20 pour lisibilité)
                n_features = min(20, len(feature_names))
                for i in range(n_features):
                    ax.arrow(0, 0, components[pc1, i], components[pc2, i],
                            head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
                    ax.text(components[pc1, i]*1.15, components[pc2, i]*1.15,
                           feature_names[i], fontsize=8, ha='center', va='center')
                
                ax.set_xlim(-1.3, 1.3)
                ax.set_ylim(-1.3, 1.3)
                ax.axhline(0, color='black', linewidth=0.5)
                ax.axvline(0, color='black', linewidth=0.5)
                
                if variance_ratio is not None: 
                    ax.set_xlabel(f'PC{pc1+1} ({variance_ratio[pc1]*100:.1f}%)', fontweight='bold')
                    ax.set_ylabel(f'PC{pc2+1} ({variance_ratio[pc2]*100:.1f}%)', fontweight='bold')
                else:
                    ax. set_xlabel(f'PC{pc1+1}', fontweight='bold')
                    ax.set_ylabel(f'PC{pc2+1}', fontweight='bold')
                
                ax.set_title(f'Cercle de Corrélation (PC{pc1+1} vs PC{pc2+1})', fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(alpha=0.3)
                
                return fig
            
            # Option 1: Cercle PC1 vs PC2
            st.write("**Option 1: Cercle PC1 vs PC2 (Top 20 variables)**")
            fig = plot_correlation_circle(pca_full.components_, X.columns. tolist(), 0, 1, variance_ratio)
            st.pyplot(fig)
            
            # Option 2: Top 15 variables les plus contributives
            st.write("**Option 2: Top 15 Variables les Plus Contributives**")
            
            top_15_indices = loading_df.head(15).index
            top_15_idx = [X.columns. tolist().index(var) for var in top_15_indices]
            
            fig, ax = plt.subplots(figsize=(12, 12))
            circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', linewidth=2)
            ax.add_artist(circle)
            
            colors_rainbow = plt.cm.rainbow(np.linspace(0, 1, len(top_15_indices)))
            
            for idx, i in enumerate(top_15_idx):
                x, y = pca_full.components_[0, i], pca_full. components_[1, i]
                ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05,
                        fc=colors_rainbow[idx], ec=colors_rainbow[idx], linewidth=2, alpha=0.7)
                ax.text(x*1.15, y*1.15, X.columns[i], fontsize=10, ha='center', va='center',
                       color=colors_rainbow[idx], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)', fontweight='bold')
            ax.set_ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)', fontweight='bold')
            ax.set_title('Cercle de Corrélation - Top 15 Variables', fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Option 3: Plusieurs plans factoriels
            st.write("**Option 3: Autres Plans Factoriels**")
            
            n_components = pca_full.components_.shape[0]
            
            if n_components >= 3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # PC1 vs PC3
                    fig = plot_correlation_circle(pca_full.components_, X.columns.tolist(), 0, 2, variance_ratio)
                    st.pyplot(fig)
                
                with col2:
                    # PC2 vs PC3
                    fig = plot_correlation_circle(pca_full.components_, X.columns.tolist(), 1, 2, variance_ratio)
                    st.pyplot(fig)
            else:
                st.info(f"ℹ️ Seulement {n_components} composantes disponibles. Plans factoriels supplémentaires non disponibles.")
            
            # Sauvegarder dans session state
            st.session_state['pca_full'] = pca_full
            st.session_state['X_pca'] = X_pca
            st.session_state['scaler_pca'] = scaler
            st.session_state['features_pca'] = X.columns. tolist()

# ============================================================
# PARTIE 4: PCA 2D
# ============================================================

with tabs[3]:
    st.header("📉 Partie 4: PCA 2D & Contributions Détaillées")
    
    st.info("Cette section se concentre sur les 2 premières composantes principales avec analyse détaillée des contributions")
    
    if st.button("🚀 Lancer PCA 2D"):
        with st.spinner("⏳ Calcul PCA 2D..."):
            # Préparer les données
            features_for_pca = [c for c in num_cols if c != 'is_ddos']
            X = datatrainset[features_for_pca]. fillna(datatrainset[features_for_pca].median())
            y = datatrainset['is_ddos']
            
            X = X.loc[: , X.nunique() > 1]
            
            # Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA 2D
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            
            st.success(f"✅ PCA 2D appliquée sur {X.shape[1]} features")
            
            # Section 4.1: Variance expliquée
            st.subheader("4.1 Variance Expliquée")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 PC1", f"{pca_2d.explained_variance_ratio_[0]*100:.2f}%")
            with col2:
                st.metric("📊 PC2", f"{pca_2d.explained_variance_ratio_[1]*100:.2f}%")
            with col3:
                total_var = pca_2d.explained_variance_ratio_.sum()
                st.metric("📊 Total (PC1+PC2)", f"{total_var*100:.2f}%")
            
            # Section 4.2: Calcul des loadings (contributions)
            st.subheader("4.2 Contributions des Features")
            
            loadings = pd.DataFrame(
                pca_2d.components_,
                columns=X.columns,
                index=['PC1', 'PC2']
            )
            
            # Extraire loadings
            pc1_loadings = loadings. loc["PC1"]
            pc2_loadings = loadings.loc["PC2"]
            
            # Contributions en pourcentage
            pc1_percent = (np.abs(pc1_loadings) / np.abs(pc1_loadings).sum()) * 100
            pc2_percent = (np.abs(pc2_loadings) / np.abs(pc2_loadings).sum()) * 100
            
            pc1_percent = pc1_percent.sort_values(ascending=False)
            pc2_percent = pc2_percent.sort_values(ascending=False)
            
            col1, col2 = st. columns(2)
            
            with col1:
                st. write("**Top 15 Contributions à PC1**")
                st.dataframe(pc1_percent.head(15).to_frame('Contribution (%)'), use_container_width=True)
            
            with col2:
                st.write("**Top 15 Contributions à PC2**")
                st. dataframe(pc2_percent. head(15).to_frame('Contribution (%)'), use_container_width=True)
            
            # Section 4.3: Pie Charts
            st.subheader("4.3 Pie Charts - Contributions")
            
            col1, col2 = st. columns(2)
            
            with col1:
                # PC1 Pie
                top_15_pc1 = pc1_percent.head(15)
                fig, ax = plt.subplots(figsize=(10, 10))
                colors_green = plt.cm. Greens(np.linspace(0.3, 1, len(top_15_pc1)))
                ax.pie(top_15_pc1, labels=top_15_pc1.index, autopct='%1.1f%%',
                      startangle=140, colors=colors_green, textprops={'fontsize': 9})
                ax.set_title('Contributions à PC1 (Top 15)', fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                # PC2 Pie
                top_15_pc2 = pc2_percent.head(15)
                fig, ax = plt. subplots(figsize=(10, 10))
                colors_blue = plt.cm.Blues(np.linspace(0.3, 1, len(top_15_pc2)))
                ax. pie(top_15_pc2, labels=top_15_pc2.index, autopct='%1.1f%%',
                      startangle=140, colors=colors_blue, textprops={'fontsize': 9})
                ax.set_title('Contributions à PC2 (Top 15)', fontweight='bold')
                st.pyplot(fig)
            
            # Section 4.4: Bar Charts
            st.subheader("4.4 Bar Charts - Contributions")
            
            col1, col2 = st. columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 8))
                top_15_pc1.plot(kind='barh', ax=ax, color='green', edgecolor='black')
                ax.set_xlabel('Contribution (%)', fontweight='bold')
                ax.set_title('Top 15 Contributions à PC1', fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 8))
                top_15_pc2.plot(kind='barh', ax=ax, color='blue', edgecolor='black')
                ax.set_xlabel('Contribution (%)', fontweight='bold')
                ax.set_title('Top 15 Contributions à PC2', fontweight='bold')
                ax. invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            # Section 4.5: Scatter Plot
            st.subheader("4.5 Projection 2D (PC1 vs PC2)")
            
            pca_df = pd.DataFrame({
                'PC1': X_pca_2d[:, 0],
                'PC2': X_pca_2d[:, 1],
                'is_ddos': y. map({0: 'Normal', 1: 'DDoS'})
            })
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            normal = pca_df[pca_df['is_ddos'] == 'Normal']
            ddos = pca_df[pca_df['is_ddos'] == 'DDoS']
            
            ax.scatter(normal['PC1'], normal['PC2'], c='green', label='Normal (0)',
                      alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            ax.scatter(ddos['PC1'], ddos['PC2'], c='red', label='DDoS (1)',
                      alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)', fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)', fontweight='bold')
            ax.set_title('Projection PCA:  Normal vs DDoS', fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Section 4.6: Heatmap des Loadings
            st.subheader("4.6 Heatmap des Loadings (Top 20)")
            
            # Contribution totale
            total_contribution = (np.abs(pc1_loadings) + np.abs(pc2_loadings)).sort_values(ascending=False)
            top_20_features = total_contribution.head(20).index
            
            loadings_top20 = loadings[top_20_features]. T
            
            fig, ax = plt.subplots(figsize=(10, 12))
            sns.heatmap(loadings_top20, annot=True, cmap='RdYlGn', center=0,
                       fmt='.3f', linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Heatmap des Loadings (Top 20 Features)', fontweight='bold')
            ax.set_xlabel('Composantes Principales', fontweight='bold')
            ax.set_ylabel('Features', fontweight='bold')
            st.pyplot(fig)
            
            # Résumé
            st.subheader("📊 Résumé")
            
            # Build top features list dynamically
            pc1_top = "\n".join([f"{i+1}. {pc1_percent.index[i]}: {pc1_percent.values[i]:.2f}%" 
                                 for i in range(min(3, len(pc1_percent)))])
            pc2_top = "\n".join([f"{i+1}. {pc2_percent.index[i]}: {pc2_percent.values[i]:.2f}%" 
                                 for i in range(min(3, len(pc2_percent)))])
            
            st.write(f"""
            - **Features analysées**: {X.shape[1]}
            - **Variance PC1**: {pca_2d.explained_variance_ratio_[0]*100:.2f}%
            - **Variance PC2**: {pca_2d.explained_variance_ratio_[1]*100:.2f}%
            - **Variance totale**: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%
            
            **Top Features PC1**:
            {pc1_top}
            
            **Top Features PC2**:
            {pc2_top}
            """)# ============================================================
# PARTIE 5:  NEURAL NETWORK + DIAGNOSTIC
# ============================================================

with tabs[4]:
    st.header("🧠 Partie 5: Neural Network & Diagnostic Complet")
    
    st.info("""
    Cette section inclut: 
    - Diagnostic du split (stratification)
    - Encodage automatique
    - Standardisation
    - **🔍 Diagnostic Data Leakage Complet:**
        - Corrélation excessive avec la cible
        - Detection de perfect predictors (AUC ≈ 1.0)
        - Features à variance nulle
        - Contamination train/test
    - Entraînement MLP Classifier
    """)
    
    # Configuration MLP
    st.subheader("⚙️ Configuration Neural Network")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hidden_layers = st.selectbox("Architecture", 
                                     [(100,), (100, 50), (150, 100, 50)],
                                     format_func=lambda x: str(x))
    with col2:
        max_iter = st.slider("Max Iterations", 100, 500, 300, 50)
    with col3:
        early_stopping = st.checkbox("Early Stopping", value=True)
    
    if st.button("🚀 Lancer Entraînement Neural Network", type="primary"):
        with st.spinner("⏳ Préparation et entraînement..."):
            
            # Section 5.1: Recombiner et split stratifié
            st.subheader("5.1 Diagnostic du Split")
            
            # Vérifier datatestset
            st.write("**Distribution dans datatestset (avant correction)**:")
            test_distrib = datatestset['is_ddos'].value_counts()
            st.write(test_distrib)
            
            if datatestset['is_ddos']. nunique() == 1:
                st.error("🚨 PROBLÈME: datatestset n'a qu'UNE classe → Split non stratifié!")
                st.info("✅ SOLUTION: Recombination et split stratifié")
            
            # Recombiner
            data_complete = pd.concat([datatrainset, datatestset], ignore_index=True)
            
            st.success(f"✅ Dataset complet: {data_complete.shape[0]} lignes")
            
            # Préparer features
            features_numeriques = [
                'mqtt_qos', 'mqtt_duplicate', 'mqtt_retained',
                'requests_per_ip', 'duplicate_rate_per_ip', 'unique_topics_per_ip',
                'hour', 'day_of_week', 'day', 'month'
            ]
            
            features_disponibles = [f for f in features_numeriques if f in data_complete.columns]
            
            st.write(f"**Features sélectionnées**:  {len(features_disponibles)}")
            st.write(features_disponibles)
            
            X_complete = data_complete[features_disponibles]. copy()
            y_complete = data_complete['is_ddos']. astype(int)
            
            # Nettoyage
            X_complete = X_complete.fillna(X_complete.median())
            num_cols_clean = X_complete.select_dtypes(include=[np.number]).columns
            X_complete[num_cols_clean] = X_complete[num_cols_clean].replace([np.inf, -np. inf], np.nan)
            X_complete[num_cols_clean] = X_complete[num_cols_clean].fillna(X_complete[num_cols_clean].median())
            
            # SPLIT STRATIFIÉ
            X_train, X_test, y_train, y_test = train_test_split(
                X_complete, y_complete,
                test_size=test_size,
                random_state=random_state,
                stratify=y_complete
            )
            
            st. success(f"✅ Split stratifié effectué!")
            
            col1, col2 = st. columns(2)
            
            with col1:
                st. write(f"**Train**: {len(y_train)} échantillons")
                st. write(y_train.value_counts())
            
            with col2:
                st.write(f"**Test**: {len(y_test)} échantillons")
                st.write(y_test. value_counts())
            
            # Section 5.2: Encodage
            st.subheader("5.2 Encodage des Variables")
            
            le_dict = {}
            cat_cols_to_encode = X_train.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols_to_encode) > 0:
                st.write(f"Encodage de {len(cat_cols_to_encode)} colonnes catégorielles...")
                
                for col in cat_cols_to_encode: 
                    le = LabelEncoder()
                    X_train[col] = le.fit_transform(X_train[col]. fillna('MISSING').astype(str))
                    le_dict[col] = le
                
                X_train = X_train.select_dtypes(include=[np.number])
                
                for col in X_test.select_dtypes(include=['object', 'category']).columns:
                    if col in le_dict:
                        vals = X_test[col].fillna('MISSING').astype(str)
                        mask = ~vals.isin(le_dict[col].classes_)
                        vals[mask] = 'MISSING'
                        X_test[col] = le_dict[col].transform(vals)
                
                X_test = X_test.select_dtypes(include=[np. number])
                st.success("✅ Encodage terminé")
            
            # Aligner colonnes
            for col in set(X_train.columns) - set(X_test.columns):
                X_test[col] = 0
            X_test = X_test[X_train.columns]
            
            st.success(f"✅ {X_train.shape[1]} features alignées")
            
            # Section 5.3: Standardisation
            st.subheader("5.3 Standardisation")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            st. success("✅ Standardisation terminée (moyenne=0, std=1)")
            
            # Section 5.4: Diagnostic Data Leakage
            st.subheader("5.4 🔍 Diagnostic Data Leakage Complet")
            
            st.info("""
            **Vérifications effectuées:**
            - ✅ Corrélation excessive avec la cible (> 95%)
            - ✅ Features avec variance nulle
            - ✅ Perfect predictors (AUC ≈ 1.0)
            - ✅ Distributions train/test suspicieusement identiques
            """)
            
            # Utiliser la vérification basique d'abord (legacy)
            suspicious = detect_data_leakage(X_train, y_train, threshold=0.95)
            
            # Puis la vérification complète
            leakage_results = comprehensive_data_leakage_check(X_train, y_train, X_test, y_test)
            
            # 0. Traiter immédiatement la variance nulle (nettoyage, pas fuite)
            zero_var_features = [item['feature'] for item in leakage_results['zero_variance']]
            if zero_var_features:
                st.info(f"ℹ️ Suppression automatique de {len(zero_var_features)} features à variance nulle (bruit inutile, pas une fuite):")
                st.write(zero_var_features)
                X_train = X_train.drop(columns=zero_var_features, errors='ignore')
                X_test = X_test.drop(columns=zero_var_features, errors='ignore')
                X_train_scaled = np.delete(X_train_scaled, [X_train.columns.tolist().index(c) for c in zero_var_features if c in X_train.columns], axis=1)
                X_test_scaled = np.delete(X_test_scaled, [X_test.columns.tolist().index(c) for c in zero_var_features if c in X_test.columns], axis=1)
                st.success(f"✅ Variance nulle nettoyée. Nouveau shape: {X_train.shape}")
            
            # Recalculer après éventuelle suppression
            leakage_results = comprehensive_data_leakage_check(X_train, y_train, X_test, y_test)
            
            # Afficher les résultats (hors variance nulle déjà traitée)
            total_issues = (len(leakage_results['high_correlation']) + 
                          len(leakage_results['perfect_predictors']) +
                          len(leakage_results['suspicious_distributions']))
            
            if total_issues == 0:
                st.success("✅ Aucun data leakage évident détecté! Le dataset semble propre.")
            else:
                st.error(f"🚨 {total_issues} problèmes potentiels de data leakage détectés!")
                
                # 1. High Correlation
                if leakage_results['high_correlation']:
                    st.warning(f"⚠️ **{len(leakage_results['high_correlation'])} Features avec corrélation excessive**")
                    df_corr = pd.DataFrame(leakage_results['high_correlation'])
                    st.dataframe(df_corr.style.background_gradient(subset=['correlation'], cmap='Reds'), 
                               use_container_width=True)
                
                # 2. Perfect Predictors
                if leakage_results['perfect_predictors']:
                    st.error(f"🚨 **{len(leakage_results['perfect_predictors'])} Perfect Predictors détectés (CRITIQUE)**")
                    df_perfect = pd.DataFrame(leakage_results['perfect_predictors'])
                    st.dataframe(df_perfect.style.background_gradient(subset=['auc'], cmap='Reds'), 
                               use_container_width=True)
                    st.warning("💡 Ces features prédisent parfaitement la cible - **Data Leakage probable!**")
                
                # 3. Suspicious Distributions
                if leakage_results['suspicious_distributions']:
                    st.warning(f"⚠️ **{len(leakage_results['suspicious_distributions'])} Features avec distributions train/test quasi identiques** (seuil strict: Δmoy/std < 0.5%, overlap > 99.5%)")
                    df_dist = pd.DataFrame(leakage_results['suspicious_distributions'])
                    st.dataframe(df_dist, use_container_width=True)
                    st.info("💡 Cela peut indiquer une contamination train/test, mais le seuil est très strict. Vérifiez si c'est attendu (mêmes distributions) ou si le split doit être régénéré.")
                    suspect_dist_features = [item['feature'] for item in leakage_results['suspicious_distributions'] if 'feature' in item]
                    if suspect_dist_features:
                        with st.expander("Actions possibles (optionnelles)"):
                            st.markdown("- Régénérer le split avec un `random_state` différent si la contamination est suspectée\n- Vérifier que la date/temps n'est pas partagée entre train/test\n- Supprimer manuellement ces features si elles codent une fuite temporelle")
                            drop_suspicious = st.checkbox("Supprimer ces features (optionnel)", value=False)
                            if drop_suspicious:
                                X_train = X_train.drop(columns=suspect_dist_features, errors='ignore')
                                X_test = X_test.drop(columns=suspect_dist_features, errors='ignore')
                                scaler_new_dist = StandardScaler()
                                X_train_scaled = scaler_new_dist.fit_transform(X_train)
                                X_test_scaled = scaler_new_dist.transform(X_test)
                                st.success(f"✅ Features supprimées (optionnel). Nouveau shape: {X_train.shape}")
                
                # Décider quelles features supprimer (critiques seulement)
                features_to_remove = set()
                
                for item in leakage_results['high_correlation']:
                    if item['severity'] == 'CRITICAL':
                        features_to_remove.add(item['feature'])
                
                for item in leakage_results['perfect_predictors']:
                    features_to_remove.add(item['feature'])
                
                if features_to_remove:
                    st.warning(f"⚠️ Suppression automatique de {len(features_to_remove)} features suspectes...")
                    st.write("**Features supprimées:**", list(features_to_remove))
                    
                    # Supprimer des dataframes
                    features_to_remove_list = list(features_to_remove)
                    X_train = X_train.drop(columns=features_to_remove_list, errors='ignore')
                    X_test = X_test.drop(columns=features_to_remove_list, errors='ignore')
                    
                    # Re-standardiser après suppression
                    scaler_new = StandardScaler()
                    X_train_scaled = scaler_new.fit_transform(X_train)
                    X_test_scaled = scaler_new.transform(X_test)
                    
                    st.success(f"✅ Features supprimées. Nouveau shape: {X_train.shape}")
                else:
                    st.info("ℹ️ Aucune feature critique à supprimer automatiquement.")
            
            st.markdown("---")
            
            # Section 5.5: Entraînement MLP
            st.subheader("5.5 Entraînement Neural Network (MLP)")
            
            progress = st.progress(0)
            status = st.empty()
            
            status.text("⏳ Entraînement en cours...")
            
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=early_stopping,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            )
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            progress.progress(100)
            status.text("✅ Entraînement terminé!")
            
            st.success(f"✅ Modèle entraîné en {train_time:.2f} secondes ({model.n_iter_} itérations)")
            if hasattr(model, 'loss_'):
                st.write(f"Loss final: {model.loss_:.4f}")
            
            # Section 5.6: Prédictions
            st.subheader("5.6 Prédictions & Métriques")
            
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🎯 Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
            with col2:
                st.metric("🔍 Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
            with col3:
                st.metric("📈 Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
            with col4:
                st.metric("⚡ F1-Score", f"{f1:.4f}", f"{f1*100:.2f}%")
            with col5:
                st.metric("📊 ROC-AUC", f"{roc_auc:.4f}", f"{roc_auc*100:.2f}%")
            
            # Classification Report
            st.write("**Classification Report**")
            report = classification_report(y_test, y_pred, 
                                          target_names=['Normal (0)', 'DDoS (1)'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df, use_container_width=True)
            
            # Matrice de confusion détaillée
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            col1, col2 = st. columns([1, 1])
            
            with col1:
                st. write("**Matrice de Confusion**")
                st.write(f"- TN (Vrais Négatifs): {tn}")
                st.write(f"- FP (Faux Positifs): {fp}")
                st.write(f"- FN (Faux Négatifs): {fn}")
                st. write(f"- TP (Vrais Positifs): {tp}")
                
                st.write("**Taux**")
                st.write(f"- TPR (Sensibilité): {tp/(tp+fn):.4f}")
                st.write(f"- TNR (Spécificité): {tn/(tn+fp):.4f}")
                st.write(f"- FPR:  {fp/(fp+tn):.4f}")
                st.write(f"- FNR: {fn/(fn+tp):.4f}")
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                           xticklabels=['Normal', 'DDoS'],
                           yticklabels=['Normal', 'DDoS'])
                ax.set_title('Matrice de Confusion')
                ax.set_xlabel('Prédiction')
                ax.set_ylabel('Réalité')
                st.pyplot(fig)
            
            # Section 5.7: 6 Visualisations
            st.subheader("5.7 Visualisations Complètes")
            
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            
            # 1. Matrice confusion
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0, 0],
                       xticklabels=['Normal', 'DDoS'],
                       yticklabels=['Normal', 'DDoS'])
            axes[0, 0].set_title('Matrice de Confusion')
            axes[0, 0].set_xlabel('Prédiction')
            axes[0, 0].set_ylabel('Réalité')
            
            # 2. Matrice confusion %
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdPu', ax=axes[0, 1],
                       xticklabels=['Normal', 'DDoS'],
                       yticklabels=['Normal', 'DDoS'])
            axes[0, 1].set_title('Matrice de Confusion (%)')
            
            # 3. Métriques
            metrics = {'Acc': accuracy, 'Prec': precision, 'Rec':  recall, 'F1': f1, 'AUC': roc_auc}
            colors = ['steelblue', 'green', 'orange', 'purple', 'red']
            bars = axes[0, 2]. bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black')
            for bar in bars:
                h = bar.get_height()
                axes[0, 2]. text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}',
                               ha='center', va='bottom', fontweight='bold')
            axes[0, 2].set_ylim(0, 1.1)
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].set_title('Métriques')
            axes[0, 2].grid(axis='y', alpha=0.3)
            
            # 4. Courbe ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[1, 0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc:.4f}')
            axes[1, 0].plot([0, 1], [0, 1], 'r--', lw=2, label='Chance')
            axes[1, 0].set_xlabel('FPR')
            axes[1, 0].set_ylabel('TPR')
            axes[1, 0].set_title('Courbe ROC')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
            
            # 5. Distribution probabilités
            proba_normal = y_proba[y_test == 0]
            proba_ddos = y_proba[y_test == 1]
            axes[1, 1].hist(proba_normal, bins=30, alpha=0.6, label='Normal', color='green', edgecolor='black')
            axes[1, 1].hist(proba_ddos, bins=30, alpha=0.6, label='DDoS', color='red', edgecolor='black')
            axes[1, 1].axvline(0.5, color='blue', linestyle='--', lw=2, label='Seuil')
            axes[1, 1].set_xlabel('Probabilité')
            axes[1, 1].set_ylabel('Fréquence')
            axes[1, 1].set_title('Distribution des Probabilités')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            
            # 6. Loss curve
            if hasattr(model, 'loss_curve_'):
                axes[1, 2].plot(model.loss_curve_, 'b-', lw=2)
                axes[1, 2].set_xlabel('Itérations')
                axes[1, 2].set_ylabel('Loss')
                axes[1, 2].set_title('Courbe de Perte')
                axes[1, 2].grid(alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Loss curve\nnon disponible',
                               ha='center', va='center', transform=axes[1, 2]. transAxes)
            
            plt.suptitle('🧠 Neural Network - Analyse Complète', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Section 5.8: Diagnostic final
            st.subheader("5.8 Diagnostic Final")
            
            if accuracy == 1.0:
                st.error("🚨 ATTENTION:  100% accuracy → Vérifier data leakage!")
            elif accuracy > 0.95:
                st.warning("⚠️ Très haute accuracy → Possible data leakage")
            elif f1 > 0.8:
                st.success("✅ Excellent modèle!")
            elif f1 > 0.6:
                st.success("✅ Bon modèle")
            else:
                st.warning("⚠️ Modèle à améliorer")
            
            # Résumé
            st.info(f"""
            **Configuration**:
            - Architecture: {hidden_layers}
            - Train: {len(y_train)} échantillons
            - Test: {len(y_test)} échantillons
            - Features: {X_train_scaled.shape[1]}
            - Temps:  {train_time:.2f}s
            
            **Performances**:
            - Accuracy:  {accuracy*100:.2f}%
            - F1-Score: {f1*100:.2f}%
            - ROC-AUC:  {roc_auc*100:.2f}%
            """)
            
            # Sauvegarder
            st.session_state['mlp_model'] = model
            st.session_state['mlp_results'] = {
                'accuracy':  accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'train_time': train_time,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'confusion_matrix': cm
            }

# ============================================================
# PARTIE 6: COMPARAISON MODÈLES
# ============================================================

with tabs[5]:
    st.header("🏆 Partie 6: Comparaison Complète de 11 Modèles ML")
    
    st.info("""
    Cette section compare **11 modèles** de Machine Learning: 
    - 2 Decision Trees (simple/profond)
    - 3 KNN (k=3/5/10)
    - 3 Random Forest (50/100/200 arbres)
    - 1 Neural Network (MLP)
    - 1 Naive Bayes
    - 1 SVM
    """)
    
    # Configuration
    st.subheader("⚙️ Configuration Comparaison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.slider("Nombre de folds (CV)", 2, 10, 3)
    
    with col2:
        models_to_compare = st.multiselect(
            "Sélectionnez les modèles",
            [
                "Decision Tree (Simple)",
                "Decision Tree (Profond)",
                "KNN (k=3)",
                "KNN (k=5)",
                "KNN (k=10)",
                "Random Forest (50 arbres)",
                "Random Forest (100 arbres)",
                "Random Forest (200 arbres)",
                "Neural Network (MLP)",
                "Naive Bayes",
                "SVM"
            ],
            default=[
                "Decision Tree (Simple)",
                "KNN (k=5)",
                "Random Forest (100 arbres)",
                "Neural Network (MLP)"
            ]
        )
    
    if st.button("🚀 Lancer Comparaison Complète", type="primary"):
        
        if len(models_to_compare) == 0:
            st.error("⚠️ Veuillez sélectionner au moins un modèle!")
            st.stop()
        
        with st.spinner("⏳ Entraînement de tous les modèles..."):
            
            # Préparer données (même processus que Partie 5)
            data_complete = pd.concat([datatrainset, datatestset], ignore_index=True)
            
            features_numeriques = [
                'mqtt_qos', 'mqtt_duplicate', 'mqtt_retained',
                'requests_per_ip', 'duplicate_rate_per_ip', 'unique_topics_per_ip',
                'hour', 'day_of_week', 'day', 'month'
            ]
            
            features_disponibles = [f for f in features_numeriques if f in data_complete.columns]
            
            X_complete = data_complete[features_disponibles]. copy()
            y_complete = data_complete['is_ddos']. astype(int)
            
            X_complete = X_complete.fillna(X_complete.median())
            num_cols_clean = X_complete.select_dtypes(include=[np.number]).columns
            X_complete[num_cols_clean] = X_complete[num_cols_clean].replace([np.inf, -np.inf], np.nan)
            X_complete[num_cols_clean] = X_complete[num_cols_clean].fillna(X_complete[num_cols_clean].median())
            
            # Split stratifié
            X_train, X_test, y_train, y_test = train_test_split(
                X_complete, y_complete,
                test_size=test_size,
                random_state=random_state,
                stratify=y_complete
            )
            
            # Standardisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            st.success(f"✅ Données préparées:  {len(y_train)} train, {len(y_test)} test")
            
            # Définir tous les modèles
            all_models_config = {
                "Decision Tree (Simple)": {
                    'model': DecisionTreeClassifier(max_depth=10, min_samples_split=20, 
                                                   min_samples_leaf=10, random_state=random_state),
                    'use_scaled': False
                },
                "Decision Tree (Profond)": {
                    'model': DecisionTreeClassifier(max_depth=20, min_samples_split=10,
                                                   min_samples_leaf=5, random_state=random_state),
                    'use_scaled': False
                },
                "KNN (k=3)": {
                    'model': KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1),
                    'use_scaled': True
                },
                "KNN (k=5)": {
                    'model': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
                    'use_scaled': True
                },
                "KNN (k=10)": {
                    'model': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
                    'use_scaled': True
                },
                "Random Forest (50 arbres)": {
                    'model': RandomForestClassifier(n_estimators=50, max_depth=15,
                                                   min_samples_split=10, random_state=random_state, n_jobs=-1),
                    'use_scaled': False
                },
                "Random Forest (100 arbres)": {
                    'model': RandomForestClassifier(n_estimators=100, max_depth=20,
                                                   min_samples_split=5, random_state=random_state, n_jobs=-1),
                    'use_scaled': False
                },
                "Random Forest (200 arbres)": {
                    'model': RandomForestClassifier(n_estimators=200, max_depth=25,
                                                   min_samples_split=5, random_state=random_state, n_jobs=-1),
                    'use_scaled': False
                },
                "Neural Network (MLP)": {
                    'model': MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                                          solver='adam', max_iter=300, random_state=random_state,
                                          early_stopping=True, validation_fraction=0.1),
                    'use_scaled': True
                },
                "Naive Bayes": {
                    'model': GaussianNB(),
                    'use_scaled': True
                },
                "SVM": {
                    'model': SVC(kernel='rbf', probability=True, random_state=random_state),
                    'use_scaled': True
                }
            }
            
            # Filtrer les modèles sélectionnés
            models_config = {k: v for k, v in all_models_config. items() if k in models_to_compare}
            
            # Entraîner tous les modèles
            all_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (name, config) in enumerate(models_config.items()):
                status_text.text(f"⏳ Entraînement:  {name}...  ({idx+1}/{len(models_config)})")
                progress_bar.progress((idx + 1) / len(models_config))
                
                try:
                    model = config['model']
                    use_scaled = config['use_scaled']
                    
                    X_train_use = X_train_scaled if use_scaled else X_train
                    X_test_use = X_test_scaled if use_scaled else X_test
                    
                    # Entraînement
                    start = time.time()
                    model.fit(X_train_use, y_train)
                    train_time = time.time() - start
                    
                    # Prédictions
                    y_pred = model.predict(X_test_use)
                    
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test_use)[:, 1]
                    else:
                        y_proba = y_pred
                    
                    # Métriques
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    try:
                        roc = roc_auc_score(y_test, y_proba)
                    except:
                        roc = np.nan
                    
                    # Cross-validation
                    cv_results = cross_validate(
                        model, X_train_use, y_train,
                        cv=cv_folds,
                        scoring=['accuracy', 'f1'],
                        return_train_score=True,
                        n_jobs=-1
                    )
                    
                    train_acc_cv = cv_results['train_accuracy'].mean()
                    test_acc_cv = cv_results['test_accuracy'].mean()
                    gap = train_acc_cv - test_acc_cv
                    
                    # Sauvegarder
                    all_results[name] = {
                        'model':  model,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'roc_auc': roc,
                        'train_time':  train_time,
                        'cv_train_acc': train_acc_cv,
                        'cv_test_acc': test_acc_cv,
                        'gap': gap,
                        'y_pred': y_pred,
                        'y_proba':  y_proba,
                        'confusion_matrix':  confusion_matrix(y_test, y_pred)
                    }
                    
                except Exception as e:
                    st.warning(f"⚠️ Erreur pour {name}: {str(e)}")
            
            status_text.text("✅ Tous les modèles entraînés!")
            
            # Section 6.1: Tableau comparatif
            st.subheader("6.1 Tableau Comparatif")
            
            comparison_df = pd.DataFrame({
                name: {
                    'Accuracy': res['accuracy'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'F1-Score': res['f1'],
                    'ROC-AUC': res['roc_auc'],
                    'Temps (s)': res['train_time'],
                    'Gap (CV)': res['gap']
                }
                for name, res in all_results.items()
            }).T
            
            st.dataframe(comparison_df. style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='lightgreen')
                                           .highlight_min(axis=0, subset=['Gap (CV)', 'Temps (s)'], color='lightgreen')
                                           .format("{:.4f}"), 
                        use_container_width=True)
            
            # Meilleur modèle
            best_name = comparison_df['F1-Score'].idxmax()
            best_results = all_results[best_name]
            
            st.success(f"🏆 **Meilleur modèle**:  {best_name} (F1-Score: {best_results['f1']:.4f})")
            
            # Section 6.2: Classement
            st.subheader("6.2 Classement par F1-Score")
            
            ranking = comparison_df.sort_values('F1-Score', ascending=False)
            
            for i, (name, row) in enumerate(ranking.iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.write(f"{medal} **{name}**: F1={row['F1-Score']:.4f}, Acc={row['Accuracy']:.4f}, Gap={row['Gap (CV)']:.4f}")
            
            # Section 6.3: Visualisations comparatives
            st.subheader("6.3 Visualisations Comparatives")
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                # F1-Score
                fig, ax = plt.subplots(figsize=(10, 6))
                comparison_df['F1-Score']. sort_values().plot(kind='barh', ax=ax, color='purple', edgecolor='black')
                ax.set_xlabel('F1-Score')
                ax. set_title('Comparaison F1-Score')
                ax. grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Temps
                fig, ax = plt. subplots(figsize=(10, 6))
                comparison_df['Temps (s)'].sort_values().plot(kind='barh', ax=ax, color='orange', edgecolor='black')
                ax.set_xlabel('Temps (secondes)')
                ax.set_title("Temps d'Entraînement")
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC-AUC
                fig, ax = plt.subplots(figsize=(10, 6))
                comparison_df['ROC-AUC'].sort_values().plot(kind='barh', ax=ax, color='red', edgecolor='black')
                ax.set_xlabel('ROC-AUC')
                ax.set_title('Comparaison ROC-AUC')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Gap (Overfitting)
                fig, ax = plt.subplots(figsize=(10, 6))
                colors_gap = ['red' if g > 0.1 else 'orange' if g > 0.05 else 'green'
                             for g in comparison_df['Gap (CV)']]
                comparison_df['Gap (CV)'].sort_values().plot(kind='barh', ax=ax, color=colors_gap, edgecolor='black')
                ax.axvline(0.05, color='orange', linestyle='--', alpha=0.5, label='Seuil 5%')
                ax.axvline(0.10, color='red', linestyle='--', alpha=0.5, label='Seuil 10%')
                ax.set_xlabel('Gap Train-Test')
                ax.set_title('Overfitting (Gap CV)')
                ax.legend()
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            # Top 3 modèles
            st.subheader("6.4 Top 3 Modèles - Toutes Métriques")
            
            top3 = comparison_df.nlargest(3, 'F1-Score')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            top3.plot(kind='bar', ax=ax, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Score')
            ax.set_title('Top 3 Modèles - Comparaison des Métriques')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(loc='lower right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.1)
            st.pyplot(fig)
            
            # Section 6.5: Analyse détaillée meilleur modèle
            st.subheader("6.5 Analyse Détaillée du Meilleur Modèle")
            
            st.write(f"**Modèle**:  {best_name}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🎯 Accuracy", f"{best_results['accuracy']:.4f}")
            with col2:
                st.metric("🔍 Precision", f"{best_results['precision']:.4f}")
            with col3:
                st.metric("📈 Recall", f"{best_results['recall']:.4f}")
            with col4:
                st.metric("⚡ F1-Score", f"{best_results['f1']:.4f}")
            with col5:
                st.metric("📊 ROC-AUC", f"{best_results['roc_auc']:.4f}")
            
            # Visualisations meilleur modèle
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            
            cm = best_results['confusion_matrix']
            y_pred_best = best_results['y_pred']
            y_proba_best = best_results['y_proba']
            
            # 1. Matrice confusion
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0],
                       xticklabels=['Normal', 'DDoS'],
                       yticklabels=['Normal', 'DDoS'])
            axes[0, 0].set_title(f'Matrice de Confusion\n{best_name}')
            axes[0, 0].set_xlabel('Prédiction')
            axes[0, 0].set_ylabel('Réalité')
            
            # 2. Matrice %
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='YlGn', ax=axes[0, 1],
                       xticklabels=['Normal', 'DDoS'],
                       yticklabels=['Normal', 'DDoS'])
            axes[0, 1].set_title('Matrice de Confusion (%)')
            
            # 3. Métriques
            metrics = {
                'Acc': best_results['accuracy'],
                'Prec': best_results['precision'],
                'Rec': best_results['recall'],
                'F1': best_results['f1'],
                'AUC': best_results['roc_auc']
            }
            colors = ['steelblue', 'green', 'orange', 'purple', 'red']
            bars = axes[0, 2]. bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black')
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}',
                                   ha='center', va='bottom', fontweight='bold')
            axes[0, 2].set_ylim(0, 1.1)
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].set_title('Métriques')
            axes[0, 2].grid(axis='y', alpha=0.3)
            
            # 4. Courbe ROC
            if not np.isnan(best_results['roc_auc']):
                fpr, tpr, _ = roc_curve(y_test, y_proba_best)
                axes[1, 0].plot(fpr, tpr, 'b-', lw=2, label=f"AUC = {best_results['roc_auc']:.4f}")
                axes[1, 0].plot([0, 1], [0, 1], 'r--', lw=2)
                axes[1, 0].set_xlabel('FPR')
                axes[1, 0].set_ylabel('TPR')
                axes[1, 0].set_title('Courbe ROC')
                axes[1, 0].legend()
                axes[1, 0]. grid(alpha=0.3)
            
            # 5. Distribution probabilités
            if not np.isnan(best_results['roc_auc']):
                proba_normal = y_proba_best[y_test == 0]
                proba_ddos = y_proba_best[y_test == 1]
                axes[1, 1].hist(proba_normal, bins=30, alpha=0.6, label='Normal', color='green', edgecolor='black')
                axes[1, 1].hist(proba_ddos, bins=30, alpha=0.6, label='DDoS', color='red', edgecolor='black')
                axes[1, 1]. axvline(0.5, color='blue', linestyle='--', lw=2)
                axes[1, 1].set_xlabel('Probabilité')
                axes[1, 1].set_ylabel('Fréquence')
                axes[1, 1].set_title('Distribution des Probabilités')
                axes[1, 1]. legend()
                axes[1, 1].grid(alpha=0.3)
            
            # 6. Feature importance
            if hasattr(best_results['model'], 'feature_importances_'):
                importances = best_results['model']. feature_importances_
                indices = np.argsort(importances)[::-1][:15]
                
                axes[1, 2].barh(range(len(indices)), importances[indices], color='coral', edgecolor='black')
                axes[1, 2].set_yticks(range(len(indices)))
                axes[1, 2].set_yticklabels([X_train.columns[i] for i in indices], fontsize=9)
                axes[1, 2].set_xlabel('Importance')
                axes[1, 2].set_title('Top 15 Features')
                axes[1, 2].invert_yaxis()
                axes[1, 2].grid(axis='x', alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Feature importance\nnon disponible',
                               ha='center', va='center', transform=axes[1, 2].transAxes)
            
            plt.suptitle(f'Analyse Détaillée - {best_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Section 6.6: Analyse Random Forest (si applicable)
            if 'Random Forest' in best_name:
                st.subheader("6.6 Analyse Spéciale Random Forest")
                
                rf_model = best_results['model']
                
                col1, col2 = st. columns([2, 1])
                
                with col1:
                    # Feature importance avec écart-type
                    importances = rf_model.feature_importances_
                    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
                    indices = np.argsort(importances)[::-1][:20]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.barh(range(len(indices)), importances[indices],
                           xerr=std[indices], color='forestgreen', edgecolor='black', alpha=0.7)
                    ax.set_yticks(range(len(indices)))
                    ax.set_yticklabels([X_train.columns[i] for i in indices], fontsize=9)
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importances (avec écart-type)')
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    st.write("**Détails Random Forest**")
                    st.write(f"- Nombre d'arbres: {rf_model.n_estimators}")
                    st.write(f"- Profondeur max: {rf_model.max_depth}")
                    st.write(f"- Min samples split: {rf_model. min_samples_split}")
                    st.write(f"- Features utilisées: {rf_model.n_features_in_}")
                    st.write(f"- Classes: {rf_model.n_classes_}")
            
            # Section 6.7: Recommandations
            st.subheader("6.7 Recommandations")
            
            best_f1 = best_results['f1']
            best_gap = best_results['gap']
            
            if best_f1 > 0.9 and best_gap < 0.05:
                st.success(f"""
                ✅ **Excellent résultat! **
                
                Le modèle **{best_name}** obtient un F1-Score de {best_f1:.4f} avec 
                un gap minimal ({best_gap:.4f}), indiquant une excellente généralisation.
                
                **Recommandations**:
                - ✅ Modèle prêt pour la production
                - ✅ Performances stables
                - ✅ Bonne généralisation
                """)
            elif best_gap > 0.1:
                st.warning(f"""
                ⚠️ **Attention à l'overfitting! **
                
                Le modèle **{best_name}** montre des signes d'overfitting (gap:  {best_gap:.4f}).
                
                **Recommandations**: 
                - Augmenter la régularisation
                - Réduire la complexité du modèle
                - Collecter plus de données d'entraînement
                - Utiliser des techniques comme dropout ou early stopping
                """)
            else:
                st.info(f"""
                ℹ️ **Bon modèle**
                
                Le modèle **{best_name}** offre un bon compromis entre performance 
                et généralisation. F1-Score: {best_f1:.4f}
                
                **Points d'attention**:
                - Tester sur de nouvelles données
                - Surveiller les performances en production
                - Considérer l'ajustement des hyperparamètres
                """)

# ============================================================
# PARTIE 7: CONCEPTS MQTT (Guide Éducatif)
# ============================================================

with tabs[6]:
    st.header("📖 Concepts MQTT & Détection DDoS")
    
    st.markdown("""
    Cette section explique tous les concepts MQTT utilisés dans l'analyse et leur importance 
    pour la détection d'attaques DDoS (Distributed Denial of Service).
    """)
    
    # Introduction MQTT
    st.subheader("🔌 Qu'est-ce que MQTT? ")
    
    st.info("""
    **MQTT** (Message Queuing Telemetry Transport) est un protocole de messagerie léger 
    conçu pour l'IoT (Internet of Things).
    
    **Architecture**:
    - **Broker**: Serveur central qui gère les messages
    - **Publisher**: Client qui publie des messages
    - **Subscriber**: Client qui s'abonne à des topics
    - **Topic**:  Canal de communication hiérarchique
    """)
    
    # Concepts par catégorie
    tab_concepts = st.tabs([
        "⏰ Temporels",
        "🌐 Réseau",
        "📡 MQTT Protocol",
        "🎯 QoS & Flags",
        "🚨 Détection DDoS"
    ])
    
    # TAB:  Temporels
    with tab_concepts[0]:
        st. subheader("⏰ Concepts Temporels")
        
        st.markdown("### @timestamp")
        st.success("""
        **Description**: Horodatage du message au format ISO 8601
        
        **Format**: `2026-01-10T14:30:00Z`
        
        **Utilité pour DDoS**:
        - 📊 Analyse de patterns temporels (pics d'activité)
        - 🔍 Détection d'attaques coordonnées
        - ⏱️ Calcul de la durée des attaques
        - 📈 Timeline des événements suspects
        
        **Exemple d'analyse**:
        ```python
        # Détecter pic d'activité anormal
        messages_per_hour = data.groupby('hour').size()
        threshold = messages_per_hour.mean() + 3 * messages_per_hour. std()
        anomalies = messages_per_hour[messages_per_hour > threshold]
        ```
        """)
        
        st.markdown("### Features Dérivées")
        
        col1, col2 = st. columns(2)
        
        with col1:
            st. info("""
            **hour** (0-23)
            - Heure du jour
            - Détecte patterns horaires
            - Exemple: attaques nocturnes
            """)
            
            st.info("""
            **day** (1-31)
            - Jour du mois
            - Patterns mensuels
            - Campagnes coordonnées
            """)
        
        with col2:
            st.info("""
            **day_of_week** (0-6)
            - Jour de la semaine
            - Patterns hebdomadaires
            - Exemple: attaques week-end
            """)
            
            st.info("""
            **month** (1-12)
            - Mois de l'année
            - Tendances saisonnières
            - Campagnes longue durée
            """)
    
    # TAB: Réseau
    with tab_concepts[1]:
        st.subheader("🌐 Concepts Réseau")
        
        st.markdown("### general_ip")
        st.error("""
        **Description**:  Adresse IP source du message
        
        **Format**: IPv4 (192.168.1.1) ou IPv6
        
        **🚨 Importance CRITIQUE pour DDoS**:
        
        **1. Identification des Sources Malveillantes**
        ```python
        # Détecter IPs avec volume anormal
        ip_counts = data['general_ip'].value_counts()
        suspicious_ips = ip_counts[ip_counts > 200]  # Seuil:  200 requêtes
        ```
        
        **2. Analyse de Distribution**
        - IP unique = attaquant unique
        - Nombreuses IPs = botnet distribué
        - Même subnet = attaque coordonnée
        
        **3. Géolocalisation**
        - Identifier pays d'origine
        - Détecter patterns géographiques
        - Bloquer régions entières si nécessaire
        
        **4. Blacklisting Automatique**
        - Bloquer IPs malveillantes en temps réel
        - Partager listes noires (threat intelligence)
        
        **5. Rate Limiting**
        - Limiter requêtes par IP
        - Exemple: max 100 msg/min par IP
        """)
        
        st.markdown("### general_mac")
        st.warning("""
        **Description**:  Adresse MAC (Media Access Control)
        
        **Format**: 00:1B:44:11:3A:B7 (6 octets)
        
        **Utilité pour DDoS**: 
        
        **1. Identification Matérielle**
        - Unique par carte réseau
        - Impossible à changer facilement (spoofing difficile)
        - Traçabilité niveau hardware
        
        **2. Détection d'Appareils Compromis**
        ```python
        # Identifier devices compromis
        ddos_macs = data[data['is_ddos']==1]['general_mac'].unique()
        compromised_devices = data[data['general_mac'].isin(ddos_macs)]
        ```
        
        **3. Corrélation IP-MAC**
        - Détecter IP spoofing (IP change, MAC identique)
        - Identifier devices légitimes
        
        **Limitation**:  Visible seulement sur réseau local (LAN)
        """)
        
        st.markdown("### general_device_name")
        st.info("""
        **Description**: Nom/identifiant du périphérique
        
        **Exemples**:  `sensor_01`, `camera_lobby`, `thermostat_main`
        
        **Utilité pour DDoS**: 
        
        **1. Traçage Rapide**
        - Identifier rapidement l'appareil problématique
        - Localisation physique
        - Maintenance ciblée
        
        **2. Patterns par Type**
        ```python
        # Analyser par type de device
        device_analysis = data.groupby('general_device_name')['is_ddos'].mean()
        vulnerable_devices = device_analysis[device_analysis > 0.5]
        ```
        
        **3. Isolation**
        - Déconnecter device compromis
        - Quarantine automatique
        - Protection du réseau
        """)
        
        st.markdown("### general_full_id")
        st.info("""
        **Description**: Identifiant complet et unique
        
        **Format**: Combinaison MAC + IP + Topic
        
        **Exemple**:  `08:B6:1F:82:12:30_192.168.1.10_iiot/weather/temp`
        
        **Utilité**:
        - Unicité garantie
        - Déduplication des messages
        - Détection de replay attacks
        """)
    
    # TAB: MQTT Protocol
    with tab_concepts[2]: 
        st.subheader("📡 Protocole MQTT")
        
        st.markdown("### mqtt_topic")
        st.success("""
        **Description**:  Chemin hiérarchique de publication/abonnement
        
        **Format Hiérarchique**:  `niveau1/niveau2/niveau3`
        
        **Exemples**:
        - `home/livingroom/temperature`
        - `factory/machine1/status`
        - `iiot/weather/bmp180/pressure`
        
        **Wildcards**:
        - `+` : Un seul niveau (`home/+/temperature` → `home/bedroom/temperature`)
        - `#` : Multi-niveaux (`home/#` → tous sous home)
        
        **🚨 Détection DDoS via Topics**:
        
        **1. Topic Flooding**
        ```python
        # Détecter flood sur un topic
        topic_counts = ddos_data['mqtt_topic'].value_counts()
        flooded_topics = topic_counts[topic_counts > 1000]
        ```
        
        **2. Topics Inhabituels**
        - Nouveaux topics non documentés
        - Noms aléatoires/malformés
        - Patterns suspects
        
        **3. Diversité des Topics**
        ```python
        # Calculer diversité par IP
        unique_topics_per_ip = data.groupby('general_ip')['mqtt_topic'].nunique()
        # Faible diversité (< 5) = suspect
        ```
        
        **4. Ciblage de Topics Critiques**
        - Attaques sur topics sensibles
        - Saturation de topics système
        """)
        
        st.markdown("### mqtt_message_type")
        st.warning("""
        **Description**: Type de paquet MQTT
        
        **Types de Paquets**:
        
        | Type | Description | Utilisation |
        |------|-------------|-------------|
        | **CONNECT** | Connexion client | Établir session |
        | **CONNACK** | Acknowledgement connexion | Confirmation broker |
        | **PUBLISH** | Publication message | Envoyer données |
        | **PUBACK** | Ack publish (QoS 1) | Confirmer réception |
        | **SUBSCRIBE** | S'abonner à topic | Recevoir messages |
        | **SUBACK** | Ack subscription | Confirmation |
        | **PINGREQ** | Keep-alive ping | Maintenir connexion |
        | **PINGRESP** | Pong | Réponse keep-alive |
        | **DISCONNECT** | Déconnexion propre | Fermer session |
        
        **🚨 Alertes DDoS**:
        
        **1.  CONNECT Flood**
        ```python
        # Taux anormal de CONNECT
        connect_rate = (data['mqtt_message_type']=='CONNECT').sum() / len(data)
        if connect_rate > 0.3:  # > 30%
            print("⚠️ CONNECT flood détecté!")
        ```
        
        **2. SUBSCRIBE Abuse**
        - Trop de subscriptions simultanées
        - Wildcard abuse (`#` sur tout)
        
        **3. PUBLISH Spam**
        - Volume massif de PUBLISH
        - Messages vides/inutiles
        """)
        
        st.markdown("### mqtt_message_value")
        st.info("""
        **Description**: Payload/contenu du message
        
        **Formats Possibles**:
        - String: `"temperature: 25.5"`
        - JSON: `{"temp": 25.5, "unit": "C"}`
        - Binary:  Données brutes
        - Array: `[249. 0, 336.0, 336.0]`
        
        **Analyse pour DDoS**:
        
        **1. Taille des Messages**
        ```python
        # Détecter messages anormalement longs
        msg_lengths = data['mqtt_message_value'].str.len()
        large_msgs = msg_lengths[msg_lengths > 10000]  # > 10KB
        ```
        
        **2. Patterns Malveillants**
        - Messages répétitifs identiques
        - Contenu aléatoire (garbage)
        - Payload bombing (messages énormes)
        
        **3. Messages Vides**
        - Spam de messages sans contenu
        - Congestion du réseau
        """)
    
    # TAB: QoS & Flags
    with tab_concepts[3]:
        st.subheader("🎯 Quality of Service & Flags")
        
        st.markdown("### mqtt_qos")
        st.error("""
        **Description**:  Niveau de garantie de livraison
        
        **🎯 Trois Niveaux de QoS**:
        
        **QoS 0 - At Most Once (Au plus une fois)**
        - 🔥 Fire and Forget
        - Aucune garantie
        - Pas d'acknowledgement
        - **Avantage**: Très rapide, léger
        - **Inconvénient**:  Perte possible
        - **Usage**: Données non critiques
        
        **QoS 1 - At Least Once (Au moins une fois)**
        - ✅ Avec acknowledgement (PUBACK)
        - Garantie de livraison
        - Possibilité de duplicatas
        - **Avantage**: Fiable
        - **Inconvénient**: Plus de bande passante
        - **Usage**: Données importantes
        
        **QoS 2 - Exactly Once (Exactement une fois)**
        - 🎯 Handshake 4-way complet
        - Garantie absolue (pas de duplicate)
        - **Processus**:  PUBLISH → PUBREC → PUBREL → PUBCOMP
        - **Avantage**: Précision maximale
        - **Inconvénient**: Très coûteux en ressources
        - **Usage**:  Données critiques (transactions)
        
        **🚨 Exploitation DDoS**:
        
        **1. QoS 2 Attack**
        ```python
        # Détecter abus de QoS 2
        qos2_rate = (data['mqtt_qos']==2).sum() / len(data)
        if qos2_rate > 0.5:  # > 50%
            print("⚠️ Attaque QoS 2 détectée!")
        ```
        
        **Pourquoi QoS 2 est dangereux? **
        - 4x plus de messages que QoS 0
        - Consomme énormément de mémoire broker
        - Ralentit tout le système
        - **DDoS amplifié**:  1 message → 4 paquets
        
        **2. QoS Mixing Attack**
        - Combiner QoS élevé + volume massif
        - Saturer la file d'attente du broker
        
        **3. Détection**: 
        ```python
        # Analyser distribution QoS
        qos_by_ip = data.groupby('general_ip')['mqtt_qos'].mean()
        # IPs utilisant majoritairement QoS 2 = suspect
        suspicious = qos_by_ip[qos_by_ip > 1.5]
        ```
        """)
        
        st.markdown("### mqtt_duplicate")
        st.error("""**Description**: Flag indiquant un message dupliqué (True/False)
        """)