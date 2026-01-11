# Classification par Réseau de Neurones - 20 Newsgroups

## Description

Ce projet implémente un modèle de classification par réseau de neurones (MLP - Multi-Layer Perceptron) pour le jeu de données **20 Newsgroups**. Le notebook contient une analyse complète incluant la préparation des données, l'entraînement du modèle, l'évaluation des performances et la comparaison avec une baseline.

## Structure du projet

```
ml-projet/
├── 20newsgroups_neural_network.ipynb  # Notebook principal
├── README.md                            # Ce fichier
└── figures/                             # Dossier pour les graphiques (créé automatiquement)
    ├── class_distribution.png
    ├── document_length_distribution.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── cross_validation_scores.png
    ├── top_errors.png
    └── learning_curve.png
```

## Prérequis

### Bibliothèques Python requises

```bash
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Installation

#### Option 1 : Installation manuelle

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

#### Option 2 : Installation via requirements.txt

Créez un fichier `requirements.txt` :

```txt
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
```

Puis installez :

```bash
pip install -r requirements.txt
```

## Instructions d'exécution

### 1. Lancer Jupyter Notebook

```bash
jupyter notebook
```

Ou avec JupyterLab :

```bash
jupyter lab
```

### 2. Ouvrir le notebook

- Ouvrez le fichier `20newsgroups_neural_network.ipynb`
- Assurez-vous que le kernel Python est sélectionné

### 3. Exécuter le notebook

#### Option A : Exécution complète

- Menu : `Cell` → `Run All`
- Ou raccourci clavier : `Shift + Enter` dans chaque cellule

#### Option B : Exécution séquentielle

Exécutez les cellules dans l'ordre en utilisant `Shift + Enter` pour chaque cellule.

### 4. Temps d'exécution estimé

- **Chargement des données** : ~10-30 secondes
- **Vectorisation TF-IDF** : ~1-2 minutes
- **Modèle baseline** : ~1 seconde
- **GridSearchCV** : ~10-30 minutes (selon la configuration)
- **Évaluation complète** : ~2-5 minutes

**Total estimé** : ~15-40 minutes

### 5. Résultats attendus

Le notebook génère :

- **Métriques de performance** : Accuracy, F1-score, matrice de confusion
- **Graphiques** : 
  - Distribution des classes
  - Longueur des documents
  - Matrice de confusion
  - Courbes ROC
  - Courbes Precision-Recall
  - Scores de validation croisée
  - Analyse des erreurs
  - Courbe d'apprentissage

Tous les graphiques sont sauvegardés automatiquement en haute résolution (300 dpi) dans le répertoire courant.

## Configuration et reproductibilité

### Seed fixe

Le notebook utilise un `random_state=42` pour garantir la reproductibilité des résultats.

### Versions des bibliothèques

Les versions des bibliothèques principales sont affichées au début du notebook. Pour vérifier vos versions :

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
```

## Jeu de données

### 20 Newsgroups

- **Source** : Scikit-learn (`sklearn.datasets.fetch_20newsgroups`)
- **Description** : Collection d'environ 20 000 documents de groupes de discussion Usenet, répartis en 20 catégories thématiques
- **Classes** : 20 catégories (comp.graphics, comp.os.ms-windows.misc, rec.sport.baseball, etc.)
- **Taille** : ~11 314 échantillons d'entraînement, ~7 532 échantillons de test
- **Format** : Texte brut (documents Usenet)
- **Licence** : Domaine public / Usage éducatif

### Téléchargement automatique

Le dataset est téléchargé automatiquement lors de la première exécution via `fetch_20newsgroups()`. Aucune action manuelle n'est requise.

## Méthodologie

### 1. Préprocessing

- **Vectorisation** : TF-IDF avec unigrammes et bigrammes
- **Réduction de dimensionnalité** : Limitation à 5000 features
- **Nettoyage** : Suppression des headers, footers, quotes et stop words
- **Normalisation** : Standardisation des features

### 2. Modèle

- **Architecture** : Multi-Layer Perceptron (MLP)
- **Optimisation** : GridSearchCV avec validation croisée
- **Régularisation** : L2 (alpha) + Early stopping
- **Hyperparamètres optimisés** :
  - Nombre et taille des couches cachées
  - Fonction d'activation (ReLU, tanh)
  - Taux d'apprentissage
  - Taille des batches
  - Coefficient de régularisation

### 3. Évaluation

- **Métriques** : Accuracy, F1-score (weighted et macro), matrice de confusion
- **Courbes** : ROC, Precision-Recall
- **Validation** : 5-fold cross-validation
- **Baseline** : Classifieur naïf (most frequent)

## Résultats typiques

Avec les paramètres par défaut, vous devriez obtenir :

- **Baseline** : Accuracy ~0.05-0.10 (selon la distribution)
- **MLP optimisé** : Accuracy ~0.75-0.85, F1-score ~0.75-0.85

*Note : Les résultats peuvent varier légèrement selon les versions des bibliothèques et l'environnement.*

## Dépannage

### Problème : Mémoire insuffisante

**Solution** : Réduire `max_features` dans le TfidfVectorizer (par exemple, 3000 au lieu de 5000)

```python
vectorizer = TfidfVectorizer(max_features=3000, ...)
```

### Problème : GridSearchCV trop long

**Solution 1** : Réduire la grille d'hyperparamètres

**Solution 2** : Utiliser RandomizedSearchCV au lieu de GridSearchCV

**Solution 3** : Réduire `cv` (par exemple, `cv=3` au lieu de `cv=5`)

### Problème : Erreur HTTP 403 lors du téléchargement du dataset

**Causes possibles** :
- Le serveur bloque les requêtes sans User-Agent approprié
- Problème de connexion Internet
- Serveur temporairement indisponible

**Solutions** :

1. **Solution automatique (déjà implémentée)** : Le notebook configure automatiquement un User-Agent. Si l'erreur persiste :

2. **Utiliser le script de téléchargement manuel** :
   ```bash
   python download_dataset.py
   ```

3. **Téléchargement manuel** :
   - Téléchargez depuis : http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
   - Trouvez le dossier de cache :
     ```python
     python -c "from sklearn.datasets import get_data_home; print(get_data_home())"
     ```
   - Extrayez le fichier `.tar.gz` dans ce dossier
   - Relancez la cellule du notebook

4. **Utiliser les données en cache** : Si vous avez déjà téléchargé les données précédemment, le notebook essaiera automatiquement de les charger depuis le cache.

### Problème : Graphiques non affichés

**Solution** : Vérifier que matplotlib est correctement installé et que le backend est configuré :

```python
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg' selon votre système
```

## Améliorations possibles

1. **Embeddings pré-entraînés** : Utiliser Word2Vec, GloVe ou BERT
2. **Architectures avancées** : LSTM, GRU, Transformers
3. **Optimisation** : RandomizedSearchCV, optimisation bayésienne
4. **Interprétabilité** : LIME, SHAP
5. **Gestion du déséquilibre** : class_weight='balanced', SMOTE

## Références

### Jeu de données

- Lang, K. (1995). Newsweeder: Learning to filter netnews. *Proceedings of the 12th International Conference on Machine Learning*.
- Documentation scikit-learn : https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset

### Bibliothèques

- Scikit-learn : https://scikit-learn.org/
- NumPy : https://numpy.org/
- Pandas : https://pandas.pydata.org/
- Matplotlib : https://matplotlib.org/
- Seaborn : https://seaborn.pydata.org/

## Licence

Ce projet est à des fins éducatives uniquement. Le jeu de données 20 Newsgroups est dans le domaine public.

## Auteur

Rim Lazrek
Youssef Berrada
Nabil Benfana

## Contact

Pour toute question ou problème, veuillez consulter la documentation ou créer une issue.

---

**Note** : Ce notebook a été conçu pour être exécutable de bout en bout. Assurez-vous d'avoir toutes les dépendances installées avant de commencer.

