# Classification d'événements — Boson de Higgs

## Présentation

Ce projet porte sur l'analyse et la classification d'événements issus de données de physique des particules. L'objectif du travail est de distinguer deux classes d'événements :

- **signal (`s`)** : événements associés au signal recherché ;
- **bruit (`b`)** : événements correspondant au bruit de fond.

Le notebook suit une démarche progressive : on commence par comprendre les données et les variables, puis on étudie leur pouvoir discriminant et leurs redondances avant de construire des modèles de classification avec **XGBoost**.

Une particularité importante du jeu de données est la présence de **poids d'événements (`Weight`)**. Ces poids sont étudiés séparément puis intégrés dans l'entraînement et l'évaluation des modèles de classification.

---

## Données

Le fichier utilisé dans le notebook est :

```text
training.csv
```
**Le fichier est retrouvable sur le lien Kaggle:** https://www.kaggle.com/competitions/higgs-boson

Le jeu de données contient :

- **250 000 événements** ;
- **33 colonnes** au total ;
- **30 variables explicatives** ;
- une colonne `Label` contenant les classes `s` et `b` ;
- une colonne `Weight` contenant le poids associé à chaque événement ;
- une colonne `EventId` identifiant les événements ;
- une variable `PRI_jet_num` indiquant la catégorie selon le nombre de jets.

Les variables commencent notamment par `DER_` et `PRI_` et correspondent à différentes grandeurs utilisées pour décrire les événements.

Le notebook met également en évidence l'utilisation de la valeur **`-999`** pour représenter une mesure indisponible dans certaines configurations d'événements.

---

## Organisation du notebook

Le notebook `boson_higgs_complet_xgboost.ipynb` est organisé en deux grandes parties :

1. **analyse exploratoire et étude des variables** ;
2. **classification avec XGBoost**.

L'idée est de ne pas passer directement au modèle : les choix effectués dans la seconde partie sont issus des analyses réalisées dans la première partie.

---

# 1. Analyse exploratoire

## 1.1. Importation et lecture des données

Les premières cellules chargent les bibliothèques utilisées pour l'analyse :

- `pandas` pour la manipulation des données ;
- `numpy` pour les calculs numériques ;
- `matplotlib` et `seaborn` pour les visualisations ;
- `scikit-learn` pour les métriques ;
- `xgboost` pour la classification dans la seconde partie.

---

## 1.2. Résumé global du jeu de données

Le notebook commence par un résumé général :

- nombre total d'événements ;
- nombre de colonnes ;
- nombre de variables explicatives ;
- nombre d'événements signal et bruit ;
- proportion de chaque classe ;
- proportion globale de valeurs codées `-999`.

Cette étape permet d'avoir une première vision de la structure du jeu de données avant de commencer l'analyse détaillée.

---

## 1.3. Analyse des variables

Une synthèse est construite pour les variables explicatives afin d'observer notamment :

- le nombre de valeurs manquantes représentées par `-999` ;
- le nombre de valeurs distinctes ;
- le type de variable au sens de l'analyse réalisée dans le notebook ;
- les caractéristiques particulières de certaines variables, par exemple les variables angulaires.

Le but est de comprendre la nature des variables avant de sélectionner celles qui seront utilisées ensuite pour la classification.

---

## 1.4. Séparation selon le nombre de jets

Les événements sont séparés en quatre groupes :

- **0 jet** ;
- **1 jet** ;
- **2 jets** ;
- **3 jets ou plus**.

Cette séparation est conservée dans toute la partie de classification. L'idée est de construire un modèle adapté à chacune de ces catégories plutôt que d'utiliser exactement le même ensemble de variables pour tous les événements.

---

## 1.5. Gestion des variables très incomplètes

La fonction `nettoyer_colonnes` permet d'identifier les variables pour lesquelles la proportion de valeurs `-999` dépasse un seuil fixé à **80 %**.

Ces variables sont retirées du sous-dataset concerné, tandis que les colonnes `EventId`, `Weight`, `Label` et `PRI_jet_num` sont conservées car elles ont un rôle spécifique dans l'analyse.

Cette étape constitue un premier filtrage des variables trop peu renseignées.

---

## 1.6. Pouvoir discriminant des variables avec l'AUC

Pour chaque catégorie de jets, le notebook calcule ensuite une **AUC variable par variable**.

Pour chaque variable :

1. les valeurs `-999` sont exclues de ce calcul ;
2. le label est transformé en variable binaire (`s = 1`, `b = 0`) ;
3. l'AUC de la variable est calculée ;
4. une AUC corrigée est utilisée pour tenir compte du cas où la variable sépare le signal et le bruit dans le sens opposé.

Le tableau obtenu contient notamment :

- l'AUC brute ;
- l'AUC corrigée ;
- la distance à `0.5` ;
- le sens de séparation entre les deux classes.

Cette analyse permet d'identifier les variables qui présentent individuellement une capacité de séparation entre signal et bruit.

---

## 1.7. Étude des corrélations

Le notebook utilise ensuite des **corrélations de Spearman** entre variables.

Pour chaque catégorie de jets, une matrice de corrélation est affichée et les paires présentant une corrélation absolue supérieure ou égale à **0.85** sont repérées.

Cette analyse complète l'AUC : une variable peut être intéressante individuellement mais apporter peu d'information supplémentaire si elle est très redondante avec une autre variable déjà sélectionnée.

Les analyses de corrélation ont notamment conduit à retirer certaines variables jugées très redondantes dans la sélection finale, par exemple :

- `DER_prodeta_jet_jet`, avec une corrélation d'environ `0.89` avec une autre variable ;
- certaines variables liées aux jets lorsque leur information était déjà fortement représentée par une autre variable.

---

## 1.8. Étude de la variable `Weight`

La colonne `Weight` est étudiée séparément.

Le notebook compare notamment :

- les proportions brutes des classes dans le fichier ;
- les proportions obtenues après prise en compte des poids.

Cette comparaison permet de montrer que le nombre d'événements observés dans le fichier et leur contribution pondérée ne représentent pas nécessairement la même chose.

Cette observation est importante pour la suite, puisque les poids sont utilisés lors de l'entraînement et de l'évaluation du classifieur.

---

# 2. Classification avec XGBoost

Après l'analyse exploratoire, le notebook passe à une étape de classification supervisée.

Le modèle choisi est **XGBoost**, utilisé séparément pour chaque catégorie de jets.

L'objectif n'est pas seulement d'obtenir une prédiction, mais de conserver la logique construite pendant l'analyse exploratoire : sélection des variables à partir de leur pouvoir discriminant et prise en compte des redondances.

---

## 2.1. Sélection des variables

Les variables utilisées par XGBoost sont définies séparément selon la catégorie de jets.

### 0 jet

Le modèle utilise notamment :

```text
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
DER_pt_h
DER_deltar_tau_lep
DER_sum_pt
DER_pt_ratio_lep_tau
DER_met_phi_centrality
PRI_tau_pt
PRI_tau_eta
PRI_lep_pt
PRI_lep_eta
PRI_met
PRI_met_sumet
```

### 1 jet

On conserve les variables précédentes et on ajoute les caractéristiques du jet principal :

```text
PRI_jet_leading_pt
PRI_jet_leading_eta
```

### 2 jets

La sélection contient notamment les variables décrivant les deux jets :

```text
DER_deltaeta_jet_jet
DER_mass_jet_jet
PRI_jet_leading_pt
PRI_jet_subleading_pt
PRI_jet_all_pt
```

ainsi que les variables communes aux autres catégories.

### 3 jets ou plus

La même sélection que pour la catégorie 2 jets est utilisée dans le notebook.

Cette sélection n'est donc pas arbitraire : elle provient des analyses d'AUC et de corrélation réalisées auparavant.

---

## 2.2. Gestion des valeurs `-999`

Avant l'entraînement, les valeurs `-999` sont remplacées par `NaN` :

```python
X = donnees_jet[variables].replace(-999.0, np.nan)
```

Cette représentation permet à XGBoost de traiter ces valeurs comme des données manquantes.

---

## 2.3. Encodage de la cible

Le label est transformé en variable binaire :

```python
y = (donnees_jet["Label"] == "s").astype(int)
```

On obtient donc :

- `1` pour le signal ;
- `0` pour le bruit.

---

## 2.4. Prise en compte des poids

Les poids de chaque événement sont récupérés depuis la colonne `Weight`.

Dans le notebook, ils sont normalisés par leur moyenne avant d'être passés au modèle :

```python
poids_normalises = poids / poids.mean()
```

Ils sont ensuite utilisés :

- pendant l'entraînement avec `sample_weight` ;
- pendant l'évaluation de validation avec `sample_weight_eval_set` ;
- pour le calcul final de l'AUC avec `sample_weight`.

La normalisation conserve les rapports entre les poids tout en évitant de travailler directement avec leur échelle brute.

---

## 2.5. Séparation entraînement / validation

Pour chaque catégorie de jets, les données sont séparées en deux ensembles :

- **80 % pour l'entraînement** ;
- **20 % pour la validation**.

La séparation est stratifiée sur la variable cible afin de conserver la répartition des classes entre les deux ensembles.

La graine `random_state=42` est utilisée pour rendre la séparation reproductible.

---

## 2.6. Modèle XGBoost

Le classifieur est construit avec les paramètres suivants :

```text
n_estimators = 500
max_depth = 5
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 10
gamma = 1.0
reg_alpha = 0.1
reg_lambda = 1.0
eval_metric = auc
early_stopping_rounds = 30
```

Le modèle utilise également `random_state=42` et tous les cœurs disponibles avec `n_jobs=-1`.

### Early stopping

L'entraînement surveille l'AUC sur l'ensemble de validation. Si la performance ne s'améliore plus pendant 30 itérations consécutives, l'entraînement s'arrête avant d'atteindre nécessairement les 500 arbres prévus.

Le notebook conserve alors le nombre d'itérations correspondant au meilleur résultat observé.

---

# 3. Évaluation des modèles

## 3.1. AUC de validation

Pour chaque catégorie de jets, l'AUC est calculée sur l'ensemble de validation avec les poids des événements.

L'AUC est choisie comme métrique principale car le problème porte sur la séparation de deux classes et que les distributions sont étudiées avec des poids associés aux événements.

Les valeurs finales sont recalculées lorsque le notebook est exécuté avec `training.csv`.

---

## 3.2. Courbes ROC

Pour chaque catégorie de jets, une courbe ROC est construite à partir des probabilités prédites sur l'ensemble de validation.

La figure finale regroupe les quatre catégories :

- 0 jet ;
- 1 jet ;
- 2 jets ;
- 3 jets ou plus.

L'AUC correspondante est affichée dans la légende de chaque courbe.

---

## 3.3. Importance des variables

Le notebook affiche également les variables les plus importantes pour chaque modèle XGBoost.

Les dix variables les plus importantes sont visualisées afin de faciliter l'interprétation du modèle et de comparer les catégories de jets.

L'importance utilisée est explicitement celle basée sur le **gain**.

---

## 3.4. Tableau récapitulatif

La dernière cellule construit un tableau regroupant, pour chaque catégorie :

- le nombre de variables utilisées ;
- la taille de l'ensemble d'entraînement ;
- la taille de l'ensemble de validation ;
- le nombre d'itérations correspondant au meilleur résultat ;
- l'AUC de validation.

Ce tableau fournit une synthèse compacte de la phase de classification.

---

# 4. Résumé de la démarche

La démarche suivie dans le projet peut se résumer ainsi :

```text
Données brutes
      ↓
Résumé statistique
      ↓
Analyse des variables et des valeurs -999
      ↓
Séparation selon le nombre de jets
      ↓
Filtrage des variables trop incomplètes
      ↓
AUC variable par variable
      ↓
Analyse des corrélations
      ↓
Étude des poids d'événements
      ↓
Sélection des variables
      ↓
XGBoost par catégorie de jets
      ↓
Poids intégrés à l'entraînement
      ↓
Validation + early stopping
      ↓
AUC pondérée + ROC
      ↓
Importance des variables
      ↓
Tableau récapitulatif
```

Cette organisation permet de relier directement l'analyse exploratoire aux choix de modélisation.

---

# 5. Technologies utilisées

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Jupyter Notebook**

---

# 6. Fichiers du projet

```text
.
├── boson_higgs_complet_xgboost.ipynb
├── training.csv
└── README_boson_higgs_complet.md
```

Le fichier `training.csv` n'est pas fourni avec le notebook : il doit être placé dans le même répertoire pour exécuter les cellules de chargement et de classification.

---

# 7. Exécution du notebook

Installer les bibliothèques nécessaires :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Puis lancer Jupyter :

```bash
jupyter notebook
```

Ouvrir ensuite :

```text
boson_higgs_complet_xgboost.ipynb
```

et vérifier que `training.csv` se trouve dans le répertoire attendu.

---

# 8. Limites du projet

Le notebook constitue principalement une étude expérimentale et une première modélisation. Plusieurs éléments pourraient être approfondis dans une version ultérieure :

- comparaison avec d'autres modèles de classification ;
- recherche plus systématique des hyperparamètres ;
- validation croisée ;
- étude plus détaillée de la stabilité des résultats ;
- analyse plus poussée de l'effet des poids ;
- interprétation plus approfondie des variables importantes.


---

# 9. Résultats

Les résultats numériques définitifs (AUC, nombre d'itérations, tailles exactes des ensembles et importances des variables) sont produits directement par l'exécution du notebook.


---

## Conclusion

Ce projet présente une chaîne complète allant de l'exploration d'un jeu de données de physique des particules jusqu'à une première approche de classification supervisée.

L'intérêt principal du travail est la continuité entre les différentes étapes : les observations faites pendant l'analyse exploratoire servent à construire la sélection de variables, puis cette sélection est utilisée dans des modèles XGBoost distincts selon le nombre de jets, avec prise en compte des poids des événements et évaluation par AUC.
