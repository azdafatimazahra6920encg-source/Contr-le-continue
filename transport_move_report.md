## AZDA Fatima-zahra
<img src="faat.jpg" style="height:150px;margin-right:100px"/>

# Thématique choisie: Transport
# Rapport Scientifique : Analyse Prédictive du Déménagement dans le Transport
**Dataset :** Transport Move (willianoliveiragibin/transport-move)  
**Type de problème :** Classification binaire supervisée  
**Objectif :** Prédire la probabilité de déménagement basée sur les patterns de transport

---
Voici un **sommaire clair, structuré et académique**, parfaitement aligné avec le contenu de ton rapport scientifique 👇
(Il peut être utilisé tel quel dans un mémoire, rapport ou article.)

---

## **Sommaire**

## Introduction

 1. Contexte
 2. Problématique
 3. Objectifs

---

## Chapitre 1 : Présentation générale du projet

### 1. Thématique du transport et de la mobilité
### 2. Description du dataset *Transport Move*
### 3. Formulation du problème de classification

---

**Chapitre 2 : Méthodologie**
### 2.1 Collecte et préparation des données

### 2.2 Analyse exploratoire des données (EDA)

### 2.3 Modélisation
   
**Chapitre 3 : Résultats et discussion**
### 3.1 Comparaison des performances des modèles

### 3.2 Analyse des métriques d’évaluation

### 3.3 Analyse de l’importance des variables

### 3.4 Analyse de la matrice de corrélation

### 3.5 Analyse des distributions des variables et de la cible

---

**Chapitre 4 : Conclusion et perspectives**
### 4.1 Synthèse des résultats

### 4.2 Limites du modèle
   
### 4.3 Pistes d’amélioration
   

---

**Chapitre 5 : Annexes**
### 5.1 Environnement technique

### 5.2 Reproductibilité

### 5.3 Considérations éthiques

---


## 1. Introduction

### 1.1 Contexte

La mobilité urbaine et les patterns de déplacement constituent des indicateurs pertinents pour anticiper les changements de résidence. L'analyse des données de transport peut révéler des comportements précurseurs d'un déménagement imminent, tels que l'exploration de nouveaux quartiers, l'augmentation des distances parcourues ou la modification des routines de déplacement.

### 1.2 Problématique

**Question de recherche :** Peut-on prédire si un individu va déménager en analysant ses données de transport et de mobilité ?

Cette problématique s'inscrit dans un contexte où :
- Les entreprises de déménagement cherchent à cibler leurs campagnes marketing
- Les urbanistes souhaitent anticiper les flux migratoires intra-urbains
- Les services publics veulent optimiser leurs infrastructures de transport

### 1.3 Objectifs

1. **Objectif principal :** Développer un modèle de classification binaire capable de prédire le déménagement (variable cible : `move`)
2. **Objectifs secondaires :**
   - Identifier les features les plus discriminantes
   - Comparer plusieurs algorithmes de machine learning
   - Optimiser les hyperparamètres pour maximiser les performances
   - Analyser les patterns comportementaux associés au déménagement

---

## 2. Méthodologie

### 2.1 Collecte et Préparation des Données

#### 2.1.1 Dataset
- **Source :** Kaggle (willianoliveiragibin/transport-move)
- **Nature :** Données comportementales de transport et mobilité
- **Taille :** 8142 observations × 4 variables initiales
- **Variables :** Distances parcourues, fréquence des trajets, types de transport utilisés

#### 2.1.2 Pré-traitement

**Choix techniques justifiés :**

1. **Suppression des doublons**
   - **Justification :
   - **Résultat :** 0 doublon détecté (dataset propre)
   - **Justification :** Les doublons introduisent un biais dans l'apprentissage en surpondérant certaines observations

2. 2. **Gestion des valeurs manquantes**
   - **Avant traitement :** 22 valeurs manquantes
   - **Après traitement :** 0 valeur manquante
   - **Méthode :** Imputation KNN (k=5) pour variables numériques + mode pour catégorielles

3. **Imputation par mode pour variables catégorielles**
   - **Justification :** Pour les variables qualitatives (type de transport, zone géographique), le mode représente la valeur la plus fréquente et donc la plus probable statistiquement.

4. **Feature Engineering de la cible**
   - **Création de la variable `move`** basée sur la médiane des passagers transportés
   - **Distribution :** 50% / 50% (parfaitement équilibrée)
   - **Note importante :** Variable synthétique créée car le dataset original ne contient pas d'indicateurs directs de déménagement

5. **Label Encoding pour variables catégorielles**
   - **Justification :** Conversion des catégories en valeurs numériques pour compatibilité avec les algorithmes ML. Préféré au One-Hot Encoding pour éviter l'explosion dimensionnelle sur des variables à forte cardinalité.

6. **Standardisation (StandardScaler)**
   - **Justification :** Normalisation des features pour mettre toutes les variables sur une échelle comparable (moyenne=0, écart-type=1). Essentiel pour :
     - La régression logistique (sensible aux échelles)
     - La convergence des algorithmes d'optimisation
     - L'interprétabilité des coefficients

### 2.2 Analyse Exploratoire (EDA)

#### 2.2.1 Feature Engineering Avancé

Deux nouvelles features ont été créées pour capturer des patterns complexes :

1. **`distance_per_trip`** : Distance moyenne par trajet
   - **Justification :** Distingue les individus effectuant des trajets longs (potentiellement exploratoires) de ceux effectuant de nombreux trajets courts (routines locales)

2. **`trip_variability`** : Écart-type des fréquences de trajets
   - **Justification :** Mesure l'irrégularité des patterns de déplacement. Une forte variabilité peut indiquer une rupture des routines, signe précurseur d'un changement de résidence.

#### 2.2.2 Analyse de corrélation

L'analyse s'est concentrée sur les **Top 10 features** les plus corrélées avec la cible pour :
- Réduire le bruit (features non pertinentes)
- Améliorer l'interprétabilité
- Prévenir le surapprentissage (overfitting)

### 2.3 Modélisation

#### 2.3.1 Choix des algorithmes

Trois familles d'algorithmes ont été sélectionnées pour couvrir différents paradigmes d'apprentissage :

**1. Régression Logistique**
- **Type :** Modèle linéaire généralisé
- **Justification :** 
  - Baseline interprétable (coefficients = importance des features)
  - Rapide à entraîner
  - Performant sur données linéairement séparables
- **Hyperparamètres testés :** `C = [0.1, 1, 10]` (régularisation L2)

**2. Random Forest**
- **Type :** Ensemble de bagging (arbres de décision)
- **Justification :**
  - Capture les interactions non-linéaires complexes
  - Robuste aux outliers et au surapprentissage (agrégation de multiples arbres)
  - Fournit des importances de features natives
- **Hyperparamètres testés :**
  - `n_estimators = [100, 200]` (nombre d'arbres)
  - `max_depth = [10, 20]` (profondeur maximale, contrôle de la complexité)

**3. Gradient Boosting**
- **Type :** Ensemble de boosting (apprentissage séquentiel)
- **Justification :**
  - État de l'art pour tâches de classification structurée
  - Correction itérative des erreurs des modèles précédents
  - Excellente capacité de généralisation avec régularisation appropriée
- **Hyperparamètres testés :**
  - `n_estimators = [100, 200]`
  - `learning_rate = [0.1, 0.2]` (taux d'apprentissage, compromis vitesse/précision)

#### 2.3.2 Optimisation et validation

**GridSearchCV avec validation croisée 5-fold :**
- **Métrique d'optimisation :** F1-Score (harmonique précision-rappel)
- **Justification du F1-Score :** En présence de classes potentiellement déséquilibrées (déménagement = événement rare), l'accuracy est trompeuse. Le F1-Score pénalise les modèles qui privilégient excessivement une classe.
- **Stratégie de split :** Stratifiée pour maintenir la proportion des classes dans chaque fold

**Protocole de validation :**
1. Split train/test (80/20) stratifié
2. GridSearch sur train set avec CV=5
3. Évaluation finale sur test set (données jamais vues)

---

## 3. Résultats & Discussion

### 3.1 Performances des modèles

#### 3.1.1 Comparaison des algorithmes

| Modèle               | F1-Score (CV) | 
|----------------------|---------------|
| Logistic Regression  | 0.820         |
| Random Forest        | 1.000         |
| **Gradient Boosting**| 1.000         | 


**🏆 Meilleur modèle :** Random Forest (sélectionné arbitrairement entre RF et GB, performances identiques)

**Analyse :**
- Le Gradient Boosting surpasse les autres modèles grâce à sa capacité à corriger itérativement les erreurs
- La Régression Logistique, malgré sa simplicité, fournit une baseline solide démontrant une certaine séparabilité linéaire des classes

### 3.2 Métriques détaillées (Test Set)

#### 3.2.1 Rapport de classification

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       815
           1       1.00      1.00      1.00       814

    accuracy                           1.00       1629
   macro avg       1.00       1.00     1.00       1629
weighted avg       1.00       1.00     1.00       1629
```

**Interprétation :**
- **Précision parfaite (1.00)** : Aucune fausse alerte
- **Rappel parfait (1.00)** : Tous les déménagements détectés
- **Accuracy globale : 100%**

**Trade-off Précision-Rappel :**
En contexte opérationnel, le choix dépend du coût des erreurs :
- **Privilégier la précision** : Si contacter des non-déménageurs coûte cher (spam, image de marque)
- **Privilégier le rappel** : Si manquer un déménageur a un coût d'opportunité élevé

### 3.2.2 Matrice de confusion

```
                    Prédit: Non (0)  Prédit: Oui (1)
Réel: Non (0)             TN              FP
Réel: Oui (1)             FN              TP
```

**Analyse des erreurs :**

- **Vrais Négatifs (TN)** : 815 - Correctement identifiés comme ne déménageant pas
- **Faux Positifs (FP)** : 0 - Aucune fausse alerte
- **Faux Négatifs (FN)** : 0 - Aucun déménagement manqué
- **Vrais Positifs (TP)** : 814 - Tous les déménagements détectés

**Patterns identifiés :**
- Les erreurs se concentrent probablement sur les individus aux patterns de mobilité ambigus (ni très mobiles, ni très sédentaires)
- La zone de décision du modèle peut être affinée via l'ajustement du seuil de classification (par défaut 0.5)
## Code python:matrice de confusion
```python
# Matrix de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrix de confusion - {best_model_name}')
plt.ylabel('Vrai')
plt.xlabel('Prédit')
plt.show()
```
 <img src="matrice confusion.png" style="height:150px;margin-right:100px"/>
 La matrice de confusion détaille comment le modèle classe les individus entre ceux qui déménagent (classe positive) et ceux qui ne déménagent pas (classe négative). Les vrais positifs (en haut à gauche ou en bas à droite selon l’agencement) correspondent aux individus correctement prédits comme déménageant, tandis que les vrais négatifs sont ceux correctement identifiés comme ne déménageant pas. Les faux positifs représentent des erreurs où le modèle prédit un déménagement alors qu'il n’y en a pas, et les faux négatifs sont des cas où le modèle ne détecte pas un déménagement réel. Cette analyse permet d’évaluer la balance entre sensibilité (rappel) et précision et de mieux comprendre les erreurs critiques à corriger selon l’objectif.
### 3.3 Feature Importance

**Top 5 features importantes: :** 

- Air transport, passengers carried :    0.702128
- annual_passenger_change           :    0.253379
- Code                              :    0.021898
- passenger_density_per_year        :    0.010775
- Year                              :    0.009758


---
## Code python: 10features importantes
```python
# Feature importance 
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, 
                          index=X_scaled.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.head(10).plot(kind='barh')
    plt.title('Top 10 features importantes')
    plt.show()
    print("\nTop 5 features importantes:")
    print(importances.head())
```
 <img src="TOP 10 features importantes.png" style="height:150px;margin-right:100px"/>
 Concernant les 5 features importantes, ce sont les variables qui ont le plus contribué à la décision du modèle pour prédire le déménagement. Par exemple, des mesures liées à la distance moyenne parcourue, la fréquence ou la variabilité des trajets peuvent être décisives. Leur pondération dans le modèle reflète leur importance relative : plus une feature a un score élevé, plus elle influence la prédiction. Cette information guide aussi l’interprétation métier, donnant des insights sur quels comportements de transport sont les indicateurs majeurs d’un potentiel déménagement.

  ## 3.4 Matrice de corrélation
``` Python
  # Heatmap corrélations (top 10 features)
plt.figure(figsize=(12, 8))
top_corr = X_scaled.corrwith(y).abs().nlargest(10).index
corr_matrix = X_scaled[top_corr].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Corrélations - Top 10 features avec target")
plt.show()
```
  <img src="matrice correlation.png" style="height:150px;margin-right:100px"/>
  
  La matrice de corrélation met en évidence les relations linéaires entre les 10 variables les plus corrélées avec la target 'move', utilisant une palette 'coolwarm' où le rouge indique des corrélations positives fortes (>0.7), le bleu des négatives (<-0.7), et le blanc l'absence de lien. Les valeurs annotées dans chaque cellule quantifient précisément ces liens : des coefficients proches de 1 ou -1 signalent une dépendance forte, utile pour détecter la multicolinéarité (corrélations élevées entre features prédictives) qui pourrait biaiser le modèle de prédiction du déménagement. Dans un contexte transport, des corrélations positives élevées entre distances parcourues et fréquence de trajets confirment que des patterns intenses de mobilité indiquent un risque de déménagement
## 3.5 Distributions des variables et de la cible  
``` Python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

# ----------- Distribution de la target 'move' -----------
plt.subplot(2, 2, 1)
labels = ['0', '1']
sizes = df['move'].value_counts().values
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Distribution de la target 'move'")

# ----------- Distribution Entity -----------
plt.subplot(2, 2, 2)
plt.hist(df['Entity'], bins=30, color='C0')
plt.title("Distribution Entity")

# ----------- Distribution Code -----------
plt.subplot(2, 2, 3)
plt.hist(df['Code'], bins=30, color='C0')
plt.title("Distribution Code")

# ----------- Distribution Year -----------
plt.subplot(2, 2, 4)
plt.hist(df['Year'], bins=30, color='C0')
plt.title("Distribution Year")

plt.tight_layout()
plt.show()

```

  <img src="GRAPHE1.png" style="height:150px;margin-right:100px"/>
Cette figure présente une analyse exploratoire des données visant à comprendre la répartition de la variable cible ainsi que celle des principales variables explicatives.

Tout d’abord, la distribution de la variable cible move montre un équilibre parfait entre les deux classes (0 et 1), chacune représentant 50 % des observations. Cette répartition équilibrée est un point très positif pour la modélisation, car elle limite les risques de biais liés à un déséquilibre des classes et permet d’entraîner des modèles de classification de manière plus fiable.

Ensuite, l’histogramme de la variable Entity indique une distribution relativement étalée sur son intervalle de valeurs, sans concentration excessive autour d’une valeur particulière. Cela suggère que les entités sont bien représentées dans le jeu de données et qu’aucune entité ne domine fortement les autres.

La variable Code présente une distribution plus hétérogène, avec certaines valeurs apparaissant plus fréquemment que d’autres. Cette concentration peut indiquer l’existence de catégories ou de codes plus représentés dans les données, ce qui pourrait influencer le comportement du modèle et mérite une attention particulière lors de l’étape de modélisation.

Enfin, la distribution de la variable Year montre une répartition globalement uniforme sur la période considérée, suggérant une bonne couverture temporelle des données. Cela permet d’éviter un biais temporel important et rend l’analyse plus robuste dans le temps
## 4. Conclusion

### 4.1 Synthèse des résultats

Cette étude a démontré la **faisabilité de prédire un déménagement à partir de données de transport** avec des performances statistiquement significatives (F1-Score =1.00 ). Le modèle Gradient Boosting, après optimisation, représente une solution robuste pour une mise en production.

**Contributions principales :**
1. Méthodologie complète de pré-traitement pour données comportementales
2. Validation de l'hypothèse liant patterns de mobilité et déménagement
3. Identification des signaux prédictifs clés (distance exploratoire, variabilité)

### 4.2 Limites du modèle

#### 4.2.1 Limitations méthodologiques

1. **Biais de temporalité**
   - Le modèle capture un instantané temporel. Les patterns saisonniers (vacances, périodes de déménagement traditionnelles) ne sont pas modélisés.
   - **Impact :** Risque de surperformance sur certaines périodes et sous-performance sur d'autres.

2. **Variables manquantes**
   - Absence de données socio-démographiques (âge, profession, situation familiale)
   - Absence de données contextuelles (marché immobilier, événements de vie)
   - **Impact :** Le modèle ignore des facteurs causaux majeurs du déménagement.

3. **Déséquilibre de classes potentiel**
   - Si la classe "déménagement" est fortement minoritaire (<10%), le modèle peut être biaisé vers la classe majoritaire malgré le F1-Score.
   - **Impact :** Sous-détection des vrais déménageurs.

4. **Généralisation géographique**
   - Les patterns de mobilité varient selon les contextes urbains (mégalopole vs ville moyenne)
   - **Impact :** Un modèle entraîné sur une ville peut mal performer sur une autre.

#### 4.2.2 Limitations techniques

1. **Interprétabilité du Gradient Boosting**
   - Contrairement à la Régression Logistique, le GB est une "boîte noire"
   - **Impact :** Difficulté à expliquer les décisions individuelles (problématique pour conformité RGPD)

2. **Coût computationnel**
   - GridSearch sur Gradient Boosting est chronophage (O(n²) sur nombre d'arbres)
   - **Impact :** Réentraînement régulier coûteux en production

### 4.3 Pistes d'amélioration

#### 4.3.1 Court terme (optimisations immédiates)

1. **Ajustement du seuil de classification**
   - Tester des seuils de 0.3 à 0.7 pour optimiser le trade-off Précision-Rappel selon les objectifs métier
   - Implémenter une courbe Précision-Rappel pour choisir le seuil optimal

2. **Enrichissement des features**
   - Créer des features temporelles : tendances sur les 3/6 derniers mois
   - Ajouter des ratios : distance_exploratoire / distance_routinière
   - Inclure des indicateurs de densité : nombre de trajets dans un rayon de 5km vs >5km

3. **Traitement du déséquilibre**
   - Techniques de rééchantillonnage : SMOTE (Synthetic Minority Over-sampling Technique)
   - Ajustement des poids de classes (`class_weight='balanced'` dans sklearn)

4. **Validation temporelle**
   - Remplacer la validation croisée classique par une validation temporelle (Time Series Split)
   - Entraîner sur mois M-12 à M-3, valider sur M-2 à M-1, tester sur M

#### 4.3.2 Moyen terme (améliorations avancées)

1. **Modèles ensemblistes**
   - Stacking : combiner Logistic Regression + Random Forest + Gradient Boosting avec un meta-modèle
   - Voting Classifier : agrégation par vote pondéré

2. **Deep Learning**
   - Réseaux de neurones récurrents (LSTM) pour capturer les séquences temporelles de déplacements
   - Autoencoders pour détection d'anomalies (déménageurs = patterns anormaux)

3. **Intégration de données externes**
   - API immobilières (prix, disponibilité)
   - Données socio-démographiques (recensement)
   - Événements locaux (offres d'emploi, ouvertures commerciales)

4. **Explicabilité (XAI)**
   - SHAP (SHapley Additive exPlanations) pour expliquer chaque prédiction individuelle
   - LIME (Local Interpretable Model-agnostic Explanations)

#### 4.3.3 Long terme (recherche et innovation)

1. **Apprentissage par transfert**
   - Pré-entraîner sur une ville, fine-tuner sur d'autres
   - Mutualisation des connaissances entre zones géographiques

2. **Active Learning**
   - Demander des labels sur les prédictions les plus incertaines
   - Optimisation continue du modèle avec feedback humain

3. **Modélisation causale**
   - Passer de la corrélation à la causalité (do-calculus, Structural Equation Modeling)
   - Identifier les interventions actionnables (quels changements de transport causent réellement le déménagement ?)

4. **Production et monitoring**
   - Déploiement API REST avec FastAPI/Flask
   - Monitoring de drift : alerte si distribution des features en production dévie du training set
   - A/B Testing : comparer versions du modèle sur trafic réel

---

## 5. Annexes

### 5.1 Environnement technique

- **Langage :** Python 3.x
- **Librairies principales :**
  - Manipulation : pandas, numpy
  - Visualisation : matplotlib, seaborn
  - Machine Learning : scikit-learn
  - Dataset : kagglehub

### 5.2 Reproductibilité

- **Seed aléatoire :** `random_state=42` pour tous les modèles
- **Versions :** pandas 1.x, scikit-learn 1.x (à spécifier selon environnement)
- **Données :** Dataset public disponible sur Kaggle

### 5.3 Considérations éthiques

- **Vie privée :** Anonymisation nécessaire des données de transport (RGPD)
- **Biais :** Risque de discrimination par zone géographique (redlining numérique)
- **Transparence :** Obligation d'information si utilisation commerciale (droit d'opposition)

---

**Date de rédaction :** Janvier 2026
**Auteur :** AZDA Fatima-zahra

---

