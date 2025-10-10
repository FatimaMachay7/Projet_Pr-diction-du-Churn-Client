# EDA — Prédiction du Churn Client (désabonnement) – Projet Data Science 

### *Auteur :*__MACHAY Fatima__
### *Date :* __2025-10-06__
### *Objectif :*

Développer un pipeline complet de Machine Learning supervisé permettant de prédire le désabonnement des clients (churn) au sein d’une entreprise de télécommunications, dans le but de mieux cibler les actions de fidélisation et de réduire la perte de clientèle. Parallèlement, analyser et explorer les données clients afin d’identifier les facteurs clés qui influencent le risque de désabonnement.


## Présentation du Projet :

L'objectif de ce projet est de prédire le churn des clients dans une entreprise de télécommunications, en utilisant des techniques d'apprentissage automatique. Le but principal est de construire un modèle prédictif capable de déterminer si un client va se désabonner (churner) en fonction de ses attributs. Ce projet utilise plusieurs algorithmes de classification et évalue la performance des modèles avec des métriques telles que la précision, le recall, le F1-score et l'AUC (Area Under the Curve).

## 📑 Table des matières :

- [Gestion de projet](#gestion-de-projet)
- [Modèles implémentés](#modèles-implémentés)
- [Installation](#installation)
- [Données](#données)
- [Prétraitement des Données](#Prétraitement-des-Données)
- [Modélisation](#modélisation)
- [Entraînement et Évaluation des Modèles](#Entraînement-et-Évaluation-des-Modèles)
- [Exécution du projet](#exécution-du-projet)
- [Tests](#tests)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Présentation :

Le churn client désigne le phénomène où les clients cessent d'utiliser les services d'une entreprise. Pour les entreprises, prédire ce churn est crucial afin de mettre en place des stratégies de fidélisation et de réduction de la perte de clients. Dans ce projet, nous appliquons des algorithmes d'apprentissage automatique pour prédire si un client va churner, en fonction de diverses caractéristiques, telles que la démographie, les détails d'abonnement, et son comportement d'utilisation.

## 🧩 Gestion de projet :

La planification initiale a été réalisée avec [Jira](https://infofatimamachay.atlassian.net/jira/software/projects/KAN/list), afin de structurer les tâches, suivre l’avancement et mieux organiser les différentes étapes du développement.

### Modèles Implémentés :

- Classificateur Random Forest; 
- Régression Logistique; 
- Support Vector Classifier (SVC).

## Installation :

Pour démarrer avec ce projet, suivez les étapes ci-dessous :

__Clonez le dépôt :__

[git clone](https://github.com/FatimaMachay7/Projet_Pr-diction-du-Churn-Client.git)

__Accédez au répertoire du projet :__
      [cd]    Projet_Pr-diction-du-Churn-Client 

1. Installez les dépendances requises :

2. Installez les dépendances avec la commande `pip install -r requirements.txt`.
3. Installez manuellement les dépendances suivantes :

pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pytest

## 📁 Données :

Les données utilisées dans ce projet proviennent du dataset Customer Churn Prediction. Ce dataset contient des informations concernant les clients : démographie, détails d'abonnement, comportement d'utilisation, et une colonne "Churn" qui indique si le client a churné (Oui) ou non (Non).

Les données sont disponibles ici :

[Voir le fichier Churn.csv](./DATA/churn.csv)


## Prétraitement des Données :


*EDA — Exploration des Données (Données clients et Churn) :*

L'Exploration des Données (EDA) est une étape essentielle pour analyser la distribution des variables et leurs interactions. Elle permet de mieux identifier les facteurs clés qui influencent le churn des clients. Cette analyse détaillée a été réalisée à l’aide d’un *notebook Jupyter*, offrant un aperçu complet du dataset.

*Analyse Descriptive des Données :* 

Les statistiques descriptives des variables principales sont fournies ci-dessous :

------------------------->`data.describe()`<--------------------------------------------------


| Feature         | SeniorCitizen | tenure     | MonthlyCharges |
|----------------|---------------|------------|----------------|
| **count**       | 7043.000000   | 7043.000000| 7043.000000    |
| **mean**        | 0.162147      | 32.371149  | 64.761692      |
| **std**         | 0.368612      | 24.559481  | 30.090047      |
| **min**         | 0.000000      | 0.000000   | 18.250000      |
| **25%**         | 0.000000      | 9.000000   | 35.500000      |
| **50% (median)**| 0.000000      | 29.000000  | 70.350000      |
| **75%**         | 0.000000      | 55.000000  | 89.850000      |
| **max**         | 1.000000      | 72.000000  | 118.750000     |

L'Exploration des Données (EDA) inclut l’analyse des distributions et des relations entre variables, ainsi que des *visualisations* pour mieux comprendre les données. Les *histogrammes* sont utilisés pour les variables numériques, tandis que les *countplots* sont privilégiés pour les variables catégorielles. Les *subplots* permettent de comparer plusieurs visualisations simultanément. Ces outils permettent d’identifier des patterns, des anomalies et d'analyser les variables avant l'entraînement du modèle.

Voici un graphique montrant l'évolution du churn des clients :

__Histogramme de la variable *Tenure* :__
![Graphique du tenure](Graphes_EDA/histogramme_tenure.png)

__Histogramme de la  variable *MonthlyCharges* :__
![Graphique du MonthlyCharges](Graphes_EDA/histogramme_MonthlyCharges.png)

__Subplot comparant *les variables catégorielles :*__

![Subplot comparant les variables catégorielles](Graphes_EDA/count_polt.png)

__📈 Matrice de corrélation : compréhension des liens entre les variables :__

![La matrice de corrélation](Graphes_EDA/matrice_correlation.png)

*Relations entre les Variables  :*

L'Analyse Exploratoire des Données (EDA) permet d'étudier les relations entre les variables et de préparer les données pour les modèles de machine learning. Les données ont été chargées à l'aide de Pandas, et la variable cible sélectionnée est churn. Les variables gender, seniorCitizen, partner, et customerID ont été exclues en raison de leur faible influence sur la prédiction du churn. L'encodage des variables catégorielles, ainsi que de churn et TotalCharges, a été effectué à l'aide de Label Encoding afin de rendre ces données compatibles avec les modèles de machine learning.

Interprétation de la matrice de corrélation : La matrice révèle des relations significatives entre certaines variables, telles que la corrélation entre charges mensuelles et tenure, ce qui permet de mieux orienter la sélection des features.


*Séparation Train-Test :* Le jeu de données est divisé en un ensemble d’entraînement et un ensemble de test avec train_test_split.

*Normalisation des Données :* Après la séparation des données en ensembles d'entraînement et de test, nous appliquons une normalisation pour uniformiser l'échelle des caractéristiques. Cela est fait avec MinMaxScaler de sklearn, qui redimensionne les valeurs des variables dans un intervalle spécifié, généralement entre [0, 1]. Cette étape garantit que toutes les caractéristiques sont sur une échelle comparable.


## Modélisation :

Trois modèles sont utilisés pour la prédiction du churn :

- Classificateur Random Forest; 
- Régression Logistique; 
- Support Vector Classifier (SVC).

## Entraînement et Évaluation des Modèles :

Chaque modèle est évalué sur des métriques telles que :

- Précision (Accuracy); 
- Précision (Precision); 
- Rappel (Recall);
- F1-Score;
- ROC- curve.

*Décision basée sur la comparaison des modèles :*

Après évaluation des trois modèles (*Random Forest, Régression Logistique, SVC*) sur des métriques clés, voici les résultats :

*Régression Logistique* excelle en rappel (0.8284), idéale pour identifier les churners (minimiser les faux négatifs).

*SVC* se distingue par la meilleure ROC-AUC (0.84), offrant une bonne discrimination entre churn et non-churn.

*Random Forest* a la meilleure accuracy (0.7828), mais un rappel plus faible (0.4665), ce qui en fait un modèle équilibré pour des prédictions globales.

Le meilleur modèle que j'ai choisi est la *Régression Logistique*, car elle offre le meilleur compromis entre rappel et F1-Score, ce qui est crucial pour ce projet.

## Exécution du Projet :


- Ouvrez le fichier Data_Churn.ipynb et exécutez les cellules dans l’ordre.

- Le notebook entraînera les trois modèles : Régression Logistique, Random Forest et SVC.

- Le notebook effectuera les étapes suivantes :
*Chargement des données;*
*Prétraitement des données;*
*Entraînement de chaque modèle;*
*Évaluation de la performance de chaque modèle;*
*Tracé des courbes pour la comparaison.*

## Tests :

Le projet inclut des tests unitaires pour vérifier :

- La cohérence des dimensions entre les variables d’entraînement et de test;
- L’évaluation correcte des modèles.

_Pour exécuter les tests :_

*pytest*

Cela exécutera tous les tests dans le répertoire tests/.

## Contribuer :

Les contributions sont les bienvenues ! Si vous trouvez un bug ou souhaitez améliorer le projet, n’hésitez pas à forker le dépôt et soumettre une demande de pull.

Pour contribuer :

- Forkez le dépôt.
- Créez une nouvelle branche.
- Effectuez vos modifications.
- Soumettez une demande de pull.

## Licence :

_Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails._