# Projet_Pr-diction-du-Churn-Client

analyse des données 
d'après les statistiques descriptives 
on a que la Moyenne = 0.162 → environ 16 % des clients sont des seniors.
La majorité des clients ne sont pas des seniors. Cette information peut influencer le churn si certains profils (ex : seniors) ont des comportements différents.
si peur ca on peux choisir 'SeniorCitizen ' comme une variable / feature X 

#### 🔹 SeniorCitizen
| Statistique | Valeur |
|--------------|--------|
| Moyenne | 0.162 |
| Écart-type | 0.369 |
| Minimum | 0 |
| Maximum | 1 |

**Interprétation :**  
- Environ **16 %** des clients sont des seniors.  
- La majorité des clients (84 %) ne sont **pas seniors**.  
- Cette variable binaire (0 = non, 1 = oui) pourrait influencer le churn si les comportements des seniors diffèrent.

---

#### 🔹 Tenure (ancienneté du client en mois)
| Statistique | Valeur |
|--------------|--------|
| Moyenne | 32.37 |
| Médiane | 29.00 |
| Écart-type | 24.56 |
| Minimum | 0 |
| Maximum | 72 |

**Interprétation :**  
- L’ancienneté moyenne est d’environ **32 mois (≈ 2,7 ans)**.  
- Une **forte variabilité** indique des clients à différents stades de fidélité.  
- Certains clients viennent tout juste d’arriver (**tenure = 0**), tandis que d’autres sont présents depuis **6 ans**.  
- Les **clients récents** pourraient être **plus susceptibles de se désabonner**.

---

#### 🔹 MonthlyCharges (montant mensuel)
| Statistique | Valeur |
|--------------|--------|
| Moyenne | 64.76 |
| Médiane | 70.35 |
| Écart-type | 30.09 |
| Minimum | 18.25 |
| Maximum | 118.75 |

**Interprétation :**  
- Le montant moyen facturé est d’environ **65 unités** (devise non précisée).  
- Les **tarifs sont très dispersés**, ce qui traduit une **diversité d’offres**.  
- Les **clients avec des charges élevées** pourraient être **plus enclins à résilier**, surtout s’ils perçoivent un rapport qualité/prix insuffisant.

---

### 🧠 Synthèse
- La population étudiée est **majoritairement non senior**.  
- L’**ancienneté moyenne** est de 2 à 3 ans, mais beaucoup de nouveaux clients sont présents.  
- Les **montants mensuels varient fortement**, ce qui reflète la segmentation tarifaire.  

👉 Ces variables sont **potentiellement influentes** dans la prédiction du **churn**, notamment :  
- `tenure` (fidélité),  
- `MonthlyCharges` (niveau de dépenses),  
- `SeniorCitizen` (profil démographique).


j'ai installer : 
# Librairies de base
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
aussi 
pandas

scikit-learn
<!-- La matrice de correlation -->
Interprétation

tenure (-0.35) → corrélation négative modérée
→ Plus un client reste longtemps (tenure élevé), moins il y a de chance qu’il résilie (churn = 1).
C’est un facteur protecteur : la fidélité réduit le churn.

TotalCharges (-0.20) → légère corrélation négative
→ Plus un client a dépensé beaucoup, moins il quitte le service.

MonthlyCharges (+0.19) → légère corrélation positive
→ Plus le coût mensuel est élevé, plus la probabilité de churn augmente.
(Les clients paient plus → peuvent être insatisfaits ou partir.)

SeniorCitizen (+0.15) → légère corrélation positive
→ Les clients âgés semblent un peu plus susceptibles de partir.
Les variables les plus corrélées avec le churn sont :

*tenure (-0.35) – plus fort lien*

*TotalCharges (-0.20)*

*MonthlyCharges (+0.19)*

*SeniorCitizen (+0.15)*

💡 Conclusion :

Les clients récents, qui paient plus cher, et qui n’ont pas encore accumulé beaucoup de dépenses sont plus susceptibles de résilier.
Les anciens clients, eux, ont tendance à rester.
pipeline : phase : importation des bibliotheque


la phase des scores_ 
leur interpretation 
#  Scores de base
| Score         | Valeur   | Interprétation                                                  |
| ------------- | -------- | --------------------------------------------------------------- |
| **Accuracy**  | `0.7821` | 🟢 *Correct (78% de bonnes prédictions)*                        |
| **Recall**    | `0.4638` | 🔴 *Mauvais : le modèle détecte mal les clients qui churnent*   |
| **Precision** | `0.6179` | 🟠 *Pas mal : quand il prédit churn, il se trompe 38% du temps* |
| **F1 Score**  | `0.5299` | 🔴 *Faible : compromis entre précision et rappel pas très bon*  |


