import pandas as pd
import numpy as np
X2 = pd.DataFrame({'a': range(5),'b': [-100, -50, 0, 200, 1000],})
print(X2)

'''Certains algorithmes, tels que le SVM, 
   fonctionnent mieux lorsque les données sont normalisées. 
   Chaque colonne doit avoir une valeur moyenne de 0 et un écart-type de 1. 
   Sklearn fournit une méthode de transformation 
   .fit qui combine à la fois .fit et .transform :
'''
#Normalisation

from sklearn import preprocessing
std = preprocessing.StandardScaler()
print(std.fit_transform(X2))

print("\n -------------------")
print(std.scale_)
print("\n --------------------")
print(std.mean_)
print("\n --------------------")
print(std.var_)

'''
   Voici une version pour les pandas. 
   N'oubliez pas que vous devrez suivre la moyenne et 
   l'écart-type d'origine si vous utilisez cette version pour le prétraitement. 
   Tout échantillon que vous utiliserez pour faire des prédictions par la suite 
   devra être normalisé avec ces mêmes valeurs :
'''


print("\n --------------------")
X_std = (X2 - X2.mean()) / X2.std()
print(X_std)
print("\n --------------------")
print(X_std.mean())
print("\n --------------------")
print(X_std.std())

# Dummy Variables

'''
   Nous pouvons utiliser les pandas pour créer des variables factices à partir de données catégorielles. 
   C'est ce que l'on appelle le codage en une seule fois ou codage indicateur. 
   Les variables factices sont particulièrement utiles si les données sont nominales (non ordonnées). 
   La fonction get_dummies dans les pandas crée plusieurs colonnes pour une colonne catégorielle, 
   chacune avec un 1 ou un 0 si la colonne originale avait cette valeur :
'''
print("\n --------------------")
X_cat = pd.DataFrame({'name': ['George', 'Paul'],'inst': ['Bass', 'Guitar'],})
print(X_cat)

'''
    Voici la version des pandas. Notez que l'option drop_first peut être utilisée pour éliminer une colonne 
    (une des colonnes factices est une combinaison linéaire des autres colonnes) :
'''
print("\n --------------------")
gd = pd.get_dummies(X_cat, drop_first=True)
print(gd)


X_cat2 = pd.DataFrame({'A': [1, None, 3],'names': ['Fred,George','George','John,Paul',],})
print("\n --------------------")
print(jn.expand_column(X_cat2, 'names', sep=','))