"""Paiement paramétrique par observation ; aucune annualisation implicite."""
import numpy as np
from .evaluation import finite_values

def compute_prime_pure(generated,trigger,indemnite):
    x=finite_values(generated)
    if trigger is None:raise ValueError('Seuil explicite requis pour la définition corrigée de l’indicateur.')
    if not np.isfinite(trigger) or not np.isfinite(indemnite) or indemnite<0:raise ValueError('Seuil fini et indemnité positive ou nulle requis.')
    probability=float(np.mean(x>trigger))
    return probability,probability*float(indemnite)

def print_actuarial_report(generated,trigger,indemnite,label='Modèle'):
    probability,cost=compute_prime_pure(generated,trigger,indemnite)
    print(f'{label}: P(indice > {trigger}) = {probability:.3%}; coût espéré par observation = {cost:.2f}.')
    print('Aucune prime annuelle : unité = maille / pas de temps, dépendances spatiales et temporelles non modélisées.')
    return probability,cost
