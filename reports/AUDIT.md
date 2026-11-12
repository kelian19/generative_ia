# Audit et corrections — copie generative_ia

Source : https://github.com/Kelian2909/generative_ia, commit `bca170c92297e2155c28663772c2bd9914987e1f`. L’historique original est conservé dans la copie ; les corrections ne sont pas poussées vers le compte source.

## Problèmes constatés et traitement

1. **Indicateur mal nommé.** La formule `(T850−T700)+(100−RH850)/5` était un proxy, pas le C-Haines annoncé. Le chargement calcule désormais les composantes CA/CB de Mills–McCaw avec point de rosée approché par Magnus et les corrections de sécheresse. Le proxy reste disponible explicitement via `method='legacy_proxy'`, sans confusion d’échelle ou de seuil.
2. **Qualité RH.** 1 839 mailles-dates ont une RH hors ]0,100] au niveau 850 hPa dans le fichier fourni. Elles sont masquées et comptées, pas tronquées en observations valides. Cette sélection est documentée, sans prétendre que toute supersaturation est physiquement impossible.
3. **Fuite de données.** Le notebook ajustait Box-Cox/MinMax sur tout le jeu puis évaluait sur le même historique. Les dates sont maintenant séparées chronologiquement, toutes leurs mailles restant ensemble. Un embargo d’un jour est appliqué à chaque frontière ; les transformations sont ajustées sur train uniquement.
4. **Borne artificielle des simulations.** MinMax + Sigmoid limitait les sorties aux bornes apprises. La cible est standardisée et les sorties sont linéaires. L’extrapolation devient possible, sans être présentée comme une validation des extrêmes physiques.
5. **VAE continu.** Une vraisemblance gaussienne avec écart-type appris remplace la simple reconstruction MSE. La génération inclut le bruit d’observation, et pas seulement les moyennes du décodeur. Le KL est explicite. Le modèle pondéré est identifié comme objectif modifié, pas comme une densité de queue calibrée.
6. **GAN/cGAN.** Logits + BCEWithLogitsLoss remplacent Sigmoid + BCELoss pour la stabilité numérique ; la formulation initiale était valide mais moins stable. Le discriminateur est gelé pendant la mise à jour du générateur. Les pertes non finies provoquent une erreur.
7. **WGAN-GP.** Suppression de la pénalité réciproque de l’écart-type, qui encourageait artificiellement la dispersion et pouvait produire un NaN avec un batch de taille 1. La pénalité de gradient utilise les mêmes dtype/device que les données. Les mises à jour critique/générateur suivent un compteur global. Le générateur part d’une référence gaussienne avec un résidu appris initialisé à zéro.
8. **Stabilité WGAN.** Le premier entraînement corrigé restait divergent. Son benchmark est conservé. Plusieurs variantes sont comparées sur validation seulement ; les paramètres retenus sont documentés. Le test ayant déjà été consulté, le résultat final reste diagnostique et ne constitue pas une nouvelle évaluation indépendante.
9. **Conditions du cGAN.** Latitude/longitude standardisées sur train ; mois encodé par sinus/cosinus. Les mois et coordonnées hors domaine appris sont refusés par l’encodeur. Comme seul juillet est observé, aucune saisonnalité annuelle n’est prétendue apprise. Ces conditions ne rendent pas les échantillons scalaires spatialement cohérents.
10. **Génération et évaluation.** Tirages à graine locale, conservation du mode train/eval, passage explicite CPU pour NumPy, refus des valeurs non finies et mise à plat des deux échantillons. Le quantile d’un indicateur n’est pas qualifié de VaR de pertes ; les anciens noms de clés `var_99_*` restent des alias compatibles.
11. **Coût paramétrique.** `indemnité × P(indice > seuil)` est un coût par observation de maille/temps, pas une prime annuelle. Le seuil hérité 25 n’est pas reconduit après changement de définition ; le benchmark choisit un seuil de démonstration, quantile 95 % de train, sans validation contractuelle.
12. **Notebook.** Les anciens graphiques, chiffres et conclusions de « production »/« stress-test validé » sont retirés. Le notebook utilise les nouvelles fonctions et les résultats du protocole décrit, sans conserver d’anciennes sorties.

## Validation et limites

Les tests couvrent formule de l’indice, masque des RH, séparation par dates, transformations, sorties non bornées, vraisemblance et bruit VAE, pénalité WGAN analytique, batches de taille 1, compteur de mises à jour, conditions et domaine, reproductibilité CPU, métriques et unité du paiement.

Le fichier contient 31 dates de juillet 2023 à 12 h, sur 41 × 61 mailles. Les observations sont dépendantes spatialement et temporellement ; les p-values KS iid restent descriptives. Il n’existe ni pertes d’assurance, ni départs de feux, ni historique annuel. Le relief peut rendre des niveaux de pression sous-surface ; aucun masque orographique n’est fourni. Le point de rosée est une approximation de Magnus, pas une extraction directe ERA5.

Le projet génère des distributions scalaires, pas des champs météorologiques cohérents. Les sorties hors du maximum historique ne constituent pas une preuve de plausibilité de scénarios rares. Une étude d’assurance exige plusieurs saisons/années, garanties, portefeuille et validation sur événements et pertes observés.

## Références

- Mills et McCaw (2010), formule du C-Haines : https://library.dbca.wa.gov.au/FullTextFiles/921091.pdf
- Kingma et Welling, VAE : https://arxiv.org/abs/1312.6114
- Goodfellow et al., GAN : https://arxiv.org/abs/1406.2661
- Gulrajani et al., WGAN-GP : https://arxiv.org/abs/1704.00028
