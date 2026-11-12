# Résultats multi-annuels (ERA5 + FIRMS, JJAS 2001–2024)

Fichier `era5_firms_jjas_2001_2024.nc` (SHA256 `aa9873f2fe1160d72f74811a3101e8b65315dd670b912f05fe12e9828eb3b00b`). Apprentissage 2001–2016, validation 2017–2020, test 2021–2024 ; seuil de démonstration 8,65 (quantile 95 % de l'apprentissage). Graine 100, 10 époques, sous-échantillon d'apprentissage de 200 000 mailles-jours.

| Méthode | Wasserstein val. | Wasserstein test | Q99 généré (test) | P(indice > seuil) test |
|---|---:|---:|---:|---:|
| Référence empirique | 0,160 | 0,207 | 10,41 | 4,83 % |
| Référence normale | 0,255 | 0,274 | 9,81 | 2,79 % |
| VAE | 0,155 | 0,208 | 10,27 | 4,12 % |
| VAE pondéré | 0,077 | 0,123 | 10,61 | 4,68 % |
| GAN | 0,540 | 0,593 | 9,71 | 3,12 % |
| WGAN-GP | 0,227 | 0,237 | 10,72 | 4,97 % |
| cGAN | 0,218 | 0,266 | 10,60 | 4,71 % |

Q99 observé sur le test : 10,77 ; fréquence observée de dépassement : 5,13 %.

## Lien avec les feux observés (FIRMS MODIS, type 0, confiance ≥ 30 %)

| Période | Taux de feu (maille-jour) | Au-dessus du seuil | Sous le seuil | Surcote | Part des feux capturée |
|---|---:|---:|---:|---:|---:|
| Apprentissage | 0,42 % | 0,59 % | 0,41 % | ×1,4 | 7 % |
| Validation | 0,23 % | 0,41 % | 0,23 % | ×1,7 | 8 % |
| Test | 0,24 % | 0,57 % | 0,22 % | ×2,4 | 12 % |

Corrélation de rang saisonnière (24 saisons) entre maille-jours au-dessus du seuil et détections : 0,57 (p = 0,004, indicative : mailles dépendantes).

## Limites

- Détections MODIS de feux de végétation (type 0, confiance ≥ 30 %) : inclut des brûlages agricoles, pas des pertes assurées.
- Indice à 12 UTC ; les feux sont comptés sur la journée UTC entière.
- Mailles dépendantes spatialement et temporellement : p-values KS et corrélations de rang indicatives.
- Distributions scalaires par maille-jour, pas des champs cohérents.
- Le seuil 95 % est un seuil de démonstration, pas un déclencheur contractuel optimisé.
- La part de détections retenues par le filtre passe d’environ 77 % (2001–2016) à 63 % (2021–2024), d’après le journal de construction ; non analysée.
- Une seule graine : les écarts entre modèles ne constituent pas une supériorité établie. Le classement diffère de celui de juillet 2023 (le GAN y était le meilleur).
