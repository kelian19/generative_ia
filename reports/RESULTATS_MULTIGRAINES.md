# Robustesse à la graine — benchmark de juillet 2023

Même protocole, mêmes données et même budget que [RESULTATS.md](RESULTATS.md) ; seule la graine change (100, 101, 102, 103, 104). Wasserstein en unités de C-Haines, plus faible = meilleure proximité marginale ; les références (`empirical_train`, `gaussian_train`) ne dépendent de la graine que par le tirage des échantillons.

Reproductibilité de la graine 100 (relancée sur Windows-11-10.0.26300-SP0) contre le benchmark publié, écart maximal de Wasserstein par méthode : empirical_train 0.0000, gaussian_train 0.0000, vae 0.0031, vae_weighted 0.0000, gan 0.0171, wgan_gp 0.0017, cgan 0.4344. Les références sont identiques ; les réseaux varient légèrement d'une plateforme à l'autre (arithmétique flottante), et le cGAN fortement : son entraînement adverse amplifie ces écarts minimes. La graine fixe la reproductibilité sur une même machine, pas d'une machine à l'autre.

## Validation (5 graines)

| Méthode | Wasserstein moyenne | Écart-type | Min – max | Rang moyen | Fois 1er |
|---|---:|---:|---:|---:|---:|
| empirical_train | 0.2952 | 0.0112 | 0.2776 – 0.3059 | 1.4 | 3 |
| gan | 0.3121 | 0.0633 | 0.2463 – 0.4140 | 2.2 | 2 |
| vae | 0.3389 | 0.0180 | 0.3134 – 0.3590 | 4.0 | 0 |
| wgan_gp | 0.3416 | 0.0342 | 0.3078 – 0.3905 | 3.6 | 0 |
| vae_weighted | 0.3538 | 0.0195 | 0.3377 – 0.3854 | 4.6 | 0 |
| cgan | 0.4398 | 0.1179 | 0.3155 – 0.5702 | 5.2 | 0 |
| gaussian_train | 0.6479 | 0.0087 | 0.6389 – 0.6572 | 7.0 | 0 |

## Test déjà consulté (5 graines, lecture descriptive)

| Méthode | Wasserstein moyenne | Écart-type | Min – max | Rang moyen | Fois 1er |
|---|---:|---:|---:|---:|---:|
| gan | 0.4529 | 0.1753 | 0.2904 – 0.7293 | 1.6 | 4 |
| empirical_train | 0.5895 | 0.0196 | 0.5687 – 0.6215 | 2.6 | 1 |
| wgan_gp | 0.6235 | 0.0809 | 0.5050 – 0.7253 | 3.4 | 0 |
| vae | 0.6363 | 0.0623 | 0.5669 – 0.7277 | 3.6 | 0 |
| cgan | 0.7234 | 0.3147 | 0.3227 – 1.1608 | 4.4 | 0 |
| gaussian_train | 0.7962 | 0.0371 | 0.7642 – 0.8553 | 6.2 | 0 |
| vae_weighted | 0.8126 | 0.0390 | 0.7573 – 0.8507 | 6.2 | 0 |

## Stabilité du classement

Meilleur modèle en validation selon la graine : empirical_train (3), gan (2). Meilleur en test : gan (4), empirical_train (1). Corrélation de rang moyenne entre validation et test : 0.58.

Lecture : l'écart entre deux réseaux n'est interprétable que s'il dépasse la dispersion due à la graine. Un classement obtenu sur une seule graine, comme celui de RESULTATS.md, ne suffit pas à désigner un meilleur modèle.
