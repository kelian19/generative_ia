# Modèles génératifs de risque climatique — copie corrigée

[![Tests](https://github.com/kelian19/generative_ia/actions/workflows/tests.yml/badge.svg)](https://github.com/kelian19/generative_ia/actions/workflows/tests.yml)

Copie de [Kelian2909/generative_ia](https://github.com/Kelian2909/generative_ia), commit `bca170c92297e2155c28663772c2bd9914987e1f`. L’historique du dépôt source et le fichier fourni sont conservés ; les corrections sont dans **kelian19/generative_ia**.

Ce projet compare VAE, VAE pondéré, GAN, WGAN-GP et cGAN sur un indicateur C-Haines approché depuis les champs ERA5 fournis. **Il s’agit d’une étude de distributions scalaires sur juillet 2023, sans validation de prime annuelle ou de scénarios de feux.**

## Installation CPU reproductible

Python 3.12 recommandé :

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q
python -m src.run_experiment --epochs 10 --n-samples 10000 --seed 100
```

Sous Windows, activer `.venv\Scripts\activate`. Le fichier requirements utilise l’index CPU officiel PyTorch pour la version `+cpu` figée. Le fichier NetCDF provient du dépôt source ; le pipeline enregistre son empreinte et ne le modifie pas.

## Corrections principales

- C-Haines avec point de rosée approché, à la place du proxy initial ; humidités hors domaine masquées et comptées.
- Séparation par dates avec embargo ; prétraitement ajusté uniquement sur train.
- Cible standardisée et sorties non bornées par Sigmoid/MinMax.
- VAE gaussien avec bruit d’observation ; objectif pondéré explicite.
- GAN/cGAN à logits ; WGAN-GP sans pénalité artificielle de dispersion, pénalité de gradient compatible device/dtype et compteur global de mises à jour.
- Conditions spatiales normalisées et mois cyclique ; domaines inconnus refusés.
- Coût paramétrique exprimé **par maille / pas de temps**, sans annualisation implicite.
- Notebook nettoyé des anciens chiffres et conclusions non validées ; tests et benchmarks reproductibles.

[Audit détaillé](reports/AUDIT.md) · [Résultats mesurés](reports/RESULTATS.md) · [Benchmark JSON](reports/benchmark.json)

## Données et protocole

Le fichier couvre 31 jours de juillet 2023, à 12 h, sur 41 × 61 mailles. Les 1 839 mailles-dates de RH hors ]0,100] au niveau 850 hPa sont masquées. Toutes les mailles d’une même date restent dans le même échantillon.

- Apprentissage : 1–18 juillet, 44 332 observations.
- Validation : 20–24 juillet, 12 097 observations.
- Test descriptif : 26–31 juillet, 14 410 observations.
- Les 19 et 25 juillet constituent l’embargo.

La sélection de stabilité WGAN s’appuie uniquement sur la validation. Le test ayant été vu lors du diagnostic initial, son résultat final n’est pas présenté comme une nouvelle validation indépendante. Les essais divergents sont conservés dans `reports/benchmark_initial.json` et les essais de validation dans `reports/wgan_*trials*.json`.

Les références simples sont un rééchantillonnage empirique de train et une normale ajustée sur train. Aucun gain n’est présumé pour un réseau. Le quantile 99 % de C-Haines n’est pas une VaR de pertes ; les p-values KS ne justifient pas de conclusions iid sur des mailles dépendantes.

## Notebook et artefacts

`notebooks.ipynb` présente le protocole et charge les résultats de benchmark. La cellule d’entraînement complet est fournie explicitement. Aucune ancienne sortie n’est embarquée. Les poids générés sont dans `artifacts/*.pt`, exclus de Git ; les paramètres de standardisation sont exportés séparément.

## Jeu multi-annuel : ERA5 2001–2024 et feux observés

Pour dépasser le mois unique, un second jeu est reconstruit depuis des sources publiques sans clé : ERA5 (via ARCO-ERA5, lecture partielle des seules mailles utiles) et les détections de feux de végétation NASA FIRMS MODIS C6.1. Juin à septembre, 12 UTC, même boîte, mailles terrestres sous 1 200 m ; le point de rosée vient de l’humidité spécifique, sans troncature de l’humidité relative.

```sh
python -m scripts.build_multiyear_dataset     # ~30 min, réseau ; écrit data/era5_firms_jjas_2001_2024.nc (34 Mo, hors Git)
python -m src.run_multiyear --epochs 10 --seed 100
python -m scripts.make_figures && python -m scripts.make_multiyear_section
```

Découpage par années entières : 2001–2016, 2017–2020, 2021–2024. Résultats : [RESULTATS_MULTIANNUEL.md](reports/RESULTATS_MULTIANNUEL.md). En bref : au seuil de démonstration (quantile 95 % de l’apprentissage), la probabilité d’au moins un feu par maille-jour est de 0,57 % au-dessus du seuil contre 0,22 % en dessous sur le test (surcote ×2,4, mais seulement ×1,4 sur l’apprentissage) ; le seuil capture 12 % des maille-jours avec feu. Le classement des réseaux diffère de celui de juillet 2023 (le GAN y était le meilleur, il est ici le moins bon). Une graine ; les détections FIRMS ne sont pas des pertes assurées.

## Robustesse à la graine

Le benchmark de juillet 2023 reposait sur une seule graine. `python -m scripts.multi_seed` le relance à l'identique pour 5 graines (100 à 104) : [RESULTATS_MULTIGRAINES.md](reports/RESULTATS_MULTIGRAINES.md).

| Méthode | Wasserstein validation (moyenne ± écart-type) | Wasserstein test | Fois 1er en test |
|---|---:|---:|---:|
| Référence empirique | **0,295 ± 0,011** | 0,590 ± 0,020 | 1 |
| GAN | 0,312 ± 0,063 | **0,453 ± 0,175** | 4 |
| VAE | 0,339 ± 0,018 | 0,636 ± 0,062 | 0 |
| WGAN-GP | 0,342 ± 0,034 | 0,624 ± 0,081 | 0 |
| cGAN | 0,440 ± 0,118 | 0,723 ± 0,315 | 0 |

En validation, aucun réseau ne bat en moyenne la référence empirique. En test, le GAN est le plus souvent premier, mais avec une dispersion forte. Le très bon résultat du cGAN avec la graine 100 (0,374) n'est pas représentatif : sa moyenne est de 0,723. Relancée sur une autre machine, la graine 100 redonne exactement les références, mais le cGAN change fortement : la graine ne garantit la reproductibilité que sur une même machine.

## Limites scientifiques

Le benchmark de juillet 2023 (un seul mois) ne permet ni apprentissage saisonnier, ni validation de risque climatique à long terme. Le cGAN conditionne des échantillons scalaires sur la localisation et le mois ; il ne génère pas un champ atmosphérique conjoint. L’extrapolation numérique ne garantit pas la plausibilité d’un extrême physique. Les liens entre indice, départs de feux et pertes ne sont pas observés. Une tarification annuelle demanderait notamment un historique plus long, des garanties, des expositions et une structure de dépendance temporelle/spatiale.

## Note méthodologique à partager

[PDF — modèles, mathématiques et validation](docs/latex/note_modeles_generatifs.pdf) · [Source LaTeX](docs/latex/note_modeles_generatifs.tex).

La note comporte dix pages et sept figures (régénérables avec `python -m scripts.make_figures`). Compilation : deux passages de `pdflatex note_modeles_generatifs.tex` (ou `tectonic -X compile`). Les résultats de benchmark sont ceux du protocole documenté, sans validation opérationnelle prétendue.
