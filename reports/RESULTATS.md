# Résultats constatés — copie corrigée

## Protocole
Source inchangée, SHA256 `321f3a53cf87ceb38b7da3a565706a0b1d4b8b2f5327ea77fc3cf3ea6e5302ec`. C-Haines avec point de rosée approché par Magnus ; 1839 mailles-dates masquées pour RH hors domaine. 77531 mailles-dates brutes, 75692 valides. Dates chronologiques et embargo ; aucun ajustement de transformation sur validation/test.

Budget : 10 époques par modèle, dont 5 standard + 5 pondérées pour le VAE pondéré ; graine 100 ; 10 000 échantillons générés. Les architectures et nombres de mises à jour diffèrent ; ce n’est pas une égalisation du coût de calcul. Les poids ne sont pas publiés ; ils sont reproductibles avec la commande du README.

Le premier WGAN a divergé. Son résultat est conservé dans benchmark_initial.json. Une initialisation résiduelle et des hyperparamètres sont étudiés sur validation : critique 0,001, générateur 0,0001, 5 mises à jour critique par mise à jour générateur, poids GP 0,1 retenu. Le WGAN final ne diverge plus dans cet essai, sans revendication de stabilité générale ou de supériorité.

Le test ayant été consulté lors du diagnostic initial, le bilan final est descriptif. Une validation réellement nouvelle exige d’autres dates/données. Aucune modification n’est choisie pour obtenir un meilleur score sur ce test.

## Mesures
Wasserstein est exprimée en unités de C-Haines, plus faible = meilleure proximité marginale. Quantile observé à 99 % sur test : **9.7857**. Seuil de démonstration : **9.4909**, quantile 95 % de train ; taux observé de dépassement sur test : **1.32 %**.

| Méthode | Wasserstein validation | Wasserstein test déjà vu | Q99 généré | P(dépassement seuil démo) |
|---|---:|---:|---:|---:|
| empirical_train | 0.2776 | 0.5687 | 11.7508 | 4.78 % |
| gaussian_train | 0.6472 | 0.8553 | 10.9723 | 3.17 % |
| vae | 0.3309 | 0.5987 | 11.3038 | 3.97 % |
| vae_weighted | 0.3448 | 0.7923 | 11.7647 | 4.60 % |
| gan | 0.3175 | 0.3446 | 11.4235 | 3.73 % |
| wgan_gp | 0.3447 | 0.6391 | 11.9065 | 5.34 % |
| cgan | 0.8911 | 0.3738 | 11.0000 | 2.95 % |

## Interprétation
Le GAN et le cGAN sont plus proches du test que la référence empirique pour la distance marginale de cet essai. Cela n’établit pas une performance générale : la validation donne un classement différent et l’échantillon ne couvre qu’un mois. Le VAE pondéré n’améliore pas systématiquement les critères. Les quantiles générés sont globalement trop élevés face aux derniers jours du mois ; les résultats ne valident donc pas les probabilités de queue.

Le WGAN corrigé a une distance de test 0.6391, contre 40.7496 dans l’essai initial divergent. Cette réduction diagnostique confirme la correction de stabilité dans cet essai ; elle ne suffit pas à qualifier le modèle de production.

Un paiement paramétrique de montant L s’écrit E[paiement] = L × P(indice > seuil), par observation de maille/temps. L’indemnité 10 000 est un exemple, pas une garantie calibrée. Aucune prime annuelle n’est calculée.

## Tests et limites
Les tests exécutés couvrent les données, les dates, la standardisation train uniquement, les architectures, le bruit d’observation VAE, le gradient penalty analytique, les derniers batches de taille 1, la reproductibilité CPU, le domaine des conditions et le calcul de paiement. Les versions effectives sont dans benchmark.json et requirements.txt.

Les mailles sont dépendantes ; les p-values KS iid sont diagnostiques. Les C-Haines générés ne sont pas des champs météorologiques cohérents ; aucune saisonnalité annuelle, fréquence de feu ou perte assurée n’est validée. Les modèles ne sont pas déclarés prêts pour une tarification d’assurance.
