# Note LaTeX et PDF

- `note_modeles_generatifs.tex` : source principale ; `section_multiannuelle.tex` est incluse et **générée** par `python -m scripts.make_multiyear_section` depuis `reports/multiyear_benchmark.json` (gabarit : `section_multiannuelle.tex.tmpl`). Ne pas l'éditer à la main.
- `note_modeles_generatifs.pdf` : version compilée, dix pages A4, sept figures.
- `figures/*.pdf` : figures vectorielles générées par `python -m scripts.make_figures`. Celles du jeu multi-annuel exigent `data/era5_firms_jjas_2001_2024.nc` (voir le README principal).

```sh
pdflatex -interaction=nonstopmode -halt-on-error note_modeles_generatifs.tex
pdflatex -interaction=nonstopmode -halt-on-error note_modeles_generatifs.tex
```

Le PDF publié a été compilé avec [Tectonic](https://tectonic-typesetting.github.io) (XeTeX) : `tectonic -X compile note_modeles_generatifs.tex`. Le préambule choisit les polices selon le moteur (iftex, fontspec).

Packages TeX : Latin Modern, babel avec locale française importable, amsmath, graphicx, caption, geometry, booktabs, tabularx, microtype, xcolor, tcolorbox, enumitem, fancyhdr, lastpage, titlesec et hyperref. Aucune publication LinkedIn n'est effectuée automatiquement.
