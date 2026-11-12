# Note LaTeX et PDF

`note_modeles_generatifs.tex` est autonome, sans image externe. Le PDF compilé comporte six pages A4 : données et séparation, C-Haines, VAE, GAN/cGAN, WGAN-GP, résultats et limites.

```sh
pdflatex -interaction=nonstopmode -halt-on-error note_modeles_generatifs.tex
pdflatex -interaction=nonstopmode -halt-on-error note_modeles_generatifs.tex
```

Packages TeX usuels : Latin Modern, babel avec locale française importable, amsmath, geometry, booktabs, tabularx, microtype, xcolor, tcolorbox, enumitem, fancyhdr, lastpage, titlesec et hyperref. Aucune publication LinkedIn n'est effectuée automatiquement.
