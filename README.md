<p align="center">
  <img src="docs/img/pipeline.svg" alt="Étapes du projet : données, audit, variables, découpage, modèles, explications" width="760">
</p>

<h1 align="center">Des données à l'IA explicable : systèmes hydrauliques</h1>

<p align="center">
  Reconnaître l'usure d'une machine hydraulique à partir de ses capteurs,<br>
  et vérifier si l'explication donnée par le modèle mérite confiance.
</p>

<p align="center">
  Projet AIML · Master AI4CI · CNAM
</p>

---

## L'idée

Un banc d'essai hydraulique répète le même cycle de travail, encore et encore. Pendant chaque cycle, ses capteurs enregistrent la pression, le débit, la température, les vibrations et la puissance du moteur.

Certains composants s'usent ou se dégradent : la valve, la pompe, l'accumulateur, le refroidisseur. Le projet cherche à répondre à deux questions :

1. **Peut-on reconnaître l'état d'un composant** à partir des seuls signaux des capteurs ?
2. **Peut-on expliquer cette décision**, et surtout, peut-on faire confiance à l'explication ?

## Comment ça marche

<p align="center">
  <img src="docs/img/how_it_works.svg" alt="Les capteurs mesurent un cycle, le modèle donne un diagnostic, puis l'explication montre quels capteurs ont compté" width="760">
</p>

1. **Les capteurs mesurent.** Chaque cycle est résumé en quelques valeurs simples par capteur : moyenne, variation, minimum, maximum et tendance.
2. **Le modèle décide.** Il a appris sur des cycles passés et donne l'état probable de chaque composant.
3. **On explique.** Plusieurs méthodes indiquent quels capteurs ont pesé dans la décision. On teste ensuite ces explications : si on retire les capteurs désignés, le modèle se trompe-t-il vraiment ? L'explication reste-t-elle la même d'un essai à l'autre ?

## Données

[Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems), UCI Machine Learning Repository, DOI 10.24432/C5CW21, licence CC BY 4.0.
