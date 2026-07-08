# SAE admission-risk simulator

Structure du projet après découpage du script monolithique original en
modules dédiés.

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application attend les fichiers de données suivants dans un dossier
`data/` placé à côté de `app.py` (inchangé par rapport à l'original) :

- `data/capacities_2025_wta_with_2024_calibration.csv`
- `data/programmes_chili_criteres_recommandation.csv`
- `data/rbd_region_map.csv`
- `data/program_filters.csv`
- `data/commune_coordinates.csv` *(optionnel, améliore le calcul de proximité)*

## Structure

```
app.py                          # Point d'entrée Streamlit : orchestration uniquement
requirements.txt
sae_app/
    __init__.py                 # Vue d'ensemble du découpage (docstring)
    constants.py                # Colonnes, seuils, chemins de fichiers, options de filtres
    i18n.py                     # Dictionnaire de traduction ES/EN + t()
    text_utils.py                # Nettoyage de texte/nombres (aucune dépendance interne)
    data_loading.py             # Lecture et validation des CSV
    program_options.py          # ProgramRecord + construction/filtrage du menu déroulant
    mtb_engine.py                # Hash SHA-256, priorités, modèle hypergéométrique (pur, sans Streamlit)
    wish_list.py                 # Parsing de la liste de préférences, classes d'équivalence
    geo.py                        # Coordonnées, distance, géocodage d'adresse
    recommendations.py           # Moteur de recommandation "programmes similaires"
    session_state.py            # Invalidation de simulation, nettoyage des clés de widgets
    ui_common.py                  # Formatage d'affichage partagé (traduction de tableaux)
    ui_simulation.py              # Rendu des résultats de simulation (résumé, sensibilité)
    ui_wish_builder.py           # Rendu du widget de construction de la liste
    ui_recommendations.py        # Rendu de la section "programmes similaires recommandés"
```

## Principe de découpage

- **Moteur de calcul pur** (`mtb_engine.py`, une bonne partie de `wish_list.py`,
  `recommendations.py`) : aucune dépendance à `streamlit` pour l'affichage —
  seul `i18n.t()` est utilisé pour les messages d'erreur traduits. Ces modules
  sont testables unitairement sans lancer l'app.
- **Chargement de données** (`data_loading.py`) : sait interpréter les CSV
  sources (encodage, noms de colonnes variables, traduction des valeurs
  FR→EN), ne sait rien de l'UI.
- **UI** (`ui_*.py`) : ne contient que des appels `st.*` et fait appel aux
  modules de calcul ; ne contient aucune formule.
- **`app.py`** : uniquement l'enchaînement des sections de la page, dans le
  même ordre que l'original. Aucune logique métier n'y a été ajoutée.

## Différences volontaires avec le script original

Ce sont des simplifications mineures, sans changement de comportement :

- Dans `ui_recommendations.py`, l'ajout d'un programme recommandé à la liste
  utilise désormais `make_builder_wish_row(...)` (déjà utilisé ailleurs)
  plutôt que de reconstruire le même dictionnaire à la main.
- Quelques imports de constantes non utilisées après le découpage
  (`RECOMMENDATION_CRITERIA_BY_COL`, `PROGRAM_GEO_SOURCE`, etc.) avaient déjà
  été supprimés dans la version précédente et le restent ici.

Le contenu du dictionnaire de traductions (`sae_app/i18n.py`) a été conservé
tel quel, y compris quelques clés qui ne sont plus référencées ailleurs dans
le code (ex. `"Safety option"`, `"Portfolio score"`) — un nettoyage de ce
fichier n'était pas dans le périmètre de cette demande.
