# Qwen Trading — Spécification expérimentale V3

**Statut :** proposition de protocole, à valider avant toute génération de dataset ou entraînement  
**Périmètre :** BTC/USDC, bougies 1H, Qwen3.5-9B ; recherche hors production  
**Date :** 9 octobre 2026  
**Principe :** ne pas confondre précision des labels, probabilité de succès et rentabilité après frais.

## 1. Résumé et décision

**Décision proposée :** ne pas relancer un SFT à trois classes `LONG_BIAS / SHORT_BIAS / NO_TRADE` identique à V2. Étudier une **V3 conditionnelle à un candidat**, où un générateur causal propose un LONG ou un SHORT, et un évaluateur détermine s'il faut l'accepter. L'évaluateur ne reçoit aucune donnée future.

Cette architecture est une **hypothèse**, pas une solution validée. Avant d'entraîner Qwen, vérifier qu'une méthode simple trouve un signal de filtrage hors échantillon, puis fixer les critères de succès et un ensemble de test prospectif inédit.

### Constats établis par les audits V2

- Le label SFT V2 associe une direction heuristique à un TP futur ; il ne représente pas directement une espérance nette exécutable.
- Le Train SFT V2 équilibré compte **659 exemples par classe** (1 977 au total), alors que le Test SFT naturel est composé de **75,14 % `NO_TRADE`**.
- Sur le Test SFT, Qwen prend **824/1 243 signaux**, contre **928/1 243** pour l'heuristique, et ne démontre pas de filtrage SHORT favorable.
- Sur Validation, les baselines probabilistes régression logistique et HistGradientBoosting n'ont pas montré d'amélioration convaincante contre des probabilités constantes fondées sur le Train.
- Backtest événementiel Validation, **7 bps/côté** (5 de frais + 2 de slippage) : **Qwen −17,687 % sur 191 positions**, heuristique **−9,873 % sur 189 positions** (capital initial 10 000, allocation de 100 % du capital simulé).
- Sur la même Validation, point mort estimé en coût total par côté : **Qwen 1,9038 bps**, heuristique **4,2485 bps**. Ces valeurs sont propres à cette simulation.
- Dans l'audit de concentration, **7 des 8 semaines Qwen** et **6 des 8 semaines heuristique** sont déficitaires ; les LONG pénalisent les deux systèmes.
- Les fenêtres d'événements 12H des observations horaires se chevauchent fortement ; elles ne doivent pas être traitées comme des observations indépendantes.

**Limites des constats :** l'historique couvre environ un an et un seul instrument ; les coûts et les fills sont simulés ; les rendements hebdomadaires ne sont pas des preuves de causalité ; le Test 2026 a déjà été consulté et n'est plus un test vierge pour les choix V3.

## 2. Question scientifique et hypothèses

**Question primaire :** à candidat et moteur d'exécution fixés, un évaluateur entraîné sur des informations causales disponibles à la décision améliore-t-il **la performance nette hors échantillon**, à risque et exposition comparables, face à l'acceptation systématique des candidats ?

Hypothèses falsifiables :

- **H0 (référence)** : aucun filtre ne surpasse de manière reproductible l'heuristique qui accepte tous les candidats, une fois prises en compte les différences de nombre de trades et les coûts.
- **H1 (filtrage)** : une probabilité conditionnelle de succès/échec ou une estimation du rendement net permet de classer les candidats de façon utile.
- **H2 (valeur propre de Qwen)** : Qwen surpasse, à entrée de données et protocole comparables, des filtres simples (prior, régression logistique, boosting), et pas seulement la cible SFT équilibrée.
- **H3 (stabilité)** : un éventuel gain reste présent sur plusieurs fenêtres chronologiques, sans dépendre d'une semaine ou d'un régime isolé.

Ne pas retenir une hypothèse à partir d'un simple meilleur résultat moyen sur une période sélectionnée a posteriori.

## 3. Univers des données et gestion temporelle

### 3.1 Historique existant

- Source : `data/processed/btc_usdc_1h_labeled.parquet` ; 8 760 bougies, OHLCV, indicateurs techniques, labels historiques.
- `timestamp` : ouverture de la bougie ; `available_at = timestamp + 1H` : disponibilité des features de cette bougie.
- `data/splits/{train,validation,test}.parquet` : splits chronologiques naturels, séparés par une purge de 24H pour les labels historiques jusqu'à 24H.
- Résultats V2 : `data/evaluation/qwen3.5-9b-trading-v2-validation/results.parquet` et `...-v2-fast/results_fast.parquet`.
- Backtest de référence : `scripts/backtest_event_driven_v1.py`, résultats `data/evaluation/backtest-event-driven-v1/validation_only/`.

### 3.2 Fuites à proscrire

Pour chaque décision à `available_at(t)`, autoriser exclusivement les features connues à cette date : OHLCV de la bougie close, indicateurs glissants **causaux**, HTF 4H/1D **closes et disponibles**, score heuristique causal, et paramètres publics du candidat (direction, ATR, TP, SL, coûts supposés). Exclure strictement toutes les colonnes `future_*`, `*_best_*`, `*_worst_*`, `*_outcome_*`, `up_move_*`, `down_move_*` et tout agrégat dérivé du futur.

Les labels calculés à partir du futur appartiennent à la **construction des cibles et à l'évaluation exclusivement**. Les dates et l'ordre des opérations de normalisation, calibrage et apprentissage doivent être contrôlés. Chaque transformation apprise (imputation, scaling, seuils, calibration) est ajustée sur la portion Train autorisée uniquement.

### 3.3 Séparation et test vierge

Les données existantes sont utiles pour l'exploration et les expériences de conception ; **ne pas présenter le Test historique déjà analysé comme confirmation finale V3**. Pour l'évaluation finale, collecter une période postérieure au 10 septembre 2026, sans modification du protocole après examen de ses résultats, ou recourir d'abord à un walk-forward chronologique strictement prédéfini pour la recherche. La période finale doit être figée avant la première évaluation.

Le délai minimal d'embargo doit couvrir l'horizon maximal de label et les événements qui franchissent les frontières. Vérifier aussi les cas d'entrée à l'ouverture suivante, d'HTF non closes et les fenêtres de positions se chevauchant.

## 4. Architecture candidate V3

```text
OHLCV + features causales disponibles à T
                |
       Générateur de candidats
       (règles fixes, versionnées)
                |
     Candidat LONG / SHORT + risque
                |
       Évaluateur probabiliste
       (prior -> logistique -> boosting -> Qwen)
                |
       ACCEPT ou REJECT, selon
       politique fixée hors Test
                |
      Moteur événementiel commun
        entrée T+1, TP/SL/timeout
                |
      PnL net, risque et robustesse
```

### 4.1 Générateur de candidats (première itération)

Utiliser le score heuristique existant : LONG si `bias_score >= 2`, SHORT si `bias_score <= -2`, aucun candidat sinon. Ce choix **figé** permet d'isoler la valeur du filtre. Une variante de générateur ne sera introduite que dans une expérience séparée, déclarée à l'avance.

**Unité de prédiction :** un candidat `(signal_timestamp, side, features, candidate_version)`. Pas un label global LONG/SHORT/NO_TRADE par bougie ; l'absence de candidat n'est pas automatiquement un exemple négatif de filtrage.

### 4.2 Le rôle de Qwen

Qwen reçoit une représentation strictement causale et structurée d'**un candidat** et renvoie une décision normalisée : `ACCEPT / REJECT`, éventuellement accompagnée d'un score de confiance utilisable seulement s'il a été calibré. Ne pas demander au modèle de raconter l'issue future comme si elle était connue. La justification textuelle éventuelle doit être traitée comme explication non vérifiée, jamais comme preuve de qualité.

Le choix du format de sortie (binaire, ordinal, ou probabilités de TP/SL/TIME) reste à sélectionner **sur Train/Validation seulement**, en cohérence avec les baselines. En phase initiale, conserver les probabilités multiclasse et calculer la politique d'acceptation séparément ; une classe `ACCEPT` construite à partir du PnL réalisé est une **étiquette rétrospective bruitée**, pas une vérité prédictive.

## 5. Construction des exemples et des événements

### 5.1 Prix et ordre des informations

- Signal connu à la clôture de la bougie `t`, soit `available_at(t)`.
- Entrée théorique à l'`open` de la bougie `t+1`, avec coût d'exécution adverse configurable.
- Niveaux TP/SL calculés à partir du prix d'entrée brut et de l'ATR **connue à t**. Paramètres de référence : TP `+1,5 ATR`, SL `−1 ATR`, horizon 12 bougies, sans optimisation V3 sur Test.
- Sortie au premier toucher ; quand TP et SL sont tous deux accessibles dans une même bougie OHLC, **SL prioritaire** dans le scénario conservateur. Mentionner séparément le taux d'ambiguïtés.
- Si aucun seuil n'est touché, sortie **au close de la dernière bougie de l'horizon** ; le rendement n'est donc **pas zéro**.
- Gaps à l'ouverture : traiter une ouverture au-delà d'un seuil comme un fill à l'ouverture, et non au niveau devenu inaccessible.

### 5.2 Cibles à sauvegarder

Pour **chaque** candidat, conserver les valeurs suivantes, générées par un simulateur d'événements réutilisable, sans décision Qwen :

| Champ | Définition |
|---|---|
| `event_outcome` | `TP`, `SL`, `TIME`, éventuellement `DATA_INVALID` |
| `duration_bars` | Nombre de bougies entre l'entrée et la sortie |
| `entry_raw`, `exit_raw` | Prix bruts appliqués par le simulateur |
| `entry_fill`, `exit_fill` | Prix après slippage |
| `gross_return`, `net_return` | Rendement notionnel du candidat |
| `net_r` | Rendement net normalisé par le risque initial |
| `known_at` | Horodatage de disponibilité du signal |
| `label_available_at` | Date à laquelle l'issue était connaissable |
| `event_version`, `fee_assumption` | Identifiants reproductibles de la simulation |

**Point méthodologique :** le moteur peut simuler tous les candidats pour produire les cibles, mais les backtests comparatifs doivent **rejouer chronologiquement** leurs décisions à une seule position ouverte. Il serait incorrect de sélectionner d'avance uniquement les candidats « exécutables » sous la politique d'une stratégie : cette sélection dépend des décisions précédentes et crée des univers de données différents selon les modèles.

### 5.3 Étiquettes et sélection

- **Cible primaire de recherche :** probabilités de `TP/SL/TIME`, plus métriques de calibration et classement des candidats.
- **Cible économique secondaire :** `net_return`/`net_r` incluant la sortie temporelle réelle et les coûts, au lieu de traiter automatiquement `TIME = 0R`.
- **Politique d'acceptation :** prédéfinie ou fixée uniquement via une validation chronologique interne : par exemple, score attendu après coûts positif **et** budget maximal de positions, sans sélectionner des seuils sur le Test historique.
- **Classes ambiguës :** ne pas masquer leur traitement. La politique conservatrice SL-first est la principale ; envisager un audit de sensibilité « ambigu exclu » sans en tirer la meilleure règle a posteriori.

## 6. Cohorte d'entraînement et baselines

1. Construire le jeu naturel de **candidats** à partir du Train chronologique, sans équilibrage artificiel comme point de départ.
2. Dédupliquer `(timestamp, side, candidate_version)` ; confirmer absence de features futures et de joins manquantes.
3. Stocker la prévalence naturelle des issues par direction et par période.
4. Baseline 0 : **tout accepter** (heuristique) dans le moteur événementiel ; baseline complémentaire : **tout refuser** (0 trade), indispensable pour interpréter les stratégies déficitaires.
5. Baseline 1 : probabilité constante issue du Train.
6. Baseline 2 : régression logistique calibrée seulement sur une tranche chronologique interne autorisée.
7. Baseline 3 : HistGradientBoosting, paramètres annoncés, contrôle strict du surapprentissage.
8. Candidat V3 : Qwen entraîné sur exactement le même univers causal et comparé sur les mêmes fenêtres de décision.

Ne pas assimiler directement les échecs des baselines précédentes à un verdict pour ces nouvelles cibles : le jeu conditionnel à des candidats et le traitement économique de TIME diffèrent. Mais **aucune amélioration n'est présumée**.

## 7. Protocole de validation

### 7.1 Étapes, avant tout SFT

**Phase A — validation des événements :** tester timing et prix sur exemples synthétiques ; contrôler OHLC, TP/SL simultanés, gaps, timeout, fin de split, frais, slippage et cohérence des timestamps. Comparer les nouveaux labels aux anciens uniquement là où les hypothèses sont identiques.

**Phase B — baseline « candidats » :** mesurer prior, régression logistique et boosting en walk-forward sur Train, et éventuellement Validation pour la confirmation préliminaire, avec métriques probabilistes et backtests nets. Ne pas multiplier des choix hyperparamétriques guidés par Validation sans la considérer ensuite comme jeu de développement.

**Phase C — SFT Qwen pilote :** entraînement minimal, une seule configuration déclarée ; contrôle de la représentation, de la fuite, de la distribution des classes et de la stabilité des sorties.

**Phase D — comparaison figée :** prédictions sauvegardées, mêmes coûts, mêmes règles de sortie et même moteur ; comparaisons contre accepter tout, refuser tout et meilleurs modèles simples sélectionnés **avant** le test final inédit.

### 7.2 Métriques primaires et secondaires

**Primaires (préenregistrées) :** performance nette hors échantillon d'une politique candidate fixée, drawdown sur equity marquée au marché si disponible, et valeur ajoutée à exposition/risque comparables. Définir avant évaluation un critère primaire unique (par exemple différence de rendement logarithmique net à budget d'exposition fixé).

**Secondaires :** log loss/Brier/calibration probabiliste par direction, nombre de positions, durée, turnover, frais, win rate, profit factor, perte maximale, performance par période et par direction, comparaison à 0 trade. Présenter ensemble l'ampleur des effets, leur incertitude et la fréquence des périodes favorables, et non uniquement des p-values.

La comparaison doit considérer les stratégies comme **appariées dans le temps**, même si leurs trades exacts diffèrent. Pour l'incertitude, privilégier un **bootstrap par blocs temporels** ou un walk-forward à fenêtres préspécifiées ; ne pas traiter les trades chevauchants comme indépendants. Toute estimation statistique doit préciser la taille des blocs et les limites dues au petit nombre de périodes.

### 7.3 Décision GO / NO-GO

**Avant Qwen :** NO-GO si les cibles restent temporellement incohérentes, si les baselines ne sont pas reproductibles, ou si aucune mesure économique pertinente n'est disponible. Un échec des baselines simples n'interdit pas de rechercher, mais augmente l'exigence de justification d'un SFT coûteux.

**Avant un test final inédit :** GO seulement avec pipeline versionné, contrôles de fuite passés, métriques et politiques figées, et comparateurs préenregistrés.

**Avant tout déploiement :** exiger plusieurs périodes réellement nouvelles, des coûts réalistes pour le lieu d'exécution, un risque acceptable, et une expérimentation paper trading. Aucune donnée présentée ici ne valide un déploiement réel.

## 8. Traçabilité et livrables

Livrables à produire dans cet ordre :

1. `scripts/build_candidate_events_v3.py` : événements et labels exécutables ; **ne pas modifier** les données V2.
2. `scripts/test_candidate_events_v3.py` : tests synthétiques et contrôles de causalité.
3. `data/evaluation/v3-candidate-audit/` : distributions par split, taux TIME/ambigu, écarts de coûts, frontières d'événements, chevauchement.
4. `scripts/evaluate_candidate_baselines_v3.py` : modèles tabulaires conditionnels et comparaison des probabilités.
5. `scripts/backtest_candidate_filters_v3.py` : moteur commun pour les filtres, résultats appariés, contrôle « all accept » et « no trade ».
6. `data/sft_v3/` et script SFT **uniquement après décision documentée**.
7. Rapport comparatif et configuration figée ; évaluation finale sur une période prospective jamais consultée.

Pour chaque expérience : stocker `git_commit`, hash des fichiers de données, versions Python/librairies, graine, paramètres, dates exactes des splits, règles de sélection, scripts et rapport. Réserver les répertoires V2 en lecture seule ; produire de nouveaux noms pour V3.

## 9. Questions ouvertes — décisions à verrouiller avant implémentation

- **Marché exécuté :** spot BTC/USDC (où le SHORT nécessite un mécanisme spécifique) ou perpétuel/marge ? Le financement, l'emprunt, la marge et la liquidation changent l'économie.
- **Frais réels :** grille du marché/exchange envisagé, ordres taker/maker, taille notionnelle, spread, impact et slippage mesuré.
- **Risques :** allocation, nombre de positions simultanées, perte maximale, règle d'arrêt, budget d'exposition cible ; l'allocation de 100 % actuelle n'est qu'une convention expérimentale.
- **Période inédite :** données postérieures à septembre 2026 et longueur minimale recherchée avant gel des métriques.
- **Politique de scoring :** modèle TP/SL/TIME puis score net, estimation directe du rendement, ou comparaison de plusieurs cibles sur Train uniquement.
- **Règles de fill :** ambiguïtés dans la bougie d'entrée, gaps, frais sur notional, unité d'ATR et événements de données manquantes.

## 10. Prochaine action recommandée

**Commencer par l'implémentation et les tests de `build_candidate_events_v3.py`, pas par l'entraînement de Qwen.**

Elle devra réutiliser les conventions du moteur événementiel existant, calculer les événements sur les candidats naturels de Train/Validation, conserver les sous-séries chronologiques originales pour éviter les erreurs de décalage d'une bougie, puis vérifier les frontières. Le Test historique peut rester archivé comme comparaison exploratoire déjà contaminée, sans rôle dans la sélection V3.

**Aucun hyperparamètre, seuil ou stratégie V3 n'est déclaré validé dans cette spécification.**
