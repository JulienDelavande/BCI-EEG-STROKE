# Documentation du Projet BCI-EEG-STROKE

## Structure des Données

### Format de stockage
- **Type de Données**: Tableau NumPy (Chaque élément du tableau représente une session pour un patient donné)

### Structure de chaque élément du tableau
1. **Identifiant Session**: Identifiant unique pour chaque session de test (e.g., '004').
2. **Côté de la Lésion**: Côté du cerveau qui est lésé (G pour Gauche, D pour Droit).
3. **Numéro de l'Essai**: Identifiant de l'essai en cours (e.g., 'Trial1').
4. **Fréquence d'Échantillonnage de l'Accélération**: Fréquence à laquelle les données d'accélération sont échantillonnées (e.g., 125 Hz).

#### Données de la Sous-Session Gauche
5. **Données Sous-Session Gauche**: Contient plusieurs sous-éléments:
    - **Côté**: Gauche (G)
    - **Index de Mouvement**: `IdvrMVTrddd`, Liste des index où un mouvement a été détecté.
    - **Données EEG**: Liste de paires `[nom, données]` où `nom` est le nom de l'électrode et `données` sont les mesures EEG correspondantes.
    - **Données Cinématiques**: `[AC, VAC, AC3d, VAC3d]` où AC et VAC sont l'accélération et la vitesse en 2D, et AC3d et VAC3d sont l'accélération et la vitesse en 3D.

#### Données de la Sous-Session Droite
6. **Données Sous-Session Droite**: Structure similaire à celle de la Sous-Session Gauche mais pour le côté droit.

7. **Index du Patient**: Identifiant unique pour le patient.

## Visualisation
[004, G, Trial1, 125, [G, IdvrMVTrddd, [EEG[nom, données]], [AC,VAC,AC3d,VAC3d]],[D, IdvrMVTrddd, [EEG[nom, données]]], [AC,VAC,AC3d,VAC3d]], patient_index]
- 004 [0]
- G [1]
- Trial1 [2]
- 125 [3]
- Donnée sous-session G [4] (liste de sous-éléments) 
    - G [4][0] (string) 
    - IdvrMVTrddd [4][1] (liste des index où un mouvement a été détecté) 
    - EEG [4][2] (liste de paires [nom, données] où nom est le nom de l'électrode et données sont les mesures EEG correspondantes)
        - paire [nom, données] [4][2][i]  (liste de 2 éléments)
            - nom [4][2][i][0] (string) 
            - données [4][2][i][1] (liste de données EEG pour une électrode donnée)
    - Accelerations et vitesse [4][3] (liste de 4 éléments)
        - AC [4][3][0] (liste de données d'accélération en 2D)
        - VAC [4][3][1] (liste de données de vitesse en 2D)
        - AC3d [4][3][2] (liste de données d'accélération en 3D)
        - VAC3d [4][3][3] (liste de données de vitesse en 3D)
- Donnée sous-session D [5]
    - D [5][0]
    - IdvrMVTrddd [5][1]
    - EEG [nom, données] [5][2]
        -paires [5][2][i]
            - nom [5][2][i][0]
            - données [5][2][i][1]
    - Accelerations et vitesse [5][3]
        - AC [5][3][0]
        - VAC [5][3][1]
        - AC3d [5][3][2]
        - VAC3d [5][3][3]
- patient_index [6]
