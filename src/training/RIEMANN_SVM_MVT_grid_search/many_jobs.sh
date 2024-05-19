#!/bin/bash

# Ce script soumet plusieurs jobs SLURM pour exécuter le script Python sur plusieurs nœuds pour la recherche d'hyperparamètres.

# Nombre total de noeuds à utiliser
NUM_NODES=6

# Détecter le dernier dossier VX dans le répertoire hyperparams
LAST_VERSION_DIR=$(ls hyperparams/ | grep '^V' | sort -V | tail -n 1 | sed 's/V//')

# Vérifier si la variable VERSION est définie, sinon utiliser la dernière version détectée
VERSION=${1:-$LAST_VERSION_DIR}

for i in $(seq 1 $NUM_NODES); do

    JOB_NAME="RM_V${VERSION}N${i}"
    OUTPUT_LOG="logs/V${VERSION}/slurm/${JOB_NAME}_out_%j.log"
    ERROR_LOG="logs/V${VERSION}/slurm/${JOB_NAME}_err_%j.log"
    # Créer le répertoire logs/V{VERSION}/slurm/ s'il n'existe pas
    mkdir -p logs/V${VERSION}/slurm/

    # Soumettre le job SLURM en passant le numéro du noeud et le nombre total de noeuds comme variables d'environnement
        sbatch --export=ALL,NODE_INDEX=$i,TOTAL_NODES=$NUM_NODES,VERSION=$VERSION \
           --job-name=$JOB_NAME \
           --output=$OUTPUT_LOG \
           --error=$ERROR_LOG \
           job_template.slurm
done
