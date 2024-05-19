#!/bin/bash
#SBATCH --job-name=train                    # Nom du job
#SBATCH --partition=p16                     # Partition (queue) à utiliser
#SBATCH --nodes=1                           # Nombre de nœuds
#SBATCH --ntasks=1                          # Nombre total de tâches MPI à lancer
#SBATCH --cpus-per-task=1                   # Nombre de cœurs par tâche (24 cœurs par nœud sur p16)
#SBATCH --time=24:00:00                     # Temps total d'exécution (hh:mm:ss)
#SBATCH --mem=64G                           # Mémoire à allouer par nœud
#SBATCH --output=logs/train.%j.out          # Fichier de sortie standard
#SBATCH --error=logs/train.%j.err           # Fichier de sortie d'erreur
#SBATCH --mail-user=julien.delavande@student.isae-supaero.fr
#SBATCH --mail-type=END

module load python/3.8                      # Chargez la version appropriée de Python, si nécessaire
cd $SLURM_SUBMIT_DIR                        # Accédez au répertoire de soumission
cd ./..                                     # Accédez au répertoire racine du projet
python -m venv venv_training                # Créez un environnement virtuel Python, si nécessaire
source venv_training/bin/activate           # Activez votre environnement virtuel Python, si vous en utilisez un
pip install --upgrade pip                   # Mettez à jour pip, si nécessaire
pip install -r requirements_training.txt    # Installez les dépendances du projet
cd $SLURM_SUBMIT_DIR                        # Accédez au répertoire de soumission

# Exécuter le script Python en passant les variables d'environnement
./../venv_training/bin/python script_train.py
