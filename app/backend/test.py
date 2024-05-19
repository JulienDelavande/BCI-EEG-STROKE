import os

def find_library_paths(lib_path, target_path):
    datas = []
    for root, dirs, files in os.walk(lib_path):
        if files:
            relative_path = os.path.relpath(root, lib_path)
            data_entry = (os.path.join(root, '*'), os.path.join(target_path, relative_path))
            datas.append(data_entry)
    return datas

# Exemple d'utilisation pour la bibliothèque mne
lib_path = 'venv_backend/Lib/site-packages/mne'
datas = find_library_paths(lib_path, 'mne')

# Affichez ou enregistrez les chemins pour utilisation dans le fichier spec
for data in datas:
    print(f"('{data[0]}', '{data[1]}'),")
