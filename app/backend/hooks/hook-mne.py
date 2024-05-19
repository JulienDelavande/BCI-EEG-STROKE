from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('mne')
datas = collect_data_files('mne', subdir=None, include_py_files=True)

print(f'hiddenimports: {hiddenimports}')
#print(f'datas: {datas}')