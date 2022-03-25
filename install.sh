rm ./dist/*
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
python setup.py bdist_wheel
pip uninstall -y torch-firewood
pip install dist/torch_firewood*
