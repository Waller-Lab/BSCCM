rm dist/*
rm build/*
python setup.py sdist bdist_wheel && python -m twine upload dist/*
