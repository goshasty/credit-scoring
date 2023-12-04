python3 -m venv ./.venv
source ./.venv/bin/activate
/home/gdemin/.local/bin/poetry install
'pre-commit' install
'pre-commit' run -a
python3 commands.py hyperopt
echo 'Hyperopt done'
python3 commands.py train
echo 'Train done'
python3 commands.py infer
echo 'Infer done'
rm -r .venv/
echo Done