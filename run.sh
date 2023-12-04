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
echo 'Infer local done'
python3 commands.py run_server '{"0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "10": 11, "11": 12, "12": 13, "13": 14, "14": 15}'
echo 'Infer sercer done'
rm -r .venv/
echo Done