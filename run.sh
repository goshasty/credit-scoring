python3 -m venv ./.venv
source ./.venv/bin/activate
/home/gdemin/.local/bin/poetry install
'pre-commit' install
'pre-commit' run -a
python3 commands.py train
python3 commands.py infer
rm -r .venv/
echo Done