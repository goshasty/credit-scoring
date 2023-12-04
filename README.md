The solving problem is credit scoring.

Original data is not public available.
So some transformation have been applied to original data

Target = 0: good clients

Target = 1: bad (fraud) clients

run.sh: 
    create empty venv, 
    install poetry, 
    run pre-commit,
    train and infer
    run_server


exmaple of running run_server:

python3 commands.py run_server '{"0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "10": 11, "11": 12, "12": 13, "13": 14, "14": 15}'

