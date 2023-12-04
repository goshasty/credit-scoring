The solving problem is credit scoring.

Original data is not public available.
So some transformation have been applied to original data

Target = 0: good clients

Target = 1: bad (fraud) clients

run.sh: 
- create empty venv, 
- install poetry, 
- run pre-commit,
- train and infer
- run_server


exmaple of running run_server:

python3 commands.py run_server '{"0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "10": 11, "11": 12, "12": 13, "13": 14, "14": 15}'



Warnings problem on python3.8:
Every time 'import mlflow' 2 warnirngs appears:


> /home/gdemin/mlops/credit_scoring/.venv/lib/python3.8/site-packages/pydantic/_internal/_fields.py:149: UserWarning: Field "model_server_url" has conflict with protected namespace "model_".

> You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
  warnings.warn(
/home/gdemin/mlops/credit_scoring/.venv/lib/python3.8/site-packages/pydantic/_internal/_config.py:321: UserWarning: Valid config keys have changed in V2:
'schema_extra' has been renamed to 'json_schema_extra'
  warnings.warn(message, UserWarning)



Sorry, I don't have the strength to fix it
