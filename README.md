# fyp-meta
MAML Implementation on the Movielens dataset.

https://grouplens.org/datasets/movielens/

For running locally on your machine without change to the code, download and move the ml-32m folder into /data/ directly and run 
1. `python data_preparation.py`
2. `python maml.py`
3. `python evaluation.py`

in that order.

Above steps are for creating MAML model and executing evaluation on the prepared data.

For the MAMO model, execute the following steps below:

1. `python data_preparation.py`
2. `python mamo.py`
3. `python evaluate_mamo.py`

Refer to the report for more details.
