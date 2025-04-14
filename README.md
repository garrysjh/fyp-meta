# fyp-meta
MAML & MAMO Implementation on the Movielens dataset.

https://grouplens.org/datasets/movielens/

For running locally on your machine without change to the code, download and move the ml-32m folder into /data/ directly and run 
1. `python data_preparation.py`
2. `python maml.py`
3. `python evaluation.py`

`python data_preparation.py` is executed to clean and prepare data for experimentation.

`python maml.py` generates a `maml_recommender.pth` which is serialized PyTorch state dictionary which contains weights, parameters of the MAML model (aka generated model)

`python evaluation.py` executes the evaluation and prints the evaluation results to the console.

To execute the MAMO experiment, download and move the ml-32m folder into /data/ directly and run
1. `python data_preparation.py` if not done so
2. `python mamo.py`
3. `python evaluate_mamo.py`

`python mamo.py` generates a `mamo_recommender.pth` which is serialized PyTorch state dictionary which contains weights, parameters of the MAMO model (aka generated model)

`python evaluate_mamo.py` executes the evaluation and prints the evaluation results to the console.


Evaluation results are given in NDCG@k, Precision@k, Recall@k, MSE, RMSE.


in that order.

Refer to the report for more details.
