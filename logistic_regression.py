from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from numpy import reshape, concatenate
from numpy import chararray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import json
#from loadfiles import loaddata


# Currently not needed:
# def save_loadeddata(p, n):
#     """saves with loaddata() loaded data as a JSON file."""
#     import json
#     with open("p_loaded_data.json", "w") as f:
#         json.dump(p, f)
#     with open("n_loaded_data.json", "w") as f:
#         json.dump(n, f)
#
# def load_loadeddata():
#     """loads with save_loadeddata() saved files
#     and returns its p and n again."""
#     import json
#     with open("p_loaded_data.json", "r") as f:
#         p = json.load(f)
#     with open("n_loaded_data.json", "r") as f:
#         n = json.load(f)
#     return p, n

def log_reg(pos, neg):
    """Builds a Logistic Regression model using the Yao-score
       of the given sample data as the feature and returns it.
    Args:
      pos (Array[String]): An array of paths to proven positive yao-scores.
      neg (Array[String]): An array of paths to proven negative yao-scores.

    Returns:
      cv (ndarray[float64]): The Cross validation score
    """
    X = []
    for i in neg:
        with open(i) as negfile:
            X += json.load(negfile).values()
    negf = len(X)
    for j in pos:
        with open(j) as posfile:
            X += json.load(posfile).values()
    X = reshape(X, (-1, 1))
    posf = len(X) - negf
    y = [0]*negf + [1]*posf
    lr = LogisticRegression()
    cv = cross_val_score(lr, chararray.astype(X, float), y, scoring="f1")
    return cv 


if __name__ == "__main__":
    cv = log_reg(["data/yaoscores/pos.json", "data/yaoscores/pos2.json"],
                 ["data/yaoscores/neg.json"])