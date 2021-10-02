import pandas as pd 
import sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1
    
    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df.drop(columns='target'), y=df.target.values)):
        print(len(train_idx), len(val_idx))


