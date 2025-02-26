from config import CONFIG
import xgboost as xgb
from loaders import loaders
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def baseline():
    x_train, x_valid, y_train, y_valid = loaders(gen_dir = CONFIG.gen_dir, hybrid_dir = CONFIG.hybrid_dir)
    
    # Training the model and prediction
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_valid)
    acc = accuracy_score(y_valid, y_hat)
    print("The accuracy of the model is: "+str(acc))

if __name__ == "__main__":
    baseline()
