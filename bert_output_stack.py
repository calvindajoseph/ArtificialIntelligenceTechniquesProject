import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

df_train = pd.read_csv('data/bert_results/train_model_outputs.csv')
df_test = pd.read_csv('data/bert_results/test_model_outputs.csv')
df_val = pd.read_csv('data/bert_results/val_model_outputs.csv')

X_train = df_train.drop(columns=['7', '8', '9', 'target']).to_numpy()
X_test = df_test.drop(columns=['7', '8', '9', 'target']).to_numpy()
y_train = df_train['target'].to_numpy()
y_test = df_test['target'].to_numpy()

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(f'accuracy : {accuracy_score(y_test, y_pred)}')
print(f'accuracy : {f1_score(y_test, y_pred, average="weighted")}')

joblib.dump(model, 'models/classical_models/stack_output.joblib')