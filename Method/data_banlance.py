
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


data_path = './data/train_data/elmo_vulnerabilities_vector.txt'
column_names = ['label'] + [f'feature_{i}' for i in range(150*16)]
data = pd.read_csv(data_path, sep=' ', header=None, names=column_names)

X = data.drop('label', axis=1)
y = data['label']


undersample = RandomUnderSampler(sampling_strategy={0: 2000, 1: 2000})
X_undersampled, y_undersampled = undersample.fit_resample(X, y)



oversample = SMOTE(sampling_strategy={2: 2000, 3: 2000, 4: 2000})
X_oversampled, y_oversampled = oversample.fit_resample(X_undersampled, y_undersampled)



undersampled_data = pd.DataFrame(X_undersampled, columns=column_names[1:])
undersampled_data = pd.concat([undersampled_data, pd.Series(y_undersampled, name='label')], axis=1)

oversampled_data = pd.DataFrame(X_oversampled, columns=column_names[1:])
oversampled_data = pd.concat([oversampled_data, pd.Series(y_oversampled, name='label')], axis=1)

balanced_data = pd.concat([undersampled_data, oversampled_data])
print(balanced_data['label'].value_counts())


balanced_data.to_csv('./data/train_data/elmo_balanced_data.csv', index=False)