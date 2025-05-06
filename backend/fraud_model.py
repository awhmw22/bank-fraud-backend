import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# 1. Generate synthetic fraud data
def generate_data(n_samples=25000):
    np.random.seed(42)
    countries = ['Pakistan', 'India', 'UAE', 'UK', 'US', 'Canada',
                 'Afghanistan', 'Nigeria', 'Venezuela', 'Russia',
                 'Iran', 'China', 'Brazil', 'Mexico', 'Germany']
    p = [0.15, 0.15, 0.1, 0.1, 0.1,
         0.05, 0.05, 0.05, 0.05, 0.05,
         0.05, 0.05, 0.03, 0.02, 0.05]
    p = [x / sum(p) for x in p]

    data = {
        'amount': np.round(np.random.exponential(scale=2000, size=n_samples), 2),
        'country': np.random.choice(countries, n_samples, p=p),
        'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples, p=[0.3, 0.3, 0.3, 0.1]),
        'device_type': np.random.choice(['mobile', 'web', 'tablet', 'atm'], n_samples, p=[0.6, 0.3, 0.05, 0.05]),
        'is_foreign': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'account_age_days': np.random.randint(1, 3650, n_samples),
        'txn_frequency': np.random.poisson(3, n_samples),
        'is_new_device': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'PKR', 'AED', 'CNY'], n_samples)
    }

    df = pd.DataFrame(data)

    df['fraud'] = False
    high_risk = ['Afghanistan', 'Nigeria', 'Venezuela', 'Russia', 'Iran']
    df.loc[(df['amount'] > 8000) & (df['country'].isin(high_risk)), 'fraud'] = True
    df.loc[(df['is_new_device']) & (df['is_foreign']) & (df['amount'] > 2000), 'fraud'] = True
    df.loc[(df['time_of_day'] == 'night') & (df['account_age_days'] < 30) & (df['amount'] > 5000), 'fraud'] = True
    df.loc[(df['txn_frequency'] > 8) & (df['amount'] > 3000), 'fraud'] = True

    # Add noise
    df.loc[df[df['fraud']].sample(frac=0.1, random_state=1).index, 'fraud'] = False
    df.loc[df[~df['fraud']].sample(frac=0.05, random_state=1).index, 'fraud'] = True

    return df

# 2. Add derived features
def add_features(df):
    country_risk = {
        'Pakistan': 2, 'India': 2, 'UAE': 1, 'UK': 1, 'US': 1,
        'Canada': 1, 'Afghanistan': 3, 'Nigeria': 3, 'Venezuela': 3,
        'Russia': 3, 'Iran': 3, 'China': 2, 'Brazil': 2, 'Mexico': 2, 'Germany': 1
    }
    df['country_risk'] = df['country'].map(country_risk)
    df['amount_to_age_ratio'] = df['amount'] / (df['account_age_days'] + 1)
    df['txn_per_day'] = df['txn_frequency'] / (df['account_age_days'] + 1)
    df['time_risk'] = df['time_of_day'].map({'morning': 1, 'afternoon': 1, 'evening': 2, 'night': 3})
    return df

# 3. Train the model
def train_model():
    print("Generating data...")
    df = generate_data(25000)

    print("Adding features...")
    df = add_features(df)

    df = df.drop(columns=['currency', 'device_type'])

    features = df.drop('fraud', axis=1)
    target = df['fraud']

    categorical_features = ['country', 'time_of_day']
    numeric_features = ['amount', 'account_age_days', 'txn_frequency', 'country_risk',
                        'amount_to_age_ratio', 'txn_per_day', 'time_risk']

    print("Building preprocessor...")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    print("Transforming and oversampling with SMOTE...")
    X_processed = preprocessor.fit_transform(features)
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X_processed, target)

    print("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.07,
        scale_pos_weight=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_resampled, y_resampled)

    print("Evaluating model...")
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    X_test_transformed = preprocessor.transform(X_test)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    print("\nEvaluation Report (Threshold = 0.5):")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # Save model only if performance is satisfactory
    recall = classification_report(y_test, y_pred, output_dict=True)['True']['recall']
    if recall >= 0.8:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        joblib.dump(pipeline, 'optimized_fraud_model.joblib')
        print("\n✅ Model trained and saved successfully.")
    else:
        print("\n⚠️ Model did NOT meet performance criteria — file not saved.")

# 4. Predict function
def predict(input_data):
    try:
        model = joblib.load('optimized_fraud_model.joblib')
        input_df = pd.DataFrame([input_data])
        input_df = add_features(input_df)
        input_df = input_df.drop(columns=['currency', 'device_type'])
        proba = model.predict_proba(input_df)[0][1]
        return {
            'fraud': bool(proba > 0.65),
            'risk_score': round(proba * 100, 2),
            'features_used': list(model.named_steps['preprocessor'].get_feature_names_out())
        }
    except Exception as e:
        return {'error': str(e)}

# 5. Entry point
if __name__ == '__main__':
    print("Training high-accuracy fraud detection model...")
    train_model()
