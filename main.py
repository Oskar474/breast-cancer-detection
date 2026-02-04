import os
from random import random
from socketserver import DatagramRequestHandler
import joblib
import pandas as pd
from keras.src.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam


def prepare_data(df):
    df = df.copy()

    df.columns = df.columns.str.replace(" ", "_")
    df.drop('id', axis=1, inplace=True)
    df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
    extracted_features = get_most_important_features(df)

    y = df['diagnosis']
    X = df[extracted_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def get_most_important_features(df, threshold=0.2):
    corr = df.corr()

    important_features = (
        corr['diagnosis']
        .abs()
        .drop('diagnosis')
        .loc[lambda x: x > threshold]
        .index
        .tolist()
    )


    return important_features

def get_simple_MLP():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(25,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name = "simple_MLP")

    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def get_tuned_model():
    model = Sequential([
        Dense(40, activation='relu', input_shape=(25,)),
        Dense(40, activation='relu'),
        Dropout(rate=0.4),
        Dense(40, activation='relu'),
        Dropout(rate=0.3),
        Dense(1, activation='sigmoid')
    ], name="tuned_NN")

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def get_paper_model():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(25,)),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dropout(0.4),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name="paper_NN")

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def get_small_cnn_model():
    model = Sequential([
        Reshape((25, 1), input_shape=(25,)),

        Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),

        Dense(1, activation='sigmoid')
    ], name="cnn_small")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_cnn_model():
    model = Sequential([
        Reshape((25, 1), input_shape=(25,)),

        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),

        Dense(1, activation='sigmoid')
    ], name="cnn")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_model_for_tuner(hp):
    model = Sequential()

    model.add(Dense(
        units=hp.Int('units_input', min_value=8, max_value=64, step=16),
        activation='relu',
        input_shape=(25,)
    ))

    for i in range(hp.Int('num_hidden_layers', 1, 4)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=8, max_value=64, step=16),
            activation='relu'
        ))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)))

    model.add(Dense(1, activation='sigmoid'))

    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_evaluate_model(model, X_train, y_train, X_test, y_test):

    if "keras" in str(type(model)).lower():

        checkpoint = ModelCheckpoint(
            filepath="best_nn_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint],
            verbose=0
        )

        plot_metrics(history)

        best_model = load_model("best_nn_model.keras")

        predictions_prob = best_model.predict(X_test, verbose=0)
        predictions = (predictions_prob > 0.4).astype(int).reshape(-1)
        os.remove("best_nn_model.keras")
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    eval_df = pd.DataFrame(
        [[model.name if hasattr(model, "name") else type(model).__name__,
          accuracy, f1, precision, recall, balanced_accuracy]],
        columns=['model', 'accuracy', 'f1_score', 'precision', 'recall', 'balanced_accuracy']
    )

    return eval_df

def plot_metrics(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png")
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss.png")
    plt.show()

def compute_mean_evaluations(models, X_train, y_train, X_test, y_test, n_runs = 1):
    all_results = []

    for i in range(n_runs):
        for model_func in models:
            model = model_func()
            current_model_eval = train_evaluate_model(model, X_train, y_train, X_test, y_test)
            all_results.append(current_model_eval)

    results_df = pd.concat(all_results, ignore_index=True)
    mean_results = results_df.groupby("model").mean().reset_index()
    print(mean_results.to_string(index=False))

def get_models():
    return [
        get_simple_MLP,
        get_paper_model,
        get_cnn_model,
        get_small_cnn_model,
        get_tuned_model,
        lambda: RandomForestClassifier(),
        lambda: LogisticRegression(),
        lambda: xgb.XGBClassifier(),
        lambda: DecisionTreeClassifier()
    ]


def get_tuner():
    tuner = RandomSearch(
        build_model_for_tuner,
        objective='val_accuracy',
        max_trials=100,
        executions_per_trial=3,
        directory='tuner_results',
        project_name='breast_cancer_nn'
    )
    return tuner

if __name__ == "__main__":

    df = pd.read_csv('data/breast-cancer.csv')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    models = get_models()

    X_train, X_test, y_train, y_test = prepare_data(df)

    compute_mean_evaluations(models, X_train, y_train, X_test, y_test, n_runs = 1)
