import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from transformers import BertTokenizer, TFBertModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib


df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

unique_classes = df_train["Class"].unique()
class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
id_to_class = {v: k for k, v in class_to_id.items()}

def bert_encode(texts, tokenizer, max_len=512):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='np',
            truncation=True
        )
        input_ids.append(encoded['input_ids'].flatten())
        attention_masks.append(encoded['attention_mask'].flatten())

    return np.array(input_ids), np.array(attention_masks)

def build_model(bert_layer, max_len=512, num_classes=30):

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_masks = Input(shape=(max_len,), dtype=tf.int32, name="attention_masks")

    bert_output = bert_layer(input_ids, attention_mask=attention_masks)
    clf_output = bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token representation
    out = Dense(num_classes, activation='softmax')(clf_output)

    model = Model(inputs=[input_ids, attention_masks], outputs=out)
    model.compile(Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def use_trained_model(text, id_to_class):
    bert_model = tf.keras.models.load_model('eclipse_bert_model_plus.h5', custom_objects={'TFBertModel': TFBertModel})
    svm_model = joblib.load('eclipse_svm_model_plus.pkl')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_plus.pkl')

    max_len = 160
    new_input_ids, new_attention_masks = bert_encode([text], tokenizer, max_len=max_len)
    cls_layer_model = tf.keras.Model(inputs=bert_model.input, outputs=bert_model.layers[-2].output)
    cls_output = cls_layer_model.predict([new_input_ids, new_attention_masks])

    tfidf_features = tfidf_vectorizer.transform([text]).toarray()
    combined_features = np.concatenate((cls_output, tfidf_features), axis=1)

    predicted_class_id = svm_model.predict(combined_features)
    predicted_class_name = id_to_class[predicted_class_id[0]]
    print(f'The predicted class for the new text is: {predicted_class_name}')

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

def train_svm_bert():
    df_train['Class'] = df_train['Class'].map(class_to_id)
    df_test['Class'] = df_test['Class'].map(class_to_id)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_layer = TFBertModel.from_pretrained("bert-base-uncased")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    max_len = 160

    train_input_ids, train_attention_masks = bert_encode(df_train["Incident"], tokenizer, max_len=max_len)
    test_input_ids, test_attention_masks = bert_encode(df_test["Incident"], tokenizer, max_len=max_len)

    train_labels = df_train["Class"].values
    test_labels = df_test["Class"].values

    model = build_model(bert_layer, max_len=max_len, num_classes=len(unique_classes))
    model.summary()

    # Learning rate schedule and early stopping
    lrate = LearningRateScheduler(step_decay)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train model
    train_history = model.fit(
        [train_input_ids, train_attention_masks],
        train_labels,
        validation_split=0.1,
        epochs=8,
        batch_size=16,
        callbacks=[lrate, early_stopping]
    )

    model.save('eclipse_bert_model_plus.h5')
    model = tf.keras.models.load_model('eclipse_bert_model_plus.h5', custom_objects={'TFBertModel': TFBertModel})

    # Extract BERT CLS layer output
    cls_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    X_train_bert = cls_layer_model.predict([train_input_ids, train_attention_masks])
    X_test_bert = cls_layer_model.predict([test_input_ids, test_attention_masks])

    # Compute TF-IDF features
    tfidf_vectorizer.fit(df_train["Incident"])
    X_train_tfidf = tfidf_vectorizer.transform(df_train["Incident"]).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(df_test["Incident"]).toarray()

    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_plus.pkl')

    # Combine BERT and TF-IDF features
    X_train_combined = np.concatenate((X_train_bert, X_train_tfidf), axis=1)
    X_test_combined = np.concatenate((X_test_bert, X_test_tfidf), axis=1)

    base_svm = SVC(kernel='rbf')

    # Bagging and Stacking with Grid Search
    bagging_svm = BaggingClassifier(estimator=base_svm, random_state=42)
    stacking_svm = StackingClassifier(
        estimators=[('svc', SVC(kernel='rbf')), ('logreg', LogisticRegression())],
        final_estimator=LogisticRegression()
    )

    param_grid = {
        'estimator__C': [1, 10, 100, 1000],
        'estimator__gamma': ['scale', 0.001, 0.01, 0.1, 1],
        'n_estimators': [5, 10, 20, 50]
    }
    
    grid_search = GridSearchCV(bagging_svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_combined, train_labels)

    best_bagging_svm = grid_search.best_estimator_
    joblib.dump(best_bagging_svm, 'eclipse_bagging_svm_model.pkl')

    # Stacking model fitting
    stacking_svm.fit(X_train_combined, train_labels)
    joblib.dump(stacking_svm, 'eclipse_stacking_svm_model.pkl')

    # Evaluation
    test_pred_bagging = best_bagging_svm.predict(X_test_combined)
    accuracy_bagging = accuracy_score(test_labels, test_pred_bagging)
    precision_bagging = precision_score(test_labels, test_pred_bagging, average='weighted', zero_division=0)

    test_pred_stacking = stacking_svm.predict(X_test_combined)
    accuracy_stacking = accuracy_score(test_labels, test_pred_stacking)
    precision_stacking = precision_score(test_labels, test_pred_stacking, average='weighted', zero_division=0)

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Bagging SVM - Accuracy: {accuracy_bagging}, Precision: {precision_bagging}')
    print(f'Stacking SVM - Accuracy: {accuracy_stacking}, Precision: {precision_stacking}')

train_svm_bert()
