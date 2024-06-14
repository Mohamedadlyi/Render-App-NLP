from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from DataPreprocessing import TextCleaner

class TextClassificationModel:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.pipeline = None

    def build_pipeline(self):
        if self.model_type == 'logistic':
            self.pipeline = Pipeline([
                ('cleaner', TextCleaner()),
                ('vectorizer', CountVectorizer()),
                ('classifier', LogisticRegression())
            ])
        elif self.model_type == 'naive_bayes':
            self.pipeline = Pipeline([
                ('cleaner', TextCleaner()),
                ('vectorizer', CountVectorizer()),
                ('classifier', MultinomialNB())
            ])

    def train(self, X_train, y_train):
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def save(self, filename):
        joblib.dump(self.pipeline, filename)

    def load(self, filename):
        self.pipeline = joblib.load(filename)

    def predict(self, text):
        return self.pipeline.predict([text])
    
    
    
    # Deep Learning Model
class DeepLearningModel:
    def __init__(self, max_len=100):
        self.max_len = max_len
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.model = None

    def build_model(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=self.max_len))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def preprocess_text(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, padding='post', maxlen=self.max_len)

    def train(self, X_train, y_train, validation_split=0.2, epochs=10, patience=3):
        self.tokenizer.fit_on_texts(X_train)
        X_train_padded = self.preprocess_text(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_train_categorical = to_categorical(y_train_encoded)
        self.build_model(len(self.tokenizer.word_index)+1, len(self.label_encoder.classes_))
        early_stopping = EarlyStopping(patience=patience)
        self.model.fit(X_train_padded, y_train_categorical, validation_split=validation_split, epochs=epochs, callbacks=[early_stopping])

    def evaluate(self, X_test, y_test):
        X_test_padded = self.preprocess_text(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred = self.model.predict(X_test_padded)
        y_pred_classes = y_pred.argmax(axis=1)
        return accuracy_score(y_test_encoded, y_pred_classes)

    def save(self, model_filename, tokenizer_filename, label_encoder_filename):
        self.model.save(model_filename)
        joblib.dump(self.tokenizer, tokenizer_filename)
        joblib.dump(self.label_encoder, label_encoder_filename)

    def load(self, model_filename, tokenizer_filename, label_encoder_filename):
        self.model = load_model(model_filename)
        self.tokenizer = joblib.load(tokenizer_filename)
        self.label_encoder = joblib.load(label_encoder_filename)

    def predict(self, text):
        cleaned_text = TextCleaner.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, padding='post', maxlen=self.max_len)
        prediction = self.model.predict(padded_sequence)
        return self.label_encoder.inverse_transform(prediction.argmax(axis=1))