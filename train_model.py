import os
import json
import numpy as np
import tensorflow as tf
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer

DATA_PATH = "faq_data.json"

MODEL_DIR = "rubert_model"

print("Инициализация модели RuBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR)
pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return questions, answers

def build_dataset(questions, answers, pipe):
    dataset = []
    for i in range(len(questions)):
        q_emb = np.mean(pipe(questions[i])[0], axis=0)
        a_emb = np.mean(pipe(answers[i])[0], axis=0)
        dataset.append((q_emb, a_emb))
    return np.array(dataset)

def create_training_data(dataset):
    X, Y = [], []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            X.append(np.concatenate([dataset[i][0], dataset[j][1]]))
            Y.append(1 if i == j else 0)
    print(Y)
    return np.array(X), np.array(Y)

def build_model(input_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(256, activation='selu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])
    return model

if __name__ == "__main__":
    print("Загрузка данных...")
    questions, answers = load_data(DATA_PATH)

    print("Построение датасета с RuBERT...")
    dataset = build_dataset(questions, answers, pipe)

    print("Создание тренировочных данных...")
    X, Y = create_training_data(dataset)

    print("Создание модели...")
    model = build_model(X.shape[1])

    print("Обучение модели... Подождите немного...")
    model.fit(X, Y, epochs=150, class_weight={0:1,1:56}, batch_size=16, validation_split=0.1)

    print("Сохранение модели в faq_model.keras...")
    model.save('faq_model.keras')
    print("Модель успешно сохранена!")