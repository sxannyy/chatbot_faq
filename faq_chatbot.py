# faq_chatbot.py
import os
import json
import numpy as np
import tensorflow as tf
from transformers import pipeline, AutoModel, AutoTokenizer

MODEL_PATH = "faq_model.keras"
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

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def get_embedding(text, pipe):
    embedding = pipe(text)[0]
    return np.mean(embedding, axis=0)

def get_answer(question, questions, answers, model, pipe):
    q_emb = get_embedding(question, pipe)
    scores = []
    for answer in answers:
        a_emb = get_embedding(answer, pipe)
        input_vector = np.concatenate([q_emb, a_emb])
        score = model.predict(np.expand_dims(input_vector, axis=0))[0][0]
        scores.append(score)
    best_match = np.argmax(scores)
    return answers[best_match]

def console_interface(model, questions, answers, pipe):
    print("FAQ Чат-бот (введите 'выход' для завершения)")
    while True:
        user_question = input("Ваш вопрос: ")
        if user_question.lower() == 'выход':
            break
        response = get_answer(user_question, questions, answers, model, pipe)
        print(f"Ответ: {response}")

if __name__ == "__main__":
    print("Загрузка данных...")
    questions, answers = load_data(DATA_PATH)

    print("Загрузка модели...")
    model = load_model(MODEL_PATH)
    console_interface(model, questions, answers, pipe)