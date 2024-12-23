from transformers import AutoModel, AutoTokenizer
import os

MODEL_NAME = "DeepPavlov/rubert-base-cased"
SAVE_DIRECTORY = "rubert_model"

def download_and_save_model(model_name, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

if __name__ == "__main__":
    print("Скачивание и сохранение модели RuBERT...")
    download_and_save_model(MODEL_NAME, SAVE_DIRECTORY)
    print("Модель успешно сохранена.")