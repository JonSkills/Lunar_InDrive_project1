import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import argparse

# Список макета (mock) данных для юриспруденции на РУ и КЗ
LEGAL_NLI_DATA = [
    # Русские примеры
    {"premise": "Разрешено курение в общественных местах.", "hypothesis": "Запрещено курение в парках и скверах.", "label": 0}, # Contradiction
    {"premise": "Штраф составляет 10 МРП.", "hypothesis": "Размер штрафа равен 10 Месячным Расчетным Показателям.", "label": 1}, # Entailment (дублирование/повтор)
    {"premise": "Гражданин имеет право на адвоката.", "hypothesis": "Государство обеспечивает безопасность на дорогах.", "label": 2}, # Neutral
    
    # Казахские примеры
    {"premise": "Қоғамдық орындарда шылым шегуге рұқсат етіледі.", "hypothesis": "Саябақтарда шылым шегуге тыйым салынады.", "label": 0}, # Contradiction
    {"premise": "Айыппұл мөлшері 10 АЕК құрайды.", "hypothesis": "Айыппұл 10 айлық есептік көрсеткішке тең.", "label": 1}, # Entailment
    {"premise": "Азаматтың адвокат алуға құқығы бар.", "hypothesis": "Мемлекет жол қауіпсіздігін қамтамасыз етеді.", "label": 2}, # Neutral
    
    # Кросс-языковые примеры
    {"premise": "Запрещено курение в парках.", "hypothesis": "Саябақтарда шылым шегуге рұқсат етіледі.", "label": 0}, # Contradiction (RU-KZ)
]

class LegalNLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Энкодинг для NLI (premise + [SEP] + hypothesis)
        encoding = self.tokenizer(
            item["premise"],
            item["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Базовая мультиязычная модель")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--output_dir", type=str, default="../local_models/legal_nli_model", help="Куда сохранить веса")
    args = parser.parse_args()

    print(f"[*] Инициализация дообучения (Fine-Tuning) {args.model_name} для Legal NLI (Kazakh/Russian)...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 3 labels: 0=Contradiction, 1=Entailment, 2=Neutral
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)

    dataset = LegalNLIDataset(LEGAL_NLI_DATA, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_steps=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("[*] Запуск цикла обучения (Training Loop)...")
    trainer.train()

    print(f"[*] Обучение завершено! Сохранение модели в {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[*] Готово.")

if __name__ == "__main__":
    train()
