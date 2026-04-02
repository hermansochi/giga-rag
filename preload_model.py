import os
from sentence_transformers import CrossEncoder

# Импортируем для принудительного отображения прогресс-бара
from huggingface_hub import snapshot_download

model_name = (
    "DiTy/cross-encoder-russian-msmarco"  # или ваш settings.CROSS_ENCODER_MODEL
)
save_path = "./model_data"

print(f"Начинаю проверку модели: {model_name}")

if not os.path.exists(save_path):
    print("Папка не найдена. Начинаю скачивание...")
    # Способ с явным прогресс-баром
    snapshot_download(
        repo_id=model_name, local_dir=save_path, local_dir_use_symlinks=False
    )

    # Инициализируем модель из уже скачанной папки, чтобы убедиться, что всё ок
    model = CrossEncoder(save_path)
    print(f"Модель успешно сохранена в {save_path}")
else:
    print(f"Модель уже существует в {save_path}")
