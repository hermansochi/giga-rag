"""
preload_model.py

Скрипт предзагрузки всех необходимых моделей в Docker-контейнер.
Загружает:
1. CrossEncoder модель (DiTy/cross-encoder-russian-msmarco)
2. spaCy модель ru_core_news_md

Все модели сохраняются в локальные папки, указанные в config.py,
чтобы контейнер использовал их без повторного скачивания.
"""

import os
import sys
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

from src.config import settings


def preload_cross_encoder():
    """Загружает и сохраняет CrossEncoder модель по пути из настроек."""
    model_name = settings.CROSS_ENCODER_MODEL
    save_path = Path(settings.CROSS_ENCODER_MODEL_PATH)

    print(f"🔄 Проверка CrossEncoder модели: {model_name}")

    if save_path.exists() and any(save_path.iterdir()):
        print(f"✅ CrossEncoder модель уже существует в {save_path}")
        return

    print(f"📥 Скачивание CrossEncoder модели {model_name}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(save_path),
        local_dir_use_symlinks=False,
    )

    # Проверка работоспособности
    try:
        model = CrossEncoder(str(save_path))
        print(f"✅ CrossEncoder успешно загружена и проверена в {save_path}")
    except Exception as e:
        print(f"⚠️ Модель скачана, но возникла ошибка при инициализации: {e}")


def preload_spacy():
    """Загружает spaCy модель ru_core_news_md и сохраняет в указанный путь."""
    model_name = settings.SPACY_MODEL_NAME
    save_path = Path(settings.SPACY_MODEL_PATH)

    print(f"🔄 Проверка spaCy модели: {model_name}")

    if save_path.exists() and any(save_path.iterdir()):
        print(f"✅ spaCy модель уже существует в {save_path}")
        return

    print(f"📥 Скачивание spaCy модели {model_name}...")

    try:
        # Скачиваем через spacy
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            capture_output=True,
            text=True
        )

        # Определяем стандартный путь установки spaCy
        default_path = Path("/usr/local/lib/python3.12/site-packages/spacy/data") / model_name

        if default_path.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.exists():
                shutil.rmtree(save_path)

            shutil.copytree(default_path, save_path)
            print(f"✅ spaCy модель успешно скопирована в {save_path}")
        else:
            print("⚠️ Модель скачана, но не найдена в стандартном пути spaCy")

    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при скачивании spaCy модели: {e.stderr}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка при работе со spaCy: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Запуск предзагрузки моделей для RAG-системы")
    print("=" * 60)

    preload_cross_encoder()
    preload_spacy()

    print("\n🎉 Предзагрузка всех моделей завершена!")
    print(f"   CrossEncoder → {settings.CROSS_ENCODER_MODEL_PATH}")
    print(f"   spaCy         → {settings.SPACY_MODEL_PATH}")