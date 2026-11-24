import os
import json
import chromadb
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from docx import Document

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

if not API_KEY:
    raise ValueError("Не знайдено GEMINI_API_KEY у файлі .env")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=30000)
)

CATEGORIES = [
    "штучний інтелект",
    "машинне навчання",
    "нейронні мережі",
    "RAG"
]

DOCUMENTS = [
    {
        "text": "Штучний інтелект — галузь інформатики, яка створює системи для виконання завдань, що потребують людського інтелекту. До них належать експертні системи та логічні правила.",
        "category": "штучний інтелект"
    },
    {
        "text": "Машинне навчання — напрям штучного інтелекту, у якому алгоритми знаходять закономірності в даних та використовують їх для прогнозування і класифікації.",
        "category": "машинне навчання"
    },
    {
        "text": "Нейронні мережі складаються зі штучних нейронів, об'єднаних у шари. Вони передають сигнали за допомогою ваг та функцій активації.",
        "category": "нейронні мережі"
    },
    {
        "text": "RAG — технологія, яка шукає інформацію у зовнішніх документах або базах даних і передає знайдений контекст мовній моделі для формування відповіді.",
        "category": "RAG"
    }
]

TESTS = [
    {
        "text": "Експертна система використовує логічні правила для прийняття рішень без додаткового навчання.",
        "category": "штучний інтелект"
    },
    {
        "text": "Алгоритм аналізує тисячі прикладів, знаходить закономірності та прогнозує результати для нових даних.",
        "category": "машинне навчання"
    },
    {
        "text": "Модель складається з декількох шарів нейронів, які передають сигнали за допомогою ваг.",
        "category": "нейронні мережі"
    },
    {
        "text": "Система шукає потрібну інформацію у зовнішніх документах і передає її мовній моделі.",
        "category": "RAG"
    },
    {
        "text": "Алгоритм навчання з учителем використовує розмічені дані для класифікації.",
        "category": "машинне навчання"
    }
]

FEW_SHOT = [
    ("Система виконує інтелектуальні завдання за правилами.", "штучний інтелект"),
    ("Алгоритм навчається на прикладах і прогнозує результати.", "машинне навчання"),
    ("Модель складається з шарів взаємопов'язаних нейронів.", "нейронні мережі"),
    ("Модель шукає інформацію в документах перед генерацією відповіді.", "RAG")
]


def get_embeddings(texts, task):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task)
    )
    return [item.values for item in result.embeddings]


def create_rag():
    db = chromadb.PersistentClient(path="chroma_db_lab6")

    try:
        db.delete_collection("documents")
    except Exception:
        pass

    collection = db.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [doc["text"] for doc in DOCUMENTS]
    embeddings = get_embeddings(texts, "RETRIEVAL_DOCUMENT")

    collection.add(
        ids=[str(i) for i in range(len(texts))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"category": doc["category"]} for doc in DOCUMENTS]
    )

    return collection


def get_rag_context(collection):
    contexts = []

    for test in TESTS:
        embedding = get_embeddings(
            [test["text"]],
            "RETRIEVAL_QUERY"
        )

        result = collection.query(
            query_embeddings=embedding,
            n_results=2
        )

        context = "\n".join(result["documents"][0])
        contexts.append(context)

    return contexts


def ask_gemini(method, contexts=None):
    examples = ""

    if method == "Few-shot":
        examples = "\nПриклади класифікації:\n"

        for text, category in FEW_SHOT:
            examples += f"Текст: {text}\nКатегорія: {category}\n"

    results = []

    for i, test in enumerate(TESTS):
        context = ""

        if method == "RAG":
            context = f"\nІнформація з бази документів:\n{contexts[i]}"

        prompt = f"""
Ти класифікуєш навчальні тексти.

Можливі категорії: {", ".join(CATEGORIES)}.

{examples}
{context}

Класифікуй текст:
{test["text"]}

Поверни лише назву однієї категорії зі списку.
Не додавай пояснень.
"""

        answer = "Не отримано"

        for attempt in range(3):
            try:
                print(f"Обробка тексту {i + 1}/5...")

                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=100
                    )
                )

                answer = response.text.strip().replace('"', '')
                break

            except Exception as e:
                print(f"Помилка: {e}")

                if "503" in str(e) or "504" in str(e):
                    if attempt < 2:
                        print("Повторна спроба через 5 секунд...")
                        time.sleep(5)
                    else:
                        print("Не вдалося отримати відповідь.")
                else:
                    print("Інша помилка API.")
                    break

        results.append(answer)
        print(f"{i + 1}. {answer}")

    return results


def calculate_accuracy(answers):
    correct = sum(
        answer.lower() == test["category"].lower()
        for answer, test in zip(answers, TESTS)
    )

    return correct / len(TESTS) * 100


def save_report(prompt, rag, few_shot, contexts):
    doc = Document()

    doc.add_heading("Лабораторна робота", 0)
    doc.add_heading("Порівняння Prompt Engineering, RAG та Few-shot", 1)

    doc.add_paragraph(
        "Мета роботи: дослідити та порівняти підходи "
        "Prompt Engineering, RAG та Few-shot на прикладі "
        "класифікації навчальних текстів."
    )

    methods = [
        ("Prompt Engineering", prompt),
        ("RAG", rag),
        ("Few-shot", few_shot)
    ]

    doc.add_heading("1. Загальні результати", 1)

    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"

    headers = ["№", "Правильна категорія", "Prompt", "RAG", "Few-shot"]

    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = header

    for i, test in enumerate(TESTS):
        cells = table.add_row().cells
        values = [
            str(i + 1),
            test["category"],
            prompt[i],
            rag[i],
            few_shot[i]
        ]

        for cell, value in zip(cells, values):
            cell.text = value

    doc.add_heading("2. Детальні результати Prompt Engineering", 1)

    for i, test in enumerate(TESTS):
        doc.add_heading(f"Тест {i + 1}", 2)
        doc.add_paragraph(f"Вхідний текст: {test['text']}")
        doc.add_paragraph(f"Правильна категорія: {test['category']}")
        doc.add_paragraph(f"Відповідь моделі: {prompt[i]}")

    doc.add_heading("3. Детальні результати RAG", 1)

    for i, test in enumerate(TESTS):
        doc.add_heading(f"Тест {i + 1}", 2)
        doc.add_paragraph(f"Вхідний текст: {test['text']}")
        doc.add_paragraph(f"Знайдений контекст: {contexts[i]}")
        doc.add_paragraph(f"Правильна категорія: {test['category']}")
        doc.add_paragraph(f"Відповідь моделі: {rag[i]}")

    doc.add_heading("4. Детальні результати Few-shot", 1)

    doc.add_paragraph("Приклади, передані моделі:")

    for text, category in FEW_SHOT:
        doc.add_paragraph(f"{text} — {category}")

    for i, test in enumerate(TESTS):
        doc.add_heading(f"Тест {i + 1}", 2)
        doc.add_paragraph(f"Вхідний текст: {test['text']}")
        doc.add_paragraph(f"Правильна категорія: {test['category']}")
        doc.add_paragraph(f"Відповідь моделі: {few_shot[i]}")

    doc.add_heading("5. Точність класифікації", 1)

    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"

    table.rows[0].cells[0].text = "Метод"
    table.rows[0].cells[1].text = "Accuracy"

    for name, answers in methods:
        cells = table.add_row().cells
        cells[0].text = name
        cells[1].text = f"{calculate_accuracy(answers):.1f}%"

    doc.add_heading("6. Порівняння підходів", 1)

    criteria = [
        ("Якість результату", 4, 5, 4),
        ("Простота реалізації", 5, 2, 4),
        ("Необхідні ресурси", 5, 2, 4),
        ("Швидкість підготовки", 5, 2, 4),
        ("Робота з новими даними", 4, 5, 3),
        ("Контроль результату", 3, 5, 4)
    ]

    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"

    for cell, value in zip(
        table.rows[0].cells,
        ["Критерій", "Prompt", "RAG", "Few-shot"]
    ):
        cell.text = value

    for criterion in criteria:
        cells = table.add_row().cells

        for cell, value in zip(cells, criterion):
            cell.text = str(value)

    doc.add_paragraph("Оцінювання виконано за шкалою від 1 до 5. Оцінки є експертними.")

    doc.add_heading("7. Практичне використання", 1)

    doc.add_paragraph(
        "Prompt Engineering підходить для генерації текстів "
        "із заданою структурою та чіткими інструкціями."
    )

    doc.add_paragraph(
        "RAG доцільно використовувати для роботи із зовнішніми "
        "документами та внутрішньою документацією."
    )

    doc.add_paragraph(
        "Fine-tuning дозволяє додатково навчити модель для адаптації "
        "до специфічних завдань і стилю. У цій лабораторній роботі "
        "для порівняння використано Few-shot, а не справжній Fine-tuning."
    )

    doc.add_heading("8. Висновок", 1)

    best = max(
        methods,
        key=lambda item: calculate_accuracy(item[1])
    )

    doc.add_paragraph(
        "Під час лабораторної роботи було реалізовано та порівняно "
        "три підходи: Prompt Engineering, RAG та Few-shot. "
        "Експеримент проводився на п'яти тестових текстах із чотирма категоріями."
    )

    doc.add_paragraph(
        f"Найкращий результат показав метод {best[0]} "
        f"з точністю {calculate_accuracy(best[1]):.1f}%."
    )

    doc.add_paragraph(
        "Вибір підходу залежить від поставленого завдання, "
        "наявних даних та необхідної точності."
    )

    doc.save("results_lab6.docx")
    print("\nWord-звіт збережено: results_lab6.docx")


def main():
    print("ЛАБОРАТОРНА РОБОТА")
    print("Порівняння Prompt Engineering, RAG та Few-shot")

    print("\n1. Створення бази документів...")
    collection = create_rag()

    print("\n2. Пошук контексту RAG...")
    contexts = get_rag_context(collection)

    print("\n3. Prompt Engineering...")
    prompt = ask_gemini("Prompt")

    print("\n4. RAG...")
    rag = ask_gemini("RAG", contexts)

    print("\n5. Few-shot...")
    few_shot = ask_gemini("Few-shot")

    print("\nРЕЗУЛЬТАТИ")

    for i, test in enumerate(TESTS):
        print(f"\nТест {i + 1}: {test['text']}")
        print(f"Правильна категорія: {test['category']}")
        print(f"Prompt: {prompt[i]}")
        print(f"RAG: {rag[i]}")
        print(f"Few-shot: {few_shot[i]}")

    print("\nТОЧНІСТЬ")

    print(f"Prompt: {calculate_accuracy(prompt):.1f}%")
    print(f"RAG: {calculate_accuracy(rag):.1f}%")
    print(f"Few-shot: {calculate_accuracy(few_shot):.1f}%")

    print("\nСтворення Word-звіту...")
    save_report(prompt, rag, few_shot, contexts)

    print("\nЕКСПЕРИМЕНТ ЗАВЕРШЕНО")


if __name__ == "__main__":
    main()