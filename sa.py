import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from collections import defaultdict
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class SentimentAnalyzer:
    """Класс для анализа тональности текста на русском языке"""
    
    def __init__(self):
        # Инициализация стоп-слов и словаря тональности
        self.russian_stopwords = set(stopwords.words('russian'))
        self.sentiment_dict = self._load_sentiment_dict()
    
    def _load_sentiment_dict(self):
        """Загружает простой словарь тональности для русского языка"""
        # В реальном проекте лучше использовать предобученную модель или более полный словарь
        return {
            'хороший': 1, 'отличный': 1, 'прекрасный': 1, 'замечательный': 1,
            'плохой': -1, 'ужасный': -1, 'отвратительный': -1, 'недостаток': -1,
            'любовь': 1, 'нравится': 1, 'восхищение': 1,
            'ненависть': -1, 'разочарование': -1, 'проблема': -1
        }
    
    def _preprocess_text(self, text):
        """Предварительная обработка текста"""
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление пунктуации
        text = ''.join([char for char in text if char not in punctuation])
        
        # Токенизация
        tokens = word_tokenize(text, language='russian')
        
        # Удаление стоп-слов
        tokens = [word for word in tokens if word not in self.russian_stopwords]
        
        return tokens
    
    def analyze_sentiment(self, text):
        """Анализирует тональность текста"""
        if not text.strip():
            return {'sentiment': 'neutral', 'score': 0, 'tokens': []}
        
        tokens = self._preprocess_text(text)
        if not tokens:
            return {'sentiment': 'neutral', 'score': 0, 'tokens': []}
        
        # Подсчет баллов тональности
        score = sum(self.sentiment_dict.get(word, 0) for word in tokens)
        
        # Определение общей тональности
        if score > 0:
            sentiment = 'positive'
        elif score < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Нормализация оценки от -1 до 1
        normalized_score = score / len(tokens) if tokens else 0
        
        return {
            'sentiment': sentiment,
            'score': normalized_score,
            'tokens': tokens,
            'word_scores': {word: self.sentiment_dict.get(word, 0) for word in tokens}
        }

class SentimentAnalysisApp:
    """Главное окно приложения"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор тональности текста")
        self.root.geometry("800x600")
        self.setup_ui()
        
        # Инициализация анализатора
        self.analyzer = SentimentAnalyzer()
        
        # Настройка стилей
        self.setup_styles()
    
    def setup_styles(self):
        """Настройка стилей интерфейса"""
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('Positive.TLabel', foreground='green', font=('Helvetica', 10, 'bold'))
        style.configure('Negative.TLabel', foreground='red', font=('Helvetica', 10, 'bold'))
        style.configure('Neutral.TLabel', foreground='blue', font=('Helvetica', 10, 'bold'))
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной контейнер
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        header = ttk.Label(
            main_frame, 
            text="Анализатор эмоциональной окраски текста", 
            font=('Helvetica', 14, 'bold')
        )
        header.pack(pady=(0, 20))
        
        # Поле ввода текста
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Введите текст для анализа:").pack(anchor=tk.W)
        
        self.text_input = tk.Text(
            input_frame, 
            height=10, 
            wrap=tk.WORD, 
            font=('Helvetica', 11),
            padx=10,
            pady=10
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)
        
        # Кнопка анализа
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.analyze_btn = ttk.Button(
            btn_frame,
            text="Анализировать текст",
            command=self.on_analyze_click
        )
        self.analyze_btn.pack()
        
        # Результаты анализа
        result_frame = ttk.LabelFrame(
            main_frame, 
            text="Результаты анализа",
            padding=10
        )
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Общий результат
        self.result_label = ttk.Label(
            result_frame,
            text="Здесь будет отображен результат анализа",
            font=('Helvetica', 11)
        )
        self.result_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Детализированная информация
        details_frame = ttk.Frame(result_frame)
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        # Таблица с оценками слов
        columns = ('word', 'score')
        self.words_table = ttk.Treeview(
            details_frame,
            columns=columns,
            show='headings',
            height=8
        )
        
        self.words_table.heading('word', text='Слово')
        self.words_table.heading('score', text='Оценка')
        self.words_table.column('word', width=200)
        self.words_table.column('score', width=100, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(
            details_frame,
            orient=tk.VERTICAL,
            command=self.words_table.yview
        )
        self.words_table.configure(yscroll=scrollbar.set)
        
        self.words_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def on_analyze_click(self):
        """Обработчик нажатия кнопки анализа"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Пустой ввод", "Пожалуйста, введите текст для анализа")
            return
        
        # Показываем индикатор загрузки
        self.analyze_btn.config(state=tk.DISABLED, text="Анализ...")
        self.result_label.config(text="Анализируем текст...")
        self.root.update()
        
        # Запускаем анализ в отдельном потоке
        Thread(target=self.perform_analysis, args=(text,), daemon=True).start()
    
    def perform_analysis(self, text):
        """Выполняет анализ текста и обновляет интерфейс"""
        try:
            result = self.analyzer.analyze_sentiment(text)
            
            # Обновляем интерфейс в основном потоке
            self.root.after(0, lambda: self.display_results(result))
        
        except Exception as e:
            self.root.after(0, lambda: self.display_error(str(e)))
        
        finally:
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL, text="Анализировать текст"))
    
    def display_results(self, result):
        """Отображает результаты анализа"""
        # Общий результат
        sentiment = result['sentiment']
        score = result['score']
        
        if sentiment == 'positive':
            style = 'Positive.TLabel'
            emoji = "😊 Положительный"
        elif sentiment == 'negative':
            style = 'Negative.TLabel'
            emoji = "😠 Отрицательный"
        else:
            style = 'Neutral.TLabel'
            emoji = "😐 Нейтральный"
        
        self.result_label.config(
            text=f"{emoji} тон. Оценка: {score:.2f}",
            style=style
        )
        
        # Детали по словам
        self.words_table.delete(*self.words_table.get_children())
        
        for word, score in result['word_scores'].items():
            tag = ''
            if score > 0:
                tag = 'positive'
            elif score < 0:
                tag = 'negative'
            
            self.words_table.insert(
                '', 
                tk.END, 
                values=(word, score),
                tags=(tag,)
            )
        
        # Настройка цветов для таблицы
        self.words_table.tag_configure('positive', foreground='green')
        self.words_table.tag_configure('negative', foreground='red')
    
    def display_error(self, message):
        """Отображает сообщение об ошибке"""
        self.result_label.config(
            text=f"Ошибка: {message}",
            foreground='red'
        )
        self.words_table.delete(*self.words_table.get_children())

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()