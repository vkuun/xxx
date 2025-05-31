from textblob import TextBlob  # Библиотека для анализа тональности текста
import tkinter as tk  # Стандартная библиотека для создания GUI
from tkinter import ttk, messagebox  # Дополнительные виджеты и диалоговые окна

class SentimentAnalyzerApp:
    def __init__(self, root):
        """Инициализация главного окна приложения"""
        self.root = root
        self.root.title("Анализатор тональности")  # Заголовок окна
        self.root.geometry("400x300")  # Размер окна (ширина x высота)
        
        self.create_widgets()  # Создание и размещение элементов интерфейса
    
    def create_widgets(self):
        """Создание всех элементов пользовательского интерфейса"""
        
        # Метка-инструкция для пользователя
        self.label = ttk.Label(self.root, text="Введите текст для анализа:")
        self.label.pack(pady=10)  # Размещение с отступом 10 пикселей сверху и снизу
        
        # Текстовое поле для ввода (многострочное)
        self.text_entry = tk.Text(self.root, height=10, width=50)
        self.text_entry.pack(pady=5)  # Размещение с отступом 5 пикселей
        
        # Кнопка для запуска анализа
        self.analyze_button = ttk.Button(
            self.root, 
            text="Анализировать", 
            command=self.analyze_sentiment  # Привязка метода-обработчика
        )
        self.analyze_button.pack(pady=10)  # Размещение с отступом
        
        # Метка для заголовка результата
        self.result_label = ttk.Label(self.root, text="Результат:")
        self.result_label.pack(pady=5)
        
        # Переменная для хранения и отображения результата
        self.result_var = tk.StringVar()
        
        # Виджет для отображения результата анализа
        self.result_display = ttk.Label(
            self.root, 
            textvariable=self.result_var,  # Привязка к переменной
            font=('Helvetica', 10, 'bold'),  # Настройки шрифта
            wraplength=380  # Максимальная длина строки перед переносом
        )
        self.result_display.pack()
    
    def analyze_sentiment(self):
        """Метод для анализа тональности введенного текста"""
        # Получаем текст из виджета Text (от начала до конца)
        text = self.text_entry.get("1.0", tk.END).strip()
        
        # Проверка на пустой ввод
        if not text:
            messagebox.showwarning("Ошибка", "Пожалуйста, введите текст для анализа")
            return
        
        try:
            # Создаем объект TextBlob для анализа текста
            analysis = TextBlob(text)
            
            # Получаем числовую оценку тональности (-1 до 1)
            polarity = analysis.sentiment.polarity
            
            # Определяем категорию тональности на основе числовой оценки
            if polarity > 0:
                result = "Положительный 😊"
            elif polarity < 0:
                result = "Отрицательный 😠"
            else:
                result = "Нейтральный 😐"
            
            # Формируем и устанавливаем результат (с округлением до 2 знаков)
            self.result_var.set(f"Тональность: {result} (Оценка: {polarity:.2f})")
            
        except Exception as e:
            # Обработка возможных ошибок анализа
            messagebox.showerror("Ошибка", f"Не удалось проанализировать текст: {str(e)}")

if __name__ == "__main__":
    # Точка входа в приложение
    root = tk.Tk()  # Создание главного окна
    app = SentimentAnalyzerApp(root)  # Создание экземпляра приложения
    root.mainloop()  # Запуск основного цикла обработки событий