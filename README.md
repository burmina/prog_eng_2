API для классификации растений с использованием TensorFlow и FastAPI. Модель автоматически загружается с Google Drive при первом запуске.


### Установка зависимостей
pip install -r requirements.txt


### Запуск сервера 
uvicorn main:app --reload

### API Endpoints
GET / - Информация о API

GET /health - Проверка статуса сервиса

POST /predict/ - Классификация изображения растения
