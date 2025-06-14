import nltk
import random
from telegram import ReplyKeyboardMarkup

from data_preparation import get_emo_dict, get_dialogues, get_menu, get_intent_dataset
from nlp_functions import clean_text, lemmatize_text, extract_entities, analyze_sentiment, correct_text
from intent_classifier import IntentClassifier, train_and_save_model

from config import MODEL_FILE_PATH, INTENT_DATASET_FILE_PATH, MENU_FILE_PATH, DIALOGUES_FILE_PATH, EMO_DICT_FILE_PATH

# Загрузка данных
EMO_DICT = get_emo_dict(EMO_DICT_FILE_PATH)
DIALOGUES = get_dialogues(DIALOGUES_FILE_PATH)
MENU = get_menu(MENU_FILE_PATH)
INTENT_DATASET = get_intent_dataset(INTENT_DATASET_FILE_PATH, MENU)

# Загрузка модели классификатора
try:
    INTENT_CLASSIFIER = IntentClassifier.load(MODEL_FILE_PATH)
except FileNotFoundError:
    print("Model not found, training new one...")
    INTENT_CLASSIFIER = train_and_save_model(INTENT_DATASET_FILE_PATH, MODEL_FILE_PATH)


class RestaurantAssistantBot:
    def __init__(self):
        self.context = {}
        self.carts = {}

        self.menu_keyboard = ReplyKeyboardMarkup(
            [
                ["🛒 Корзина", "📋 Меню"],
                ["❌ Очистить корзину", "✅ Оформить заказ"]
            ],
            resize_keyboard=True,
            one_time_keyboard=False
        )

    def _get_user_cart(self, user_id):
        """Получение корзины пользователя"""
        if user_id not in self.carts:
            self.carts[user_id] = []

        return self.carts[user_id]

    def _calculate_order_stats(self, cart):
        """Вычисление средних параметров заказа"""
        if len(cart) == 0:
            return None

        stats = {
            'spiciness': 0.0,
            'vegetarian': 0.0,
            'saltiness': 0.0,
            'sweetness': 0.0,
            'count': len(cart)
        }

        for item in cart:
            stats['spiciness'] += item['spiciness']
            stats['vegetarian'] += item['vegetarian']
            stats['saltiness'] += item['saltiness']
            stats['sweetness'] += item['sweetness']

        stats['spiciness'] /= stats['count']
        stats['vegetarian'] /= stats['count']
        stats['saltiness'] /= stats['count']
        stats['sweetness'] /= stats['count']

        return stats

    def _find_recommendation(self, cart):
        """Поиск блюда для рекомендации"""
        if len(cart) == 0:
            return None

        order_stats = self._calculate_order_stats(cart)

        if order_stats is None:
            return None

        best_match = None
        best_score = -1

        for item, item_data in MENU.items():
            if any(cart_item['name_lower'] == item_data["name_lower"] for cart_item in cart):
                continue

            score = 0.0
            score += 1.0 - abs(item_data['spiciness'] - order_stats['spiciness'])
            score += 1.0 - abs(item_data['saltiness'] - order_stats['saltiness'])
            score += 1.0 - abs(item_data['sweetness'] - order_stats['sweetness'])

            # Штраф за невегетарианское блюдо в вегетарианском заказе
            if order_stats['vegetarian'] > 0.5 and item_data['vegetarian'] < 0.5:
                score -= 0.5

            # Штраф за вегетарианское блюдо в невегетарианском заказе
            if order_stats['vegetarian'] < 0.5 and item_data['vegetarian'] > 0.5:
                score -= 0.15

            # Бонус за десерт после основного блюда
            if order_stats['sweetness'] < 0.3 and item_data['sweetness'] > 0.7:
                score += 0.2

            if score > best_score:
                best_score = score
                best_match = item_data

        return best_match

    def _generate_response(self, text):
        """Генерация ответа на основе датасета диалогов"""
        prepared_text = lemmatize_text(correct_text(clean_text(text)))
        words = set(prepared_text.split(" "))

        mini_dataset = []
        for word in words:
            if word in DIALOGUES:
                mini_dataset += DIALOGUES[word]

        responses = []
        for question, answer in mini_dataset:
            prepared_question = lemmatize_text(correct_text(clean_text(question)))

            if abs(len(prepared_text) - len(prepared_question)) / len(prepared_question) < 0.2:
                distance = nltk.edit_distance(prepared_text, prepared_question)
                weighted_distance = distance / len(prepared_question)

                if weighted_distance < 0.2:
                    responses.append([weighted_distance, question, answer])

        if responses:
            return min(responses, key=lambda x: x[0])[2]

    def _handle_greeting(self):
        """Обработка намерения приветствия"""
        responses = INTENT_DATASET["intents"]["greeting"]["responses"]

        return random.choice(responses)

    def _handle_menu_request(self):
        """Обработка намерения получить меню"""
        responses = INTENT_DATASET["intents"]["menu_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        menu_text = self.show_menu()

        response += menu_text

        return response

    def _handle_cart_request(self, user_id):
        """Обработка намерения получить данные о корзине"""
        responses = INTENT_DATASET["intents"]["cart_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        self._get_user_cart(user_id)
        cart_text = self.show_cart(user_id)

        response += cart_text

        return response

    def _handle_order_request(self, entities, user_id):
        """Обработка намерения заказать блюдо"""
        responses = INTENT_DATASET["intents"]["order_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        possible_items = [entity["normal"] for entity in entities if entity["type"] == "MENU_ITEM"]
        prepared_possible_items = [lemmatize_text(correct_text(clean_text(item))) for item in possible_items]

        ordered_item = None
        for item, item_data in MENU.items():
            if lemmatize_text(correct_text(clean_text(item_data["name"]))) in prepared_possible_items:
                ordered_item = (item, item_data)
                break

        if ordered_item is not None:
            response += f"{ordered_item[1]["name"]} - {ordered_item[1]["price"]} руб.\n\n"

            self._get_user_cart(user_id)
            self.carts[user_id].append(MENU[ordered_item[0]])

            response += f"Товар добавлен в ваш заказ"

            return response

        return "Извините, я не понял, что вы хотите заказать. Можете уточнить или попробовать еще раз?"

    def _handle_price_request(self, entities):
        """Обработка намерения узнать цену блюда"""
        responses = INTENT_DATASET["intents"]["price_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        possible_items = [entity["normal"] for entity in entities if entity["type"] == "MENU_ITEM"]
        prepared_possible_items = [lemmatize_text(correct_text(clean_text(item))) for item in possible_items]

        correct_item = None
        for item, item_data in MENU.items():
            if lemmatize_text(correct_text(clean_text(item_data["name"]))) in prepared_possible_items:
                correct_item = (item, item_data)
                break

        if correct_item is not None:
            response += f"{correct_item[1]["name"]} - {correct_item[1]["price"]} руб.\n"
            response += f"{correct_item[1]["description"]}\n\n"
            response += f"Если хотите заказать это - просто напишите об этом"

            return response

        return (f"Извините, я не понял, на какие блюда вы хотите узнать цену, "
                f"попробуйте снова или посмотрите все меню целиком")

    def _handle_complete_order_request(self, user_id):
        """Обработка намерения оформить заказ"""
        responses = INTENT_DATASET["intents"]["complete_order_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        response += self.complete_order(user_id)

        return response

    def _handle_clear_cart_request(self, user_id):
        """Обработка намерения очистить корзину"""
        responses = INTENT_DATASET["intents"]["clear_cart_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        self.clear_cart(user_id)

        return response

    def _handle_goodbye(self):
        """Обработка намерения прощания"""
        responses = INTENT_DATASET["intents"]["goodbye"]["responses"]

        return random.choice(responses)

    def _handle_generated_answer(self, response):
        """Обработка сгенерированного по датасету диалогов ответа"""
        return response

    def _handle_unknown(self):
        """Обработка нераспознанного намерения"""
        responses = INTENT_DATASET["failure_phrases"]

        return random.choice(responses)

    def show_menu(self):
        """Генерация текста меню"""
        menu_text = "🍽 *Наше меню*:\n\n"
        for item, details in MENU.items():
            menu_text += f"*{details["name"]}*:\n"
            menu_text += f"{details["description"]}\n"
            menu_text += f"Цена: {details["price"]} руб.\n\n"
        menu_text += "Если хотите что-то заказать, то напишите об этом"

        return menu_text

    def show_cart(self, user_id):
        """Генерация текста содержимого корзины"""
        cart = self._get_user_cart(user_id)

        if len(cart) == 0:
            return "Ваша корзина пуста."

        total = 0

        cart_text = "🛒 *Ваша корзина*:\n\n"
        for item in cart:
            cart_text += f"• {item["name"]} - {item["price"]} руб.\n"
            total += item["price"]
        cart_text += f"\n*Итого: {total} руб.*"

        return cart_text

    def clear_cart(self, user_id):
        """Очистка корзины"""
        self.carts[user_id] = []

        return "Корзина очищена"

    def complete_order(self, user_id):
        """Оформление заказа"""
        cart = self._get_user_cart(user_id)

        if len(cart) == 0:
            return "Ваша корзина пуста. Добавьте что-нибудь из меню."

        total = 0

        order_text = "✅ *Ваш заказ оформлен!*\n\n"
        for item in cart:
            order_text += f"• {item["name"]} - {item["price"]} руб.\n"
            total += item["price"]
        order_text += f"\n*Итого: {total} руб.*\n\n"
        order_text += "Спасибо за заказ! Ожидайте подтверждения."

        self.clear_cart(user_id)

        if self.context[user_id]["sentiment"] > 0.4 and self.context[user_id]["recommendation_counter"] > 10:
            recommendation = self._find_recommendation(cart)

            if recommendation is not None:
                order_text += ("\n\nМы провели анализ на основе вашего заказа и думаем это может вам понравиться. "
                               "Вы можете заказать это прямо сейчас или когда посетите нас в следующий раз\n\n")
                order_text += "🌟 *Рекомендуем попробовать*:\n"
                order_text += f"{recommendation['name']} - {recommendation['price']} руб.\n"
                order_text += recommendation['description']

                self.context[user_id]["recommendation_counter"] = 0

        return order_text

    def handle_message(self, text, user_id):
        """Обработка сообщения"""
        prepared_text = lemmatize_text(correct_text(clean_text(text)))

        sentiment = analyze_sentiment(prepared_text, EMO_DICT)
        entities = extract_entities(prepared_text)
        potential_intent = INTENT_CLASSIFIER.predict(prepared_text)

        intent = None
        for example in INTENT_DATASET["intents"][potential_intent]["examples"]:
            prepared_example = lemmatize_text(correct_text(clean_text(example)))
            distance = nltk.edit_distance(prepared_text, prepared_example)
            if prepared_example and distance / len(prepared_example) <= 0.5:
                intent = potential_intent
                break

        if user_id not in self.context:
            self.context[user_id] = {
                "last_intent": None,
                "sentiment": 0,
                "entities": [],
                "recommendation_counter": 5
            }

        new_user_sentiment = (self.context[user_id]["sentiment"] + sentiment) / 2
        if new_user_sentiment > 1:
            new_user_sentiment = 1
        elif new_user_sentiment < -1:
            new_user_sentiment = -1

        new_user_recommendation_counter = self.context[user_id]["recommendation_counter"] + 1

        self.context[user_id] = {
            "last_intent": None,
            "sentiment": new_user_sentiment,
            "entities": entities,
            "recommendation_counter": new_user_recommendation_counter
        }

        if intent is not None:
            self.context[user_id]["last_intent"] = intent

            if intent == "greeting":
                return self._handle_greeting()
            elif intent == "menu_request":
                return self._handle_menu_request()
            elif intent == "cart_request":
                return self._handle_cart_request(user_id)
            elif intent == "order_request":
                return self._handle_order_request(entities, user_id)
            elif intent == "price_request":
                return self._handle_price_request(entities)
            elif intent == "complete_order_request":
                return self._handle_complete_order_request(user_id)
            elif intent == "clear_cart_request":
                return self._handle_clear_cart_request(user_id)
            elif intent == "goodbye":
                return self._handle_goodbye()
        else:
            response = self._generate_response(text)

            if response is not None:
                return self._handle_generated_answer(response)

        return self._handle_unknown()

    def get_user_sentiment(self, user_id):
        """Получение настроения пользователя"""

        return self.context[user_id]["sentiment"]
