import json
import re

from spellchecker import SpellChecker
import pymorphy2
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, Doc
from yargy import Parser
from yargy.pipelines import morph_pipeline
from yargy.interpretation import fact

from config import MENU_FILE_PATH

# Инициализация NLP инструментов
embedding = NewsEmbedding()
morph_analyzer = pymorphy2.MorphAnalyzer()
morph_vocab = MorphVocab()
segmenter = Segmenter()
spell_checker = SpellChecker(language='ru')
morph_tagger = NewsMorphTagger(embedding)
ner_tagger = NewsNERTagger(embedding)
syntax_parser = NewsSyntaxParser(embedding)

# Подготовка для определения сущностей
with open(MENU_FILE_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)
MENU_DISHES = list(data.keys())
menu_item = fact('MenuItem', ['name'])
menu_rule = morph_pipeline(MENU_DISHES).interpretation(menu_item.name).interpretation(menu_item)
menu_parser = Parser(menu_rule)


def clean_text(text):
    """Очистка текста от лишних символов и приведение к нижнему регистру"""
    lowered_text = text.lower()
    stripped_text = lowered_text.strip()
    symbol_filtered_text = re.sub(r'[^а-яёa-z0-9-\s]', '', stripped_text)
    space_filtered_text = re.sub(r'\s+', ' ', symbol_filtered_text)
    cleaned_text = space_filtered_text.strip()
    return cleaned_text


def correct_text(text):
    """Исправление опечаток в тексте"""
    words = text.split()
    corrected_words = [spell_checker.correction(word) if word in spell_checker else word for word in words]
    corrected_text = ' '.join(corrected_words)
    return corrected_text


def lemmatize_text(text):
    """Лемматизация текста"""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return ' '.join([token.lemma for token in doc.tokens])


def extract_entities(text):
    """Извлечение сущностей"""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    entities = []
    for span in doc.spans:
        span.normalize(morph_vocab)
        entities.append({
            'type': span.type,
            'text': span.text,
            'normal': span.normal
        })
    for match in menu_parser.findall(text):
        entities.append({
            'type': 'MENU_ITEM',
            'text': match.fact.name,
            'normal': match.fact.name.lower()
        })
    return entities


def analyze_sentiment(text, emo_dict):
    """Анализ тональности текста"""
    words = text.split()
    sentiment_score = 0
    matched_words = 0
    for word in words:
        lemma = lemmatize_text(word)
        if lemma in emo_dict:
            sentiment_score += emo_dict[lemma]
            matched_words += 1
    if matched_words > 0:
        return sentiment_score / matched_words
    return 0
