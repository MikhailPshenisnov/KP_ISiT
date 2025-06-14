import json

from nlp_functions import clean_text, lemmatize_text, correct_text


def get_emo_dict(file_path):
    """Считывание словаря для определения тональности текста"""
    emo_dict = dict()
    with open(file_path, "r", encoding="utf-8") as file:
        is_first_line_flag = True
        for line in file.readlines():
            if is_first_line_flag:
                is_first_line_flag = False
                continue
            term, _, value, _, _, _, _, _ = line.strip().split(';')
            emo_dict[term] = float(value)
    return emo_dict


def get_dialogues(file_path):
    """Считывание датасета диалогов и приведение его в корректный формат"""
    with open(file_path, "r", encoding="utf8") as file:
        content = file.read()

    raw_dialogues = content.split('\n\n')
    dialogues = [raw_dialogue.split('\n')[:2] for raw_dialogue in raw_dialogues]

    filtered_dialogues = []
    questions = set()

    for dialogue in dialogues:
        if len(dialogue) != 2:
            continue

        question, answer = dialogue
        question = lemmatize_text(correct_text(clean_text(question[2:])))
        answer = answer[2:]

        if question != '' and question not in questions:
            questions.add(question)
            filtered_dialogues.append([question, answer])

    structured_dialogues = {}

    for question, answer in filtered_dialogues:
        words = set(question.split(' '))
        for word in words:
            if word not in structured_dialogues:
                structured_dialogues[word] = []
            structured_dialogues[word].append([question, answer])

    structured_dialogues_cut = {}
    for word, pairs in structured_dialogues.items():
        pairs.sort(key=lambda pair: len(pair[0]))
        structured_dialogues_cut[word] = pairs[:1000]

    return structured_dialogues_cut


def get_menu(file_path):
    """Считывание данных из файла с меню"""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def get_intent_dataset(file_path, menu):
    """Считывание данных из файла с датасетом намерений, а также расширение датасета при помощи меню и шаблонов"""
    menu_dishes = list(menu.keys())

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for intent in list(data["intents"].keys()):
        examples = []

        for example in data["intents"][intent]["examples"]:
            if "<DISH>" in example:
                for dish in menu_dishes:
                    examples.append(example.replace("<DISH>", dish))
            else:
                examples.append(example)

        data["intents"][intent]["examples"] = examples

    return data
