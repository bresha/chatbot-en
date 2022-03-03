import json
import spacy
from spacy.matcher import Matcher
import re
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import numerizer
import numpy as np

def load_json(filename):
    with open(filename) as f:
        content = json.load(f)
        return content


foods_labels = load_json("data/foods_labels.json")
stories = load_json("data/stories.json")
menu = load_json("data/menu.json")
foods = [food["meal"] for food in menu]

aff_neg_data = load_json("data/aff_neg_data.json")

nlp = spacy.load("en_core_web_md")
matcher = Matcher(nlp.vocab)

aff_neg_classifier = load("models/aff_neg_pipe.plk")

foods_classifier = load("models/foods_grid_mnb.pkl")


def parse_order_seg(order_seg):
    number_of_items = 0
    item = ""

    clf_prob = foods_classifier.predict_proba([order_seg])
    idx = np.argmax(clf_prob)
    if clf_prob[0][idx] > 0.13:
        item = foods_labels[str(idx)]

        order_doc = nlp(order_seg)
        numbers = order_doc._.numerize()
        if len(numbers.keys()) > 1:
            raise Exception("To many numbers found")
        elif len(numbers.keys()) == 0:
            number_of_items = 1
        else:
            number_of_items = int(list(numbers.values())[0])

        return {item: number_of_items}

    return {}

def parse_order_input(order=None):
    while True:
        if order is None:
            order = input(">>")
        order_split_coma = order.lower().split(",")
        order_split_and = []
        for order_split in order_split_coma:
            order_split_and += order_split.split(" and ")
        orders = []
        for order_seg in order_split_and:
            order_from_seg = parse_order_seg(order_seg)
            if bool(order_from_seg):
                orders.append(order_from_seg)
        if len(orders) > 0:
            return orders
        else:
            print("Invalid input. Please enter a valid order")
            order = None

def parse_anything_else():
    additional_orders = []
    while True:
        answer = input(">>")
        result = aff_neg_classifier.predict([answer])
        
        if result[0] == 0:
            return additional_orders
        else:
            additional_orders.append(parse_order_input(answer))
            print("Anything else?")

def parse_delivery_address():
    return input(">>")

def parse_name():
    while True:
        full_name = input(">>")
        full_name_doc = nlp(full_name)
        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
        matcher.add('FULL_NAME', [pattern])
        matches = matcher(full_name_doc)
        if len(matches) == 1:
            match_id, start, end = matches[0]
            span = full_name_doc[start:end]
            return span.text
        else:
            print("Cannot recognize full name. Please enter valid full name!")

def parse_email():
    while True:
        email = input(">>")
        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if(re.fullmatch(regex, email)):
            return email
        else:
            print("Invalid email. Please enter a valid email address.")

def parse_telephone():
    while True:
        telephone = input(">>")
        # https://regex101.com/r/NcxUp9/1
        regex = r'(\+\d{2,3} ?)?(\d{2,3}){1} ?(\d{3,4}) ?(\d{3,4})'
        if re.fullmatch(regex, telephone):
            return telephone
        else:
            print("Invalid telephone number. Please enter a valid telephone number")

def parse_number_of_persons():
    while True:
        number_of_persons = input(">>")
        number_of_persons_doc = nlp(number_of_persons)
        values = [token for token in span.subtree if token.like_num]
        if len(values) == 1:
            return values[0]
        else:
            print("Invalid input. Please specify number of persons!")

def parse_date_and_time():
    while True:
        date_and_time = input(">>")
        # https://regex101.com/r/nSnavm/1
        regex = r'[a-z]+ ?\d+([a-z]{2})?,? \d{4} \d{1,2}:\d{2} ?([a-z]{2})?'
        if re.fullmatch(regex, date_and_time.lower()):
            return date_and_time
        else:
            print("Invalid date and time. Please enter valid date and time")


def parse_reservation_meals():
    # fix this
    while True:
        answer = input(">>")
        answer_doc = nlp(answer)
        negation_tokens = [tok for tok in answer_doc if tok.dep_ == 'neg']
        if len(negation_tokens) > 1:
            return []
        else:
            return parse_order_input(answer)

def order_food_menu(stories):
    story = stories["order_menu_path"]
    order = {}
    order["foods"] = []
    for item in story:
        item_keys = list(item.keys())
        print(item[item_keys[0]])
        if item_keys[0] == "order_question":
            order["foods"] = parse_order_input()
        elif item_keys[0] == "anything_else":
            orders = parse_anything_else()
            if len(orders) > 0:
                order["foods"] += orders
        elif item_keys[0] == "delivery_address":
            order["address"] = parse_delivery_address()
        elif item_keys[0] == "name":
            order["name"] = parse_name()
        elif item_keys[0] == "email":
            order["email"] = parse_email()
        elif item_keys[0] == "telephone":
            order["telephone"] = parse_telephone()
    return order
        
def book_table_menu(stories):
    story = stories["book_table_path"]
    reservation = {}
    for item in story:
        item_keys = list(item.keys())
        print(item[item_keys[0]])
        if item_keys[0] == "number_of_persons":
            reservation["number_of_persons"] = parse_number_of_persons()
        elif item_keys[0] == "date_and_time":
            reservation["date_and_time"] = parse_date_and_time()
        elif item_keys[0] == "name":
            reservation["name"] = parse_name()
        elif item_keys[0] == "order_meals":
            reservation["reservation_meals"] = parse_reservation_meals()
    return reservation

def main_menu(stories):
    print(stories["starting_menu"]["greet"])
    while True:
        print("1) ", stories["starting_menu"]["options"][0])
        print("2) ", stories["starting_menu"]["options"][1])
        answer = input(">>>")
        if answer == "1" or answer == "2":
            return answer
        else:
            print("Unknown entry.")


def main():
    mm_answer = main_menu(stories)
    order = {}
    reservation = {}
    if mm_answer == "1":
        order = order_food_menu(stories)
    elif mm_answer == "2":
        reservation = book_table_menu(stories)
    else:
        raise Exception("Not valid input for main menu")
    print(order)
    print(reservation)


if __name__ == "__main__":
    main()