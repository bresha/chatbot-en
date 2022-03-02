import json
import spacy
from spacy.matcher import Matcher
import re
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer


def load_json(filename):
    with open(filename) as f:
        content = json.load(f)
        return content

def create_vectorizer_data():
    examples = []
    for item in aff_neg_data:
        for value in item["examples"]:
            examples.append(value)
    return examples

stories = load_json("data/stories.json")
menu = load_json("data/menu.json")
foods = [food["meal"] for food in menu]

aff_neg_data = load_json("data/aff_neg_data.json")

nlp = spacy.load("en_core_web_md")
matcher = Matcher(nlp.vocab)
    
vectorizer = TfidfVectorizer()
vectorizer.fit(create_vectorizer_data())

aff_neg_clf = load("models/aff_neg_clf.joblib")

def create_food_patterns():
    patterns = []
    for food in foods:
        patterns.append([{"IS_DIGIT": True}, {"LEMMA": food.lower()}])
    return patterns

def parse_order_input(order=None):
    while True:
        if order is None:
            order = input(">>")
        order_doc = nlp(order.lower())
        patterns = create_food_patterns()
        matcher.add("FOODS_PATTERNS", patterns)
        matches = matcher(order_doc)
        if len(matches) > 0:
            orders = []
            for match_id, start, end in matches:
                order = order_doc[start:end]
                orders.append(order)
            return orders
        else:
            print("Invalid input. Please enter a valid order")
            order = None

def parse_anything_else():
    while True:
        answer = input(">>")
        vector = vectorizer.transform([answer])
        result = aff_neg_clf.predict(vector.toarray())
        
        if result[0] == 0:
            return []
        else:
            return parse_order_input(answer)

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
                order["foods"].append(orders)
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