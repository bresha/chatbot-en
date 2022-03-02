import requests
from bs4 import BeautifulSoup
import json

link = "https://www.w3schools.com/w3css/w3css_web_tmp_pizza.asp"

r = requests.get(link)

content = r.text

soup = BeautifulSoup(content, 'html.parser')

menu = soup.find('div', id='menu')

meals = []
ingredients = []
prices = []
for meal in menu.find_all('b'):
    meals.append(meal.text)
for ingredient in menu.find_all('p'):
    ingredients.append(ingredient.text)
for price in menu.find_all('span', class_='w3-right'):
    prices.append(price.text)


menu = []
for meal, ingredient, price in zip(meals, ingredients, prices):
    menu.append({'meal': meal, 'ingredient': ingredient, 'price': price})

data_string = json.dumps(menu)

with open('menu.json', 'w') as f:
    f.write(data_string)