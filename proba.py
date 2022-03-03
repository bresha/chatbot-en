text = "I would like ravioli, chicken soup and meat town"

text_split_comma = text.lower().split(',')

text_split_and = []
for order in text_split_comma:
    text_split_and += order.split(' and ')

print(text_split_and)

