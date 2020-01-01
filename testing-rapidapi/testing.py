import requests

url = "https://currency-exchange.p.rapidapi.com/exchange"

querystring = {"q":"1.0","from":"SGD","to":"MYR"}

# headers = {
#     'x-rapidapi-host': "currency-exchange.p.rapidapi.com",
#     'x-rapidapi-key': "921d086fc6msh6478ca20853758ap12d2afjsncfba5f390914"
#     }

response = requests.request("GET", url,  params=querystring)

print(response.text)