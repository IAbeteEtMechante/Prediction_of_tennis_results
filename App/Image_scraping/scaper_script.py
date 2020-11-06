

import requests, bs4

CITY = "paris"
FILE = CITY + ".txt"

f = open(FILE, "x");
f.close()


res = requests.get('https://' + CITY +'.startups-list.com/')
# res = requests.get('https://quebecstartups.com/')
res.raise_for_status()
startupSoup = bs4.BeautifulSoup(res.text, 'html.parser')


elems = startupSoup.select('p')

f = open(FILE, "a")
for el in elems:
    if el.getText().strip() == "":
        continue
    if "We search the best weekend and long-haul flight deals" in el.getText().strip():
        continue
    bos_token = '<BOS>'
    eos_token = '<EOS>'
    data = bos_token + ' ' + el.getText().strip() + ' ' + eos_token + '\n'
    f.write(data)
f.close()
