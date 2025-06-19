fhand = open('song.txt', encoding='utf-8')

frekvencija = {}

for linija in fhand:
    linija = linija.rstrip()

    for rijec in linija.split():
        rijec = rijec.lower()

        if rijec in frekvencija:
            frekvencija[rijec] += 1
        else:
            frekvencija[rijec] = 1

fhand.close()

for rijec in frekvencija:
    ponavljanje = frekvencija[rijec]
    if ponavljanje == 1:
        print(rijec)
