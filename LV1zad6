fhand = open('SMSSpamCollection.txt', encoding='utf-8')

zbrojHamRijeci = 0
zbrojSpamRijeci = 0
brojHam = 0
brojSpam = 0
brojUsklicnika = 0

for linija in fhand:
    linija = linija.strip()

    if not linija:
        continue

    if linija.startswith('ham'):
        tip = 'ham'
        sadrzaj = linija[4:].strip()
    elif linija.startswith('spam'):
        tip = 'spam'
        sadrzaj = linija[5:].strip()
    else:
        continue

    duljina = len(sadrzaj.split())

    if tip == 'ham':
        zbrojHamRijeci += duljina
        brojHam += 1
    elif tip == 'spam':
        zbrojSpamRijeci += duljina
        brojSpam += 1

        if sadrzaj.endswith('!'):
            brojUsklicnika += 1

fhand.close()

prosjekHam = zbrojHamRijeci / brojHam if brojHam > 0 else 0
prosjekSpam = zbrojSpamRijeci / brojSpam if brojSpam > 0 else 0

print(prosjekHam)
print(prosjekSpam)
print(brojUsklicnika)
