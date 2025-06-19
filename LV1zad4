imeDatoteke = input("Upišite ime tekstualne datoteke: ")

try:
    f = open(imeDatoteke)
    zbrojPouzdanosti = 0.0
    brojPojava = 0

    for linija in f:
        linija = linija.strip()
        if linija.startswith("X-DSPAM-Confidence:"):
            for element in linija.split():
                try:
                    vrijednost = float(element)
                    zbrojPouzdanosti += vrijednost
                    brojPojava += 1
                    break
                except ValueError:
                    continue
    f.close()

    if brojPojava > 0:
        prosjek = zbrojPouzdanosti / brojPojava
        print("Prosječna pouzdanost je:", prosjek)
    else:
        print("Nisu pronađene odgovarajuće vrijednosti.")
except:
    print("Greška: datoteka nije pronađena.")
