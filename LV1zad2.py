
try:
    ocjena = float(input("Unesite ocjenu (0.0 - 1.0): "))

    if 0.0 <= ocjena <= 1.0:
        if ocjena >= 0.9:
            print("A")
        elif ocjena >= 0.8:
            print("B")
        elif ocjena >= 0.7:
            print("C")
        elif ocjena >= 0.6:
            print("D")
        else:
            print("F")
    else:
        print("Ocjena mora biti u rasponu između 0.0 i 1.0.")

except ValueError:
    print("Neispravan unos. Molimo unesite broj.")
