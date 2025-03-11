def total_euro(radniSati, placaSat):
    ukupno = radniSati * placaSat
    return ukupno


radniSati=float(input('Unesite broj radnih sati:'))
placaSat=float(input('Unesite placu po radnom satu:'))

print('Radni sati: ', radniSati)
print('Placa po satu: ', placaSat)
print('Ukupan iznos je: ', total_euro(radniSati, placaSat))
