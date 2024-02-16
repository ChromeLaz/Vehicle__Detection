import cv2

# Carica il classificatore Haar
car_cascade = cv2.CascadeClassifier('cars.xml')

# Leggi l'immagine
img = cv2.imread('machine_test.jpg')

# Converti in scala di grigi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Rileva i veicoli
cars = car_cascade.detectMultiScale(gray, 1.3, 1)

# Disegna il rettangolo attorno a ogni veicolo rilevato
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Mostra l'immagine
cv2.imwrite('output.jpg', img)
cv2.waitKey()
