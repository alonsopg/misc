# -*- coding: utf-8 -*-

from nltk import word_tokenize
from nltk.stem import SnowballStemmer

opinions = ["Este es un post de juguetes de aprendizaje \
automático. En realidad, contiene no mucho \
material interesante.",
"Las bases de datos de imágenes proporcionan \
capacidades de almacenamiento.",
"La mayoría de las bases de datos de imágenes \
imágenes seguras de forma permanente.",
"Los datos de imagen de tienda bases de datos.",
"Imagina almacenar bases de datos de bases de \
datos de imágenes. Almacenar datos. Bases de datos \
de imágenes de datos de la tienda."]

stemmer = SnowballStemmer('spanish')

print stemmer.stem('cuando')
print stemmer.stem('apprenderla')

text = 'En su parte de arriba encontramos la ";zona de mandos";,' \
       ' donde se puede echar el detergente, aunque en nuestro ' \
       'caso lo al ser gel lo ponemos directamente junto con' \
       ' la ropa.'


stemmed_text = [stemmer.stem(i) for i in word_tokenize(opinions)]
print stemmed_text