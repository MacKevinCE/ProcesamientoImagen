**Nombre** Hermes Mac Kevin Cabanillas Encarnación\
\
**Número Usp:** 10659953\
\
**Área del proyecto:** Inpainting\
\
**Contexto del Problema:** Desde sus primeros inicios, la fotografía ha tenido la función de conservar la memoria histórica, tanto en los grandes acontecimientos como en los, no menos importantes, eventos familiares. Desde nacimientos hasta recordatorios en funerales, la fotografía nos muestra esos seres queridos en distintas épocas de su vida. Y muchos de estos recuerdos han sufrido accidentes, maltratados por el tiempo y nos hemos reusados a perder dichos recuerdos de mucho valor para nosostros.\
\
**Objetivo del proyecto:** Digitalización y Restauración de imágenes Familiares antiguas.
<img src="https://raw.githubusercontent.com/MacKevinCE/procesamiento-imagen/master/Restauracion-foto-antigua-810x540.jpg" alt="Drawing" style="display: block;width: 25rem;margin: 2rem auto 0 auto;"/>
<p style="display: block;text-align: center;;width: 25rem;margin: auto;">A la izquiera imagen digitalizada original y a la derecha imagen restaurada y mejorada.</p>

\
**Etapas :**
* Digitalización de las imágenes
* Restauración
  * Por detección de bordes: Si la imagen presenta pérdida de información en ciertas regiones (partes recortadas de la imagen), podemos determinar las regiones dañadas rastreando los bordes que estos definen, para ello se aplica algún filtro de detección de bordes como el filtro Canny.\
Una vez encontrados los bordes podemos intentar recuperar la parte cortada con un proceso de difusión del color, en este caso, con la ecuación de difusión del calor.\
La restauración de la imagen es sensible al método de rastreo de bordes y por supuesto a la técnica de difusión aplicada. Esto es solo un avance inicial ya que los daños en la imagen pueden presentar una geometría mucho más complicada, en particular los deterioros ocasionados por el paso del tiempo.
  * Mediante Pirámide Gaussiana: siguiendo el siguiente algoritmo posible:
     1. Mientras haya píxeles desconocidos hacer
        1. Interpolar en cuatro direcciones los píxeles desconocidos.
        2. Reducir la imagen mediante operadores de agregación.
     2. Mientras no esté en el nivel 1 de la pirámide hacer.
        1. Ampliar la imagen mediante información intervalar.
        2. Sustituir los píxeles desconocidos por los píxeles de la imagen ampliada.
* Mejoramiento de las fotos
   * Ecualización del histograma: Es una transformación que pretende obtener un histograma con una distribución uniforme. Es decir, que exista el mismo número de pixels para cada nivel de gris del histograma.\
El resultado de la ecualización maximiza el contraste de una imagen sin perder información de tipo estructural.
  *  Filtros de suavizado: Utilizados principalmente para la reducción de ruido.\
El centro es el más importante y otros píxeles son
inversamente ponderado (promediados) en función de
su distancia desde el centro de la máscara.
  * Filtro Median: Reemplaza el valor de un pixel por la media de lo valores de grises en el vecindario de un pixel.\
Utilizada para reducir ruidos al azar de una imagen.
  * Filtros para definir Bordes y Gradientes :Se utiliza para acentuar detalles y bordes de una imagen.
  * Filtro de Nitidez (sharpining/unsharping):
     * Sharping: Mejora la nitidez de los detalles de las
imágenes borrosas, mediante la eliminación de baja frecuencia de la información espacial de la imagen original.
     * Unsharping: Genera una imagen borrosa a partir de una imagen nítida (contrario a sharping)
* Metrica de evaluacion : La finalidad de este trabajo es de restaurar imágenes donde parte de su información se ha perdido. Por lo tanto, para poder medir si las modificaciones introducidas en el algoritmo aportan una mejora al mismo, es necesario el uso de una medida para comparar el resultado obtenido con la imagen original. Esto permitirá poder comparar y analizar los resultados obtenidos.\
Se ha decidido utilizar el error cuadrático medio (en inglés Mean Square Error, en adelante MSE) como medida para comparar dos imágenes. El MSE de una imagen consigo misma es 0, por lo tanto, cuanto menor sea el valor obtenido, mejor será la aproximación a la imagen original.
