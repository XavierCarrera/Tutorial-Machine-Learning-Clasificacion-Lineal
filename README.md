# Tutorial Machine Learning: Clasificación Linear

## Introducción

Uno de los problemas con los datos es que no siempre hay una continuidad entre ellos. En estos casos nos referimos a un problema que no se puede explicar cuantitativamente sino cualitativamente. Es decir, los problemas de clasificación. Lidiar con ellos usando regresión linear es complicado, por lo que es necesario ocupar otras técnicas.

El propósito de Machine Learning es predecir valores desconocidos a partir de datos histórico. Imaginemos que tenemos un problema en el que tenemos datos de conejos, perros y pollos, con los cuales queremos entrenar a nuestro modelo para predecir animales por nosotros. Una regresión linear no sirve en estas circunstancias porque los valores son **categóricos** (cualitativos) en vez de numéricos. 

## Matriz de Indicadores

Para el problema anterior, podríamos proponer dos soluciones: 

1. Poner etiquetas *x1*, *x2* y *x3* para identificar a cada uno de los animales.
2. Podríamos crear una matriz con valores booleanos para identificar a los animales. Ergo (1,0,0), (0,1,0) y (0,0,1).

En este caso, el enfoque clásico de machine learning es utilizar valores booleanos porque no hay ninguna relación ordinal entre los animales. A esto también se le conoce como *hot encoding*. Esto hace más sencillo manipular e interpretar datos. Con esto podemos armar vectores que conformarán nuestra matriz. Si tuvieramos un conjunto de animales que fuera perro, pollo, conejo, pollo, nuestra matriz se vería de la siguiente forma:

    [0  1  0]
    [0  0  1]
    [1  0  0]
    [0  0  1]
    
Ahora tenemos una buena forma de representar clasificaciones, por lo que podemos hacer predicciones con variables predictoras.

El objetivo de un algoritmo de clasificación es encontrar la clase o etiqueta a la que pertence un data point. Podemos hacer esto estimando la pobabilidad de que dicho data point pertenezca a dicha clase y tomando la calse con el mayor valor. Para esto, necesitamos una función que resulte en un valor que cumpla con los criterios de clase y otra que resulte en valor cercano a cero representando que cae fuera de estos criterios. Es decir, decide si un data point se ajusta a cierta clase. Esta función, curiosamente, puede ser construida con una regresión linear. 

Empecemos creando una variable ficticia *y* para esta clase con cada punto en nuestro data set. La variable *y* representa si un punto representa a la calse. Se le asigna 1 si pertenece a la clase y 0 si no pertenece. Dicha función *y* sería equivalente a un variable a predecir y es lo que deberíamos predecir en una línea de mejor ajuste. Esto se vería así:

![Linea Regresion Clasificación](https://ds055uzetaobb.cloudfront.net/brioche/uploads/xMXsqZIQxz-screen-shot-2019-01-07-at-95316-pm.png?width=1200)

Nos sirve utilizar la misma fórmula para regresión linear:

![reg linear1]()

Sin olvidar el orden de los animales (conejos, perros, pollos), supongamos que queremos saber si un data point nuevo es un perro o no. Suponiendo que tenemos dos datos sobre los animales (tamaño en cms y peso en g), la matriz *A* con datos de entrenamiento con tres animales diferentes se vería así:

    [50    2500]
    [120  15000]
    [35    8000]
   
El vector *->b* para identificar a un perro se vería así:

    [0]
    [0]
    [1]
    
Finalmente tendríamos que comparar el nivel en que cada data point se ajusta a varias clases. En este caso, integraríamos a *x1* que representa el tamaño y *x2* que representa el peso. A partir de esto, podríamos investigar sobre los datos de todos los conejos en la región y diseñar una función que los decribe de la siguiente manera:

*f*Perro(*->x*) = 0.02*x2* + 3*x3* - 25

Si tuvieramos un nuevo animal que midiera 130 cms y pesara 17000 gramos, notaríamos fácilmente que lo podríamos clasificar como un perro dado que:

*f*Perro(Animal nuevo) = 0.02 · 17000 + 3 · 390 - 25 = 1485

lo cual es un valor por encima del valor que podría representar los datos de un conejo (por ejemplo, 45 cms y 2300 gramos) o un pollo (por ejemplo, 55 cms y 12000).

Aunque las funciones son útiles para clasificar, hay que tomar en cuenta que no son la mejor manera de medir probabilidades. 
