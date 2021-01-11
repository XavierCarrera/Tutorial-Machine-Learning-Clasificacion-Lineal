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

Aunque las funciones son útiles para clasificar, hay que tomar en cuenta que no son la mejor manera de medir probabilidades. Al ser una función linear, es capaz de resultar en valores mayores a uno y menores a cero. Esto no nos sirve si estamos midiendo probabilidades. Al presentar el problema de esta forma, podemos producir valores negativos o que son demasiado grandes. 

## Clasificación Logística

Transformar una función de clasificación linear en una función de clasificación sigmoidal es muy útil cuando estamos trabajando con probabilidades. 

![Sigmoide](https://lucashomil.github.io/datascience/images/sigmoid-function-sized.png) 

Sin embargo, existe una forma de calcula una función sigmoidal de manera más directa. A esta se le conoce como la clasificación logística. Supopngamos que tenemos *n* cantidad de variables *x1, x2,... xn* usando una clasificación logística que calcula pesos *m1, m2, ... mn* y un sesgo en donde el sigmoide σ (*m1x1 + m2x2 + ... mn xn + b*) descibe bastante bien a nuestros datos.

Un ejemplo sería el siguiente:

![Linea sigmoide](https://ds055uzetaobb.cloudfront.net/brioche/uploads/d5R21cBp7B-2-2-1actual.png?width=1200)

Acá hay una variable predictora y cada punto tiene un valor 1 cuando es cierto o 0 cuando es nulo. La sigmoide está diseñada para maximizar la probabilidad de clasificar correctamente entre dos clases. Esto es diferente a una clasificación con una matriz de indicadores, ya que hay una sola función para dos clases y no una función por cada clase.

Supongamos que queremos saber la probabilidad de que un alumno con promedio GPA de 3.2 (*x1*) y con 15 años de edad (*x2*) podría pasar un examen, dado que los pesos de cada valor es de *m1* = 5, *m2* = 1 y el sesgo *b* = -29.

Utilizando la función de la sigmoide podemos ver que:

    *f*(3.2, 15) = 5 ⋅ 3.2 + 1 ⋅ 15 - 29 = 2
    
    σ(f(3.2,15)) = σ(2)
    
                 = e² / 1 + e² 
                                
                 = 0.88
                 
Con la regresión logística creamos una función sigmoidal que describe nuestros datos. A esto tendríamos que agregar la propiedad de maximización. 

Supongamos que tenemos dos clases: una positiva y una negativa. Para generar una función de mejor ajuste tenemos que analizar un conjunto de data points, los cuales sabemos que pertenecen a una de las dos clases. La función final (*p ->x*) nos dará la posibilidad de un *-> x* positivo.

Hay que tomar en cuenta que no todas las funciones de probabilidad son iguales. Hay que diferenciar entre calcular las probabilidades de nuestro data set dadas las clasificaciones dado que una función de probabilidad *p(-> x)* sea correcta. Esto significa usar *p(-> x)* para calcular la probabilidad de que cada data point esté en una clase y luego tomar los productos de los resultados.

Si asignamos a cada punto *-> xi* una variable *yi* que esta puesta a 1 si *-> x* es positiva y 0 si es negativa. Expresado matemáticamente se vería así:

![funcion sigmoide](asdsa.as)

Esta cantidad, la probabilidad de que todos los puntos conocidos tengan una clase es lo que llamamos maximización en algoritmos de clasificación logística. El proceso es conocido como el **método de máxima verosimilitud** porque encontramos la similitud más posible en una función sigmoide.

Dado que la máxima verosimilitud es resultado de probabilidades, el techo siempre será 1 que significará una similitud exacta. Todas las probabilidades serán resultado de una función logística. Sin embargo, las probabilidades nunca serán 1 o 0. 

Por ejemplo, si tenemos eventos independientes respecto a las probabilidades de sacar una bola roja entre bolas azules. Podríamos calcular las probabilidades  tomando dos mundos en (3,2) y (-5,3). La probabilidad de sacar la bola roja es simplemente: 1 - *p(x,y)*, lo que es igual a:

    P = p(3, 2)(1 - p(-5,3))
    
      =     e⁶/1 + e⁶ ⋅ (1 - e⁻1/ 1 + e⁻1)
      
      = 0.73
      
Hasta este momento hemos estimados las probabilidades de una regresión logística, más no como hacer una clasificación. Las clasificaciones son usualmente hechas poniendo un límite de probabilidades para dividir dos clases.

Por ejemplo, si clasificamos entre rojo y azul y los límites de probabilidad so *p* = 0.8. Los únicos puntos sobre el 80% serán clasificados como rojo. Por lo general los limites de probabilidad están basados en las necesidades de la situación. En situaciones reales, el límite puede ser bajo si el resultado no tiene grandes implicaciones. Pero si fuera el caso, habría que levantarlo por precaución (imagina si tuvieramos que determinar si una persona tiene una enfermedad seria o no).

Alternativamete, podríamos tomar una función logística y tranformarla en logarítmica. Tomando el logaritmo de una regresión logística nos da una función linear que matemáticamente se ve así:

![mate logarítmica](asdas.as)

Esta función linear se le conoce como logit. Aunque no es buena calculando probabilidades, es útil para comparasiones y optimizaciones. Esta es una alternativa cuando tenemos resultados extraños con una función logística.

## Análisis Discriminante Linear
