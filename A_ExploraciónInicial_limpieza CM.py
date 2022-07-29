###############################################################################
###########################Importación de paquetes#############################
###############################################################################
import numpy as np ###manejo de arrays
import pandas as pd ### para manejo de datos
import sqlite3 as sql ### conexión con SQL
import plotly.graph_objs as go ### gráficos
import funciones as fn ### para funciones
from mlxtend.preprocessing import TransactionEncoder #para separar los generos
from datetime import datetime # para el cambio de formato a fecha
from surprise import Reader 

import os  ### para ver y cambiar directorio de trabajo
os.getcwd()
#os.chdir('c:\\Users\\jhans\\OneDrive\\Desktop\\Marketing')

###### para ejecutar sql y conectarse a bd #######

conn=sql.connect('db_movies')
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

### para verificar las tablas que hay disponibles
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())


ratings = pd.read_sql('select * from ratings', conn)
movies = pd.read_sql('select * from movies', conn)

###############################################################################
#############################Exploración inicial###############################
###############################################################################

### Identificar campos de cruce y verificar que estén en mismo formato ####
### verificar duplicados

movies.info() #no hay nulos, las variables tienen asignado un tipo de variable correcto.
movies.head()
movies.duplicated().sum() # no hay registros duplicados.

ratings.info() #no hay nulos, las variables tienen asignado un correcto tipo de variable, a excepción de la variable "timestamp". 
ratings.head()
ratings.duplicated().sum()  # no hay registros duplicados
 
##### Descripción base de ratings


cr=pd.read_sql(""" select 
                          "rating", 
                          count(*) as conteo 
                          from ratings 
                          group by "rating"
                          """, conn)


data  = go.Bar( x=cr.rating,y=cr.conteo, text=cr.conteo, textposition="outside")
Layout=go.Layout(title="Count of ratings",xaxis={'title':'Rating'},yaxis={'title':'Count'})
go.Figure(data,Layout)
## El anterior gráfico de barras nos muestra el conteo de calificaciones, donde 
## las puntajes de 4 y 3 respectivamente, son las dos categorías de puntuaciones que mayor participación tiene, por otro lado
## los puntajes con menos asignación son los de 0.5, 1.5 y 1, respectivamente.

rating_users=pd.read_sql(''' select "UserId" as user_id,
                         count(*) as cnt_rat
                         from ratings
                         group by "UserId"
                         order by cnt_rat desc
                         ''',conn )


data  = go.Scatter(x = rating_users.index, y= rating_users.cnt_rat)
Layout= go.Layout(title="Ratings given per user",xaxis={'title':'User Count'}, yaxis={'title':'Ratings'})
go.Figure(data, Layout) 
## Del anterior gráfico se puede extraer la catidad de calificaciones que han dado los usuario,
##por lo que, se puede observar la persona que más películas ha calificado, lo ha hecho a 2698 producciones cinematográficas,
## este número va disminuyendo hasta llegar a usuarios que han hecho calificaciones de 20, siendo este el mínimo número de calificaciones por usuario; 
##de este modo, no se hace necesario aplicar algún filtro con relación a estas variables, debido a que con un mínimo de 20, se pueden obtener resultados confiables. 

rating_movies=pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         order by cnt_rat desc
                         ''',conn )

data  = go.Scatter(x = rating_movies.index, y= rating_movies.cnt_rat)
Layout= go.Layout(title="Ratings received per movie",xaxis={'title':'movie Count'}, yaxis={'title':'Ratings'})
go.Figure(data, Layout)
## Del anterior gráfico se observa el número de calificaciones dado a cada película, donde se encuentra que la película que más han calificado, le han asignado puntuaciones 329 veces,
##por otro lado, existen películas que solo la han calificado hasta solo una vez,
## por lo que, es necesario aplicar un filtro y dejar únicamente las películas que tengan un número de calificaciones significativos para el estudio. 

###############################################################################
#########################Limpieza y transformaciones###########################
###############################################################################

##En SQL se filtran las películas que no tiene más de 10 calificaciones.
fn.ejecutar_sql('preprocesamiento cm.sql', cur)

pd.read_sql('select count(*) ratings_final', conn)
ratings =pd.read_sql('select * from  ratings_final',conn)

pd.read_sql('select count(*) movies_final', conn)
movies =pd.read_sql('select * from  movies_final',conn)


######para separar géneros en base de datos
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)
genres  = genres.astype(int)
movies = movies.drop(['genres'], axis = 1) 
movies =pd.concat([movies, genres],axis=1)
movies = movies.drop(['movieId:1'], axis = 1) 


## Creación del la variable "premiere_year", la cual se extrae del título de la película, siendo esta el año de estreno de la producción. 
movies['premiere_year'] = movies['title'].str[-5:-1]

## Asignación correcta del tipo de dato de la variable "timestamp" y el renombre de la misma.
lista = ratings["timestamp"]
lista2 = []
for i in lista:
    lista2.append ((datetime.fromtimestamp(i)).strftime("%d-%m-%Y")) #Mediante este ciclo se convierte de timestamp a datatime.

ratings = ratings.drop(['timestamp'], axis = 1) #Se elimina la variable "timestamp".
df = pd.DataFrame(lista2) #Se crea el DataFrame con la columna con la fecha modificada.

ratings =pd.concat([ratings, df],axis=1) #Se une la base ratings con la columna con la fecha modificada.
ratings = ratings.rename(columns={0: 'view_time'}).reset_index() #Se cambia el nombre de la varible por "view_time", siendo esta la fecha en que fue vista la película.
ratings = ratings.drop(['index'], axis = 1) # Eliminación de la variable "index", que se agregó despues de hacer la concatenación.
ratings["view_time"]=pd.to_datetime(ratings['view_time']) #Se le vuelve asignar el formato de DateTime. 

##Se comprueba que no haya nulos, datos duplicados y además que exista una asignación correcta de tipo de dato. 
movies.info() #no hay nulos, las variables tienen asignado un tipo de variable correcto.
movies.duplicated().sum() # no hay registros duplicados.

ratings.info() #no hay nulos, las variables tienen asignado un correcto tipo de variable, a excepción de la variable "timestamp". 
ratings.duplicated().sum()  # no hay registros duplicados

movies = movies.rename(columns={'cnt_rat': 'amountRat'}) #se cambia el nombre de la variable "cnt_rat".


movies.to_sql('movie2',conn)
movies=pd.read_sql('select* from movie2',conn).drop(['index'], axis = 1) 


ratings.to_sql('ratings2',conn)
ratings=pd.read_sql('select* from ratings2',conn).drop(['index'], axis = 1) 

