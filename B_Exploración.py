
###############################################################################
########################### IMPORTACIÓN PAQUETES ##############################
###############################################################################
import numpy as np ###manejo de arrays
import pandas as pd ### para manejo de datos
import sqlite3 as sql ### conexión con SQL
import plotly.graph_objs as go ### gráficos
import funciones as fn ### para funciones
from mlxtend.preprocessing import TransactionEncoder #para separar los generos
from datetime import datetime # para el cambio de formato a fecha
from surprise import Reader 


conn=sql.connect('db_movies')
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

movies=pd.read_sql('select* from movie2',conn)
ratings=pd.read_sql('select* from ratings2',conn)
ratings["view_time"]=pd.to_datetime(ratings['view_time'])

fn.ejecutar_sql('exploración cm.sql', cur)
pd.read_sql('select count(*) rating_movies', conn)
rating_movies =pd.read_sql('select * from  rating_movies',conn)

###############################################################################
############################ EXPLORACIÓN DE DATOS #############################
###############################################################################

###################### Top 10 películas más calificadas ####################### 
cr= pd.read_sql('''select title, 
            count(*) as conteo 
            from rating_movies
            group by title
            order by conteo desc limit 10
            ''', conn)


data  = go.Bar( x=cr.title,y=cr.conteo, text=cr.conteo, textposition="outside")
Layout=go.Layout(title="top 10 rated movies",xaxis={'title':'title'},yaxis={'title':'Count'})
go.Figure(data,Layout)
##Se encuentra que el top 10 de películas más calificadas, donde la película 'Forrest Gump' es la más calificada con 329 calificaciones
##seguida por 'Shawshank Redemption' con 317 y en tercer lugar 'Pulp Fiction' con 307. Cabe mencionar que, todas las películas en este top 
##fueron estrenadas en el siglo XX.


############################ Conteo de los generos ###########################
 

##Los generos más éxitosos de la plataforma de películas son: drama, comedia y acción. Por otro lado,
##los generos con menos participación son: documentales, cine negro y películas del oeste

##################### Número de lanzamientos por año ########################
base = rating_movies.groupby(['premiereYear'])[['movieId']].count().sort_values('premiereYear', ascending= True).reset_index()

data  = go.Scatter(x = base.premiereYear, y= base.movieId)
Layout= go.Layout(title="number of premieres per year",xaxis={'title':'premiereYear'}, yaxis={'title':'count '})
go.Figure(data, Layout) 

##En este gráfico se puede observar registros desde el 1922 hasta el 2018, donde el año con el mayor número de lanzamientos fue el del 1995, 
##con 5602 estrenos cinematográficos, después de lograrse ese pico, comienza a descender hasta tener 
## menos de 500 lanzamientos, esto entendido, no como estrenos en general en el mundo, sino solo 
##las producciones que ingresan a la plataforma.


##################### Número de vistas por año ###############################
base = ratings.groupby(['view_time'])[['movieId']].count().reset_index()
base =base.resample('1Y', on='view_time').sum().reset_index()
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=base.view_time,
    y=base.movieId,
    name = 'behavior of views by year '
))
fig.update_layout(title_text='behavior of views by year ', xaxis={'title':'Year'},yaxis={'title':'Count'}, title_x=0.5, width=1200)
fig.show()

##El gráfico muestra registros desde el 1996, es decir que  apartir de ese año se comenzaron a subir 
##películas en la plataforma, el año con mayor número de vistas fue el 2000 con 7971 vistas, desde la apertura
##se ha tenido mucha variación respecto a las vistas, por lo que no se encuentra algún patrón importante. Adicionalmente,
##se esperaba mayor protagonismo por parte de los últimos años, debido al boom que ha tenido las plataformas de películas. 


################### Número de calificaciones por puntaje #####################
base= pd.read_sql('''select movieRating,
            count(*) as conteo 
            from rating_movies
            group by movieRating
            order by conteo desc
            ''', conn)

fig = go.Figure(data=[go.Table(
    header=dict(values=[['<b>RATING</b>'],
                  ['<b>TOTAL</b>']],
                fill_color='#636EFA',
                align='center',
                font_size=15,
                font_color = 'white',
                height=32),
    cells=dict(values=[base.movieRating, base.conteo],
               fill_color='lavender',
               align='center', 
               font_size=15,
               height=25))
])

fig.update_layout(width=600, height=800)
fig.show()

#El puntaje que más calificaciones recibió fue 4, lo que quiere decir que en general los usuarios
#se sienten satisfechos con el contenido que ven en la plataforma online. 

################### Películas con mejores calificaciones promedio #####################
base= pd.read_sql('''select title, 
            round(avg(movieRating),2) as average
            from rating_movies 
            group by title
            order by average desc limit 10
            ''', conn)

data  = go.Bar(x=base.title,y=base.average, text=base.average, textposition="outside")
Layout=go.Layout(title="top 10 best average rating",xaxis={'title':'title'},yaxis={'title':'Count'}, width=1000, height=700)
go.Figure(data,Layout)

#En esta gráfica se puede evidenciar que el puntaje promedio más alto es de 4.59 para la película
#'Secrets & Lies' de 1996, seguida de 'Guess Who's Coming to Dinner' de 1997 con un 4.5 de puntaje.
#'Secrets & Lies' es una película que ha ganado múltiples premios a lo largo de su
#permanencia en el mundo del cine (IMDb, 2022)
#Referencia: IMDb. (2022). Awards Secrets & Lie. https://m.imdb.com/title/tt0117589/awards/?ref_=tt_awd


################### Películas con peores calificaciones #####################
base= pd.read_sql('''select title, 
            round(avg(movieRating),2) as average
            from rating_movies 
            group by title
            order by average asc limit 10
            ''', conn)

data  = go.Bar(x=base.title,y=base.average, text=base.average, textposition="outside")
Layout=go.Layout(title="top 10 worst average rating",xaxis={'title':'title'},yaxis={'title':'Count'}, width=1000, height=700)
go.Figure(data,Layout)

#De la gráfica se puede decir que existen puntajes promedio muy bajos para ciertas películas,
#donde la que tiene el peor puntaje promedio es 'Problem child' de 1990 con un 1,58 de calificación.

