
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


conn=sql.connect('db_movies')
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

movies=pd.read_sql('select* from movie2',conn)
ratings=pd.read_sql('select* from ratings2',conn)
ratings["view_time"]=pd.to_datetime(ratings['view_time'])

fn.ejecutar_sql('exploración cm.sql', cur)
pd.read_sql('select count(*) rating_movies', conn)
rating_movies =pd.read_sql('select * from  rating_movies',conn)

###############################################################################
#############################Exploración de datos##############################
###############################################################################

#######################Top 10 películas más calificadas######################## 
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
cr= pd.read_sql('''select
             sum(Action) as Action, sum(Adventure) as Adventure, sum(Animation) as Animation, sum(Children) as Children,
            sum(Comedy) as Comedy, sum(Crime) as Crime, sum(Documentary) as Documentary, sum(Drama) as Drama,
            sum(Fantasy) as Fantasy, sum("Film-Noir") as FilmNoir, sum(Horror) as Horror, sum(IMAX) as IMAX,
            sum(Musical) as Musical, sum(Mystery) as Mystery, sum(Romance) as Romance, 
            sum("Sci-Fi") as SciFi, sum(Thriller) as Thriller, sum(War) as War, sum(Western) as Western
            from movie2
            ''', conn)

cr1= cr.transpose()
cr1 = cr1.rename(columns={0: 'conteo'})
cr1 =cr1.rename(columns={'': 'generos'}).reset_index()
cr1 = cr1.rename(columns={"index": 'generos'})

data  = go.Bar( x=cr1.generos ,y=cr1.conteo, text=cr1.conteo, textposition="outside")
Layout=go.Layout(title="count of genres ",xaxis={'title':'genres'},yaxis={'title':'Count'})
go.Figure(data,Layout)

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


##################### Número de vistas por año ############################3
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

