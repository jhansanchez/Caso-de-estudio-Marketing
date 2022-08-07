#################################################################################
########################### Importación de paquetes #############################
#################################################################################
import numpy as np ### manejo de arrays
import pandas as pd ### para manejo de datos
import sqlite3 as sql ### conexión con SQL
import plotly.graph_objs as go ### gráficos
import funciones as fn ### para funciones
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ### para análisis interactivo
from mlxtend.preprocessing import TransactionEncoder ### para separar los géneros
from datetime import datetime ### para el cambio de formato a fecha
 
#### Paquete para sistemas de recomendación surprise 
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import SlopeOne

####Paquete para sistema basado en contenido ####
from sklearn import neighbors
 
conn=sql.connect('db_movies')
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

#### ver tablas disponibles en base de datos ###
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()
 
movies=pd.read_sql('select* from movie2',conn)
ratings=pd.read_sql('select* from ratings2',conn)
ratings["view_time"]=pd.to_datetime(ratings['view_time'])
 
fn.ejecutar_sql('exploración cm.sql', cur)
pd.read_sql('select count(*) rating_movies', conn)
rating_movies =pd.read_sql('select * from  rating_movies',conn)
 

movies=pd.read_sql('select* from movie2',conn).drop(['index'], axis = 1)
movies.info()
movies['premiere_year']=movies.premiere_year.astype('int')
movies.info()

##### escalar para que año esté en el mismo rango ###
sc=MinMaxScaler()
movies[["premiere_yearSC"]]=sc.fit_transform(movies[['premiere_year']])
 
##### eliminar filas que no se van a utilizar ###
movies_dum1=movies.drop(columns=['movieId','title','amountRat','premiere_year'])

###############################################################################
######### 4. Sistema de recomendación basado en contenido KNN #################
############## Con base en todo lo visto por el usuario #######################
###############################################################################

usuarios=pd.read_sql('select distinct (userId) as userId from rating_movies',conn)

def recomendar(userId=list(usuarios['userId'].value_counts().index)):
    
    ratings=pd.read_sql('select *from rating_movies where userId=:user',conn, params={'user':userId})
    l_movies_r=ratings['movieId'].to_numpy()
    movies_dum1[['movieId','title']]=movies[['movieId','title']]
    movies_r=movies_dum1[movies_dum1['movieId'].isin(l_movies_r)]
    movies_r=movies_r.drop(columns=['movieId','title'])
    movies_r["indice"]=1 
    centroide=movies_r.groupby("indice").mean() 
        
    movies_nr=movies_dum1[~movies_dum1['movieId'].isin(l_movies_r)] 
    movies_nr=movies_nr.drop(columns=['movieId','title'])
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nr) 
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0]
    recomend_b=movies.loc[ids][['title','movieId']]
    leidos=movies[movies['movieId'].isin(l_movies_r)][['title','movieId']]
    
    return recomend_b


#recomendar(610)

print(interact(recomendar))

######################################################
##### Generar base con recomendaciones por usuario ####
########################################################

recomendaciones=recomendar(1)
recomendaciones["userId"]= 1

for i in range(2, 611):
    recomendaciones1= recomendar(i)
    recomendaciones1["userId"]= i
    recomendaciones = pd.concat([recomendaciones, recomendaciones1])
    

#user_id=610
#recomendaciones=recomendar(610)
#recomendaciones["user_id"]=user_id

recomendaciones.to_excel('recomendaciones.xlsx',index=False)
recomendaciones.to_csv('recomendaciones.csv',index=False)