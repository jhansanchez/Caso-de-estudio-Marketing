###############################################################################
###########################Importación de paquetes#############################
###############################################################################
import numpy as np ###manejo de arrays
import pandas as pd ### para manejo de datos
import sqlite3 as sql ### conexión con SQL
import plotly.graph_objs as go ### gráficos
import funciones as fn ### para funciones
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from mlxtend.preprocessing import TransactionEncoder #para separar los generos
from datetime import datetime # para el cambio de formato a fecha
 
####Paquete para sistemas de recomendación surprise
###Puede generar problemas en instalación local de pyhton. Genera error instalando con pip
 
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import SlopeOne
from surprise.prediction_algorithms import matrix_factorization, SVD

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
 
 
#######################################################################
######## Sistema de recomendación basado en contenido - Manual ########
#######################################################################
 
movies=pd.read_sql('select* from movie2',conn).drop(['index'], axis = 1)
 
movies.info()
movies['premiere_year']=movies.premiere_year.astype('int')
movies.info()
 
 
##### escalar para que año esté en el mismo rango ###
 
sc=MinMaxScaler()
movies[["premiere_yearSC"]]=sc.fit_transform(movies[['premiere_year']])
 
 
## eliminar filas que no se van a utilizar ###
 
movies_dum1=movies.drop(columns=['movieId','title','amountRat','premiere_year'])
 
###### mostrar películas recomendadas #####
 
movie='Forrest Gump (1994)'
ind_movie=movies[movies['title']==movie].index.values.astype(int)[0]
correlaciones= movies_dum1.corrwith(movies_dum1.iloc[ind_movie,:],axis=1)
correlaciones.sort_values(ascending=False)
 
def recomendacion(movie = list(movies['title'])):
     
    ind_movie=movies[movies['title']==movie].index.values.astype(int)[0]  
    similar_movies = movies_dum1.corrwith(movies_dum1.iloc[ind_movie,:],axis=1)
    similar_movies =  similar_movies.sort_values(ascending=False)
    top_similar_movies= similar_movies.to_frame(name="correlation").iloc[0:15,]
    top_similar_movies['title']=movies["title"]
   
    return top_similar_movies
 
print(interact(recomendacion))
 
 
############################################################################
#####1. Sistema de recomendación filtro colaborativo basado en usuario #####
############################################################################
 
ratings=pd.read_sql('select * from ratings2', conn)

###### leer datos desde tabla de pandas
reader = Reader(rating_scale=(0.5, 5.0))

###las columnas deben estar en orden estándar: user item rating
data   = Dataset.load_from_df(ratings[['user_id','movieId','movie_rating']], reader)
 

models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline(),SlopeOne()] 
results = {}

for model in models:
 
    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result


performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')


param_grid = { 'sim_options' : {'name': ['msd','cosine', ], \
                                'min_support': [5,6,7,10], \
                                'user_based': [False, True]}
             }

gridsearchKNNWithMeans = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], \
                                      cv=2, n_jobs=2)
                                    
gridsearchKNNWithMeans.fit(data)

gridsearchKNNWithMeans.best_params["rmse"]
gridsearchKNNWithMeans.best_score["rmse"]


################# Realizar predicciones

trainset = data.build_full_trainset()


sim_options       = {'name':'msd','min_support':5,'user_based':True}
model = KNNWithMeans(sim_options=sim_options)
model=model.fit(trainset)


predset = trainset.build_anti_testset() 
predictions = model.test(predset) ### función muy pesada
predictions_df = pd.DataFrame(predictions)


def recomendaciones(user_id,n_recomend=15):
    
    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid','est']]
    recomendados.to_sql('reco',conn,if_exists="replace")
    
    recomendados=pd.read_sql('''select a.*, b.title 
                             from reco a left join rating_movies b
                             on a.iid=b.movieId ''', conn)

    return(recomendados)

np.set_printoptions(threshold=sys.maxsize)

us1=recomendaciones(user_id=21,n_recomend=5)
us1=us1.drop_duplicates() 


#######################################################################
#### Sistema de recomendación basado en contenido KNN #################
#######################################################################

##### ### entrenar modelo #####

model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(movies_dum1)
dist, idlist = model.kneighbors(movies_dum1) 

distancias=pd.DataFrame(dist)
id_list=pd.DataFrame(idlist) 


def moviesRecommender(movie_name = list(movies['title'].value_counts().index)):
    movies_list_name = []
    movies_id = movies[movies['title'] == movie_name].index
    movies_id = movies_id[0]
    for newid in idlist[movies_id]:
        movies_list_name.append(movies.loc[newid].title)
    return movies_list_name

print(interact(moviesRecommender))

#######################################################################
#### Sistema de recomendación basado en contenido KNN #################
#### Con base en todo lo visto por el usuario #######################
#######################################################################

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

recomendar(30)


print(interact(recomendar))

