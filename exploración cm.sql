
----procesamientos---




---crear tabla con las películas que han sido vistas por más de 10 usuarios
drop table if exists movies_sel;



create table movies_sel as select movieId,
                         count(*) as cnt_rat
                         from ratings
                         group by movieId
                         having cnt_rat >10
                         order by cnt_rat desc ;

drop table if exists movies1;
create table movies1 as select movieId, title, genres
                        from movies;



drop table if exists  ratings_final;

create table  ratings_final as
select a.UserId as user_id,
a.movieId as movieId,
a.rating as movie_rating,
a.timestamp as timestamp
from ratings a 
inner join movies_sel b
on a.movieId =b.movieId;


drop table if exists movies_final;

create table movies_final as select 
a.*,
b.*
from movies1 a inner join
movies_sel b on a.movieId = b.movieId;



drop table if exists movies2;
--create table movies2 as select movieId, title, 
                       -- from movies;



--drop table if exists rating_movies;

--create table rating_movies as 
--select  a.*, b.*
--from ratings1 a left join
--movies1 b on a.movieId = b.movieId;

drop table if exists ratingsA;
create table ratingsA as select "user_id" as userId, movieId, "movie_rating" as movieRating , "view_time" as vierTime
                     from ratings2;


drop table if exists moviesA;
create table moviesA as select movieId, title, "Action" as Acccion, "Adventure" as Adventure, "Animation" as Animation,
       "Children" as Children, "Comedy" as Comedy, 
       "Crime" as Crime, "Documentary" as Documentary, "Drama" as Drama, "Fantasy" as Fantasy,
       "Film-Noir" as FilmNoir, "Horror" as Horror, "IMAX" as IMAX, "Musical" as Musical, "Mystery" as Mystery, "Romance" as Romance,
       "Sci-Fi" as SciFi, "Thriller" as Thriller, "War" as War, "Western" as Western, "premiere_year" as premiereYear
       from movie2;


drop table if exists rating_movies;

create table rating_movies as 
select  a.*, b.*
from ratingsA a left join
moviesA b on a.movieId = b.movieId;


