
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



--drop table if exists movies2;
--create table movies2 as select movieId, title, 
                       -- from movies;



--drop table if exists rating_movies;

--create table rating_movies as 
--select  a.*, b.*
--from ratings1 a left join
--movies1 b on a.movieId = b.movieId;

