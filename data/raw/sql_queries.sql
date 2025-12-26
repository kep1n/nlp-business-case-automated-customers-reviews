select * from games where author_id = 76561199103288511

select * from reviews where appid = 19000 order by author_id

select count(*) from reviews where voted_up = 0
select count(*) from reviews where voted_up = 1


select count(*) from reviews

-- DELETE FROM reviews WHERE rowid NOT IN (SELECT MIN(rowid) FROM reviews GROUP BY appid, author_id, review);

-- delete from reviews where appid = 622650