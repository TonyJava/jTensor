#!/bin/bash
for i in {101..150};
do
	curl "http://gobase.org/9x9/book/part3/game_$i.sgf" > "game$i.sgf"
done
