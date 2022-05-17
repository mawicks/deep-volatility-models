SYMBOLS="bnd edv tyd gld vnq vti spy qqq qld xmvm vbk xlv fxg rxl fxl ibb vgt iyf xly uge jnk"

args=""
for symbol in $SYMBOLS
do
    args="$args --symbol $symbol"
done
python ./train_univariate.py $* $args
