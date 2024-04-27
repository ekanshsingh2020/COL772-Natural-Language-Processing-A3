if [ "$1" == "train" ]; then
    python3 train.py $2 $3
else
    python3 inference.py $2 $3 $4
fi