#rm snapshots/*

#export NAME=$(basename "$PWD")
#echo $NAME

#creates docker and leaves it open
nvidia-docker run -dit liammcgold/gunpowder:tutorial /bin/bash

#liammcgold/gunpowder:tutorial \



python -u Train_and_predict_Tensorflow.py