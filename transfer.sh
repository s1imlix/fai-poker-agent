#!/bin/bash

LOCAL_DIR="."
REMOTE_DIR="~/fai-final"
HOST="b11902083@ws3.csie.ntu.edu.tw"
rsync -avz $LOCAL_DIR $HOST:$REMOTE_DIR

