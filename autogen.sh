#!/bin/bash

#
#   /$$$$$$              /$$                                                                             /$$          
#  /$$__  $$            | $$                                                                            | $$          
# | $$  \ $$ /$$   /$$ /$$$$$$    /$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$$         /$$$$$$$ /$$   /$$  /$$$$$$$  /$$$$$$ 
# | $$$$$$$$| $$  | $$|_  $$_/   /$$__  $$ /$$__  $$ /$$__  $$| $$__  $$       /$$_____/| $$  | $$ /$$__  $$ |____  $$
# | $$__  $$| $$  | $$  | $$    | $$  \ $$| $$  \ $$| $$$$$$$$| $$  \ $$      | $$      | $$  | $$| $$  | $$  /$$$$$$$
# | $$  | $$| $$  | $$  | $$ /$$| $$  | $$| $$  | $$| $$_____/| $$  | $$      | $$      | $$  | $$| $$  | $$ /$$__  $$
# | $$  | $$|  $$$$$$/  |  $$$$/|  $$$$$$/|  $$$$$$$|  $$$$$$$| $$  | $$      |  $$$$$$$|  $$$$$$/|  $$$$$$$|  $$$$$$$
# |__/  |__/ \______/    \___/   \______/  \____  $$ \_______/|__/  |__/       \_______/ \______/  \_______/ \_______/
#                                          /$$  \ $$                                                                  
#                                         |  $$$$$$/                                                                  
#                                          \______/
#                                                                                                   - by JoseCarlos95

# ADVANCED CONFIGURATION
NVCC=/usr/local/cuda-8.0/bin/nvcc
ARCH=compute_35
CODE=sm_35

# BASIC CONFIGURATION
PROJECT_NAME="GameOfLife"
PROJECT_FILES=("main" "GameOfLife")

echo 'Building files'

rm -rf obj
mkdir obj

for FILE in "${PROJECT_FILES[@]}"; do
    echo "Generating $FILE"
    $NVCC -G -g -O0 -gencode arch=$ARCH,code=$CODE  -odir "obj" -M -o "obj/$FILE.d" "src/$FILE.cu"
    $NVCC -G -g -O0 --compile --relocatable-device-code=false -gencode arch=$ARCH,code=$ARCH -gencode arch=$ARCH,code=$CODE  -x cu -o  "obj/$FILE.o" "src/$FILE.cu"
done

rm -rf bin
mkdir bin

echo "Generating binaries"

PROJECT_OBJECTS=$(printf 'obj/%s.o ' "${PROJECT_FILES[@]}")

$NVCC --cudart static --relocatable-device-code=false -gencode arch=$ARCH,code=$ARCH -gencode arch=$ARCH,code=$CODE -link -o "bin/$PROJECT_NAME" $PROJECT_OBJECTS
