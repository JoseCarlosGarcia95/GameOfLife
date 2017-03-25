/*
  ============================================================================
  Name        : GameOfLife.cu
  Author      : José Carlos
  Version     :
  Copyright   : 
  Description : CUDA compute reciprocals
  ============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

__global__ void kernel_game_of_life(int * status, int * new_status, int N);
int * matrix2vector(int ** matrix, int size);
void vector2matrix(int * vector, int ** matrix, int size);
void generate_new_game(int N, int iterations, int term, char * input_file);
