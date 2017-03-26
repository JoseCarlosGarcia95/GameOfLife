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
#include <string.h>

#define DEAD_CELL 0
#define ALIVE_CELL 1

void printCurrentState(int ** __state, int N, int term) {
  system("clear");
  for(int i = 0; i < N;i++)
  {
      for(int j = 0; j < N;j++) {
	if(__state[i][j] == DEAD_CELL)
	  printf(" ");
	else
	  printf("+");
      }
      printf("\n");
    }
}


__device__ int MOVEMENTS_SIZE = 8;

__device__ int MOVEMENTS[8][2] =
  {
    {0, 1},
    {1, 0},
    {0, -1},
    {-1, 0},
    {1, 1},
    {-1, -1},
    {1, -1},
    {-1, 1}
  };

__global__ void kernel_game_of_life(int * status, int * new_status, int N) {
  int i, j, neighbour_x, neighbour_y, neighbours_a, neighbours_d, neighbours_status, current_status;

  i = (threadIdx.x + blockIdx.x * blockDim.x) % N;
  j = (threadIdx.x + blockIdx.x * blockDim.x) / N;

  current_status = status[i*N+j];

  neighbours_a = neighbours_d = 0;
  for(int z = 0; z < MOVEMENTS_SIZE;z++) {
    neighbour_x = i + MOVEMENTS[z][0];
    neighbour_y = j + MOVEMENTS[z][1];

    if(neighbour_x < 0 || neighbour_y < 0 || neighbour_x >= N || neighbour_y >= N) {
      continue;
    }
    neighbours_status = status[neighbour_x*N + neighbour_y];

    if(neighbours_status == 0) {
      neighbours_d++;
    } else if(neighbours_status == 1) {
      neighbours_a++;
    }
  }


  if(current_status == 0 && neighbours_a == 3) {
    new_status[i*N+j] = 1;
  } else if (current_status == 1 && (neighbours_a > 3 || neighbours_a < 2)) {
    new_status[i*N+j] = 0;
  }
}


int * matrix2vector(int ** matrix, int size) {
  int * vector = (int*)malloc(size*size*sizeof(int)), i, j;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      vector[i*size + j] = matrix[i][j];
    }
  }
  return vector;
}

void vector2matrix(int * vector, int ** matrix, int size) {
  int i, j;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      matrix[i][j] = vector[i*size + j];
    }
  }
}

void read_status_matrix(int ** matrix, int N, char * input_file) {
  int i, c;
  FILE * fp;

  fp = fopen(input_file, "r");

  i = 0;
  do {
    c = fgetc (fp);
    if (c != '\n' && c != EOF && i < N*N) {
      matrix[i / N][i % N] = c - '0';
      i++;
    }
  } while (c != EOF);

  fclose(fp);
}

void generate_new_game(int N, int iterations, int term, char * input_file)
{
  int i, j, minGridSize, blockSize, gridSize;

  int ** __status_matrix, * __status_vector, * __status_dev_vector, * __new_status_dev_vector;

  __status_matrix = (int**)malloc(sizeof(int*)*N);

  for(i = 0; i < N;i++)
    __status_matrix[i] = (int*)malloc(sizeof(int)*N);

  srand(time(NULL));

  if(strlen(input_file) == 0) {
    for(i = 0; i < N;i++)
    {
      for(j = 0; j < N;j++)
    	{
	  __status_matrix[i][j] = rand() % 2;
    	}
    }
  } else {
    read_status_matrix(__status_matrix, N, input_file);
  }

  __status_vector = matrix2vector(__status_matrix, N);


  cudaMalloc((void**)&__status_dev_vector, N*N*sizeof(int));
  cudaMalloc((void**)&__new_status_dev_vector, N*N*sizeof(int));

  cudaMemcpy(__status_dev_vector, __status_vector, N*N* sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(__new_status_dev_vector, __status_vector, N*N* sizeof(int), cudaMemcpyHostToDevice);

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_game_of_life, 0, N*N);
  gridSize = (N*N + blockSize - 1) / blockSize;

  printCurrentState(__status_matrix, N, term);

  for(i = 0; i < iterations;i++) {
    kernel_game_of_life<<<gridSize, blockSize >>>(__status_dev_vector, __new_status_dev_vector, N);
    cudaDeviceSynchronize();

    cudaMemcpy(__status_vector, __new_status_dev_vector, N*N* sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(__status_dev_vector, __new_status_dev_vector, N*N* sizeof(int), cudaMemcpyDeviceToDevice);

    vector2matrix(__status_vector, __status_matrix, N);
    printCurrentState(__status_matrix, N, term);
    sleep(1);
  }

  cudaFree(__new_status_dev_vector);
  cudaFree(__status_dev_vector);
  free(__status_matrix);
  free(__status_vector);
}
