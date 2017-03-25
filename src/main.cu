#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "GameOfLife.cuh"

int main(int argc, const char * argv[]) {

  char * opt_input;
  int opt_term, opt_its, opt_size;

  opt_term = 1;
  opt_its = 100;

  opt_input = (char*) malloc(1);
  opt_input[0] = 0;
  
  switch(argc) {
  case 1:
      printf("Syntax: %s size it (term input)\n\n", argv[0]);
      printf("== Options ==\n");
      printf("size: Size of the game\n");
      printf("it: The number of iterations before ending. Negative iterations generate an infinite game\n");
      printf("term: Show output in terminal or generate an OpenGL frame.\n");
      printf("input: Use a predefined game.\n");
      break;
  case 5:
    opt_input = (char*)malloc(strlen(argv[3]) + 1);
    strcpy(opt_input, argv[4]);
  case 4:
    opt_term = atoi(argv[3]);
  case 3:
    opt_its = atoi(argv[2]);
  case 2:
    opt_size = atoi(argv[1]);

    generate_new_game(opt_size, opt_its, opt_term, opt_input);
    break;
  default:
    printf("Syntax: %s it (term input)\n\n", argv[0]);
    printf("== Options ==\n");
    printf("size: Size of the game\n");
    printf("it: The number of iterations before ending. Negative iterations generate an infinite game\n");
    printf("term: Show output in terminal or generate an OpenGL frame.\n");
    printf("input: Use a predefined game.\n");
    break;
    
  }
}