/******************************************************************************
 * Parallelization:  Sieve of Eratosthenes
 * By Aaron Weeden, The Shodor Education Foundation, Inc.
 *
 * OpenMP code
 *  -- to run, use ./sieve.exe -n N, where N is the value under which to find
 *     primes.
 *  -- to list the primes, include the -o option.
 *  -- see http://shodor.org/petascale/materials/UPModules/sieveOfEratosthenes/
 *     for module document with discussion of the code and its algorithm
 *****************************************************************************/
#include <math.h> // for sqrt
#include <omp.h> // for omp_get_wtime
#include <stdbool.h> // for true and false
#include <stdio.h> // for printf
#include <stdlib.h> // for malloc and free
#include <unistd.h> // for getopt

int main(int argc, char **argv)
{
  // Declare variables
  int N = 16; // The positive integer under which we are finding primes
  int sqrtN; // The square root of N, which is stored in a variable to
             // avoid making excessive calls to sqrt(N)
  int c; // Used to check the next number to be circled
  int m; // Used to check the next number to be marked
  int *list; // The list of numbers -- if list[x] equals 1, then x is marked.
             // If list[x] equals 0, then x is unmarked.
  char next_option = ' '; // Used for parsing command line arguments
  double startTime = omp_get_wtime(); // Start the timer
  bool is_printing = false; // Decide whether to print the primes
  
  // Parse command line arguments -- see the manual page for getopt
  while((next_option = getopt(argc, argv, "n:o")) != -1)
  {
    switch(next_option)
    {
      case 'n':
        N = atoi(optarg);
        break;
      case 'o':
        is_printing = true;
        break;
      case '?':
      default:
        fprintf(stderr, "Usage: %s [OPTIONS]\n",
          "-n N, where N is the number under which to find primes\n"
          "-o to print out the primes after they have been found\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
  sqrtN = (int)sqrt(N);
  
  // Allocate memory for list
  list = (int*)malloc(N * sizeof(int));
  
  // Exit if malloc failed
  if(list == NULL)
  {
    fprintf(stderr, "Error: Failed to allocate memory for list.\n");
    exit(EXIT_FAILURE);
  }
  // Run through each number in the list
#pragma omp parallel for private(c)
  for(c = 2; c < N; c++)
  {
    // Set each number as unmarked
    list[c] = 0;
  }
  // Run through each number in the list up through the square root of N
  for(c = 2; c <= sqrtN; c++)
  {
    // If the number is unmarked
    if(list[c] == 0)
    {
      // Run through each number bigger than c and less than N
#pragma omp parallel for private(m)
      for(m = c+1; m < N; m++)
      {
        // If m is a multiple of c
        if(m % c == 0)
        {
          // Mark m
          list[m] = 1;
        }
      }
    }
  }
  if (is_printing)
  {
    // Run through each number in the list
    for(c = 2; c <= N - 1; c++)
    {
      // If the number is unmarked
      if(list[c] == 0)
      {
        // The number is prime; print it
        printf("%d ", c);
      }
    }
    printf("\n");
  }
  // Deallocate memory for list
  free(list);
  
  // Stop the timer, print the total elapsed time
  printf("Runtime: %f seconds\n", omp_get_wtime() - startTime);

  return 0;
}
