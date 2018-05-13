/* Model of heat diffusion - a 2D rectangular grid of cells is initialized with
   a temperature on the four edges and a temperature in the middle. At each
   step, each cell recomputes its heat as the average of its four neighbors.

   Command line options:
     -x: number of columns
     -y: number of rows
     -s: number of steps
     -t: temperature of top edge
     -l: temperature of left edge
     -r: temperature of right edge
     -b: temperature of bottom edge
     -m: temperature of middle
     -o: output results at each step (no argument)

   Author: Aaron Weeden, Shodor
*/
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int RowCount = 21;
int ColCount = 11;
int StepCount = 100;
double TopEdge = 100.0;
double LeftEdge = 0.0;
double RightEdge = 0.0;
double BottomEdge = 0.0;
double MiddleInit = 0.0;
bool IsOutputting = false;

double **CellArray;
double **NewCellArray;

void ParseArgs(int argc, char **argv)
{
  char c;

  // See the manual page for getopt
  while((c = getopt(argc, argv, "x:y:s:t:l:r:b:m:o")) != -1)
  {
    switch(c)
    {
      case 'x':
        ColCount = atoi(optarg);
        break;
      case 'y':
        RowCount = atoi(optarg);
        break;
      case 's':
        StepCount = atoi(optarg);
        break;
      case 't':
        TopEdge = atof(optarg);
        break;
      case 'l':
        LeftEdge = atof(optarg);
        break;
      case 'r':
        RightEdge = atof(optarg);
        break;
      case 'b':
        BottomEdge = atof(optarg);
        break;
      case 'm':
        MiddleInit = atof(optarg);
        break;
      case 'o':
        IsOutputting = true;
        break;
      case '?':
      default:
        fprintf(stderr, "Usage: %s OPTIONS\n"
          "-x ColCount\n"
          "-y RowCount\n"
          "-s StepCount\n"
          "-t TopEdge\n"
          "-l LeftEdge\n"
          "-r RightEdge\n"
          "-b BottomEdge\n"
          "-m MiddleInit\n"
          "-o turns on output\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
}

void AllocateMemory()
{
  int i;
  CellArray = (double**)malloc(RowCount * sizeof(double*));
  for (i = 0; i < RowCount; i++)
  {
    CellArray[i] = (double*)malloc(ColCount * sizeof(double));
  }
  NewCellArray = (double**)malloc(RowCount * sizeof(double*));
  for (i = 0; i < RowCount; i++)
  {
    NewCellArray[i] = (double*)malloc(ColCount * sizeof(double));
  }
}

void Initialize()
{
  int row;
  int col;
  for (row = 0; row < RowCount; row++)
  {
    for (col = 0; col < ColCount; col++)
    {
      CellArray[row][col] = MiddleInit;
    }
  }
}

void PrintCellArray(int const step)
{
  int row;
  int col;
  printf("Time step %d\n", step);
  for (row = 0; row < RowCount; row++)
  {
    for (col = 0; col < ColCount; col++)
    {
      printf("%6.3f ", CellArray[row][col]);
    }
    printf("\n");
  }
  printf("\n");
}

double AddTopNeighbor(int const row, int const col)
{
  return row > 0 ? CellArray[row - 1][col] : TopEdge;
}

double AddLeftNeighbor(int const row, int const col)
{
  return col > 0 ? CellArray[row][col - 1] : LeftEdge;
}

double AddRightNeighbor(int const row, int const col)
{
  return col < ColCount - 1 ? CellArray[row][col + 1] : RightEdge;
}

double AddBottomNeighbor(int const row, int const col)
{
  return row < RowCount - 1 ? CellArray[row + 1][col] : BottomEdge;
}

void CalcNewCellArray()
{
  int row;
  int col;
  for (row = 0; row < RowCount; row++)
  {
    for (col = 0; col < ColCount; col++)
    {
      NewCellArray[row][col] = 0.0;
#pragma omp parallel sections
      {
#pragma omp section
        {
          NewCellArray[row][col] += AddTopNeighbor(row, col);
        }
#pragma omp section
        {
          NewCellArray[row][col] += AddLeftNeighbor(row, col);
        }
#pragma omp section
        {
          NewCellArray[row][col] += AddRightNeighbor(row, col);
        }
#pragma omp section
        {
          NewCellArray[row][col] += AddBottomNeighbor(row, col);
        }
      }
      NewCellArray[row][col] *= 0.25;
    }
  }
}

void CopyNewCellArray()
{
  int row;
  int col;
#pragma omp parallel for private(row, col)
  for (row = 0; row < RowCount; row++)
  {
    for (col = 0; col < ColCount; col++)
    {
      CellArray[row][col] = NewCellArray[row][col];
    }
  }
}

void Simulate()
{
  int step;
  for (step = 0; step < StepCount; step++)
  {
    if (IsOutputting) {
      PrintCellArray(step);
    }
    CalcNewCellArray();
    CopyNewCellArray();
  }
}

void FreeMemory()
{
  int i;
  for (i = RowCount-1; i >= 0; i--)
  {
    free(NewCellArray[i]);
  }
  free(NewCellArray);
  for (i = RowCount-1; i >= 0; i--)
  {
    free(CellArray[i]);
  }
  free(CellArray);
}

int main(int argc, char **argv)
{
  // Start the timer
  double startTime = omp_get_wtime();

  // Get the number of threads
  int numThreads;
#pragma omp parallel
  {
    numThreads = omp_get_num_threads();
  }
  ParseArgs(argc, argv);
  AllocateMemory();
  Initialize();
  Simulate();
  FreeMemory();

  // Stop the timer, print the total elapsed time
  printf("Runtime: %f seconds with %d threads \n",
    omp_get_wtime() - startTime,
    numThreads);

  return 0;
}
