#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define N 480//(2*2*2*2*2*2+2)
double maxeps = 0.1e-7;
int itmax = 10;
double w = 0.5;

int rank, size;
MPI_Comm comm = MPI_COMM_WORLD;
double eps;

void relax(double *local_A, int local_rows, int total_rows);
void init(double *local_A, int local_rows, int total_rows);
void verify(double *local_A, int local_rows, int total_rows);
void dop_rows(double *local_A, int local_rows, int total_rows);

int main(int argc, char **argv)
{
    int it;
    double start, end;
    
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // распределение строк между процессами
    int total_rows = N;
    int base_rows = total_rows / size;
    int remainder = total_rows % size;
    
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    int start_row = 0;
    
	int i;
    for (i = 0; i < rank; i++) {
        start_row += base_rows + (i < remainder ? 1 : 0);
    }
    
	// число строк для процесса - local_rows + 2 дополнительные
    double *local_A = (double*)malloc((local_rows + 2) * N * sizeof(double));
    if (local_A == NULL) {
        MPI_Abort(comm, 1);
    }
    
    start = MPI_Wtime();
    
	// Инициализация части массива
    init(local_A, local_rows, total_rows);
    
    for(it = 1; it <= itmax; it++)
    {
        eps = 0.;
        relax(local_A, local_rows, total_rows);
        if (eps < maxeps) break;
    }
    
    if (rank == 0) {
        printf("var13_mpi\n");
    }
    verify(local_A, local_rows, total_rows);
    
    end = MPI_Wtime();
    if (rank == 0) {
        printf("Total time: %6f\n", end - start);
    }
    
    free(local_A);
    MPI_Finalize();
    return 0;
}

void init(double *local_A, int local_rows, int total_rows)
{
    int start_row = 0;
    int base_rows = total_rows / size;
    int remainder = total_rows % size;
    
	int i;
    for (i = 0; i < rank; i++) {
        start_row += base_rows + (i < remainder ? 1 : 0);
    }
    
    for(i = 0; i < local_rows + 2; i++)
    {
		int j;
        for(j = 0; j < N; j++)
        {
            int global_i = start_row + i - 1;
            if(global_i < 0 || global_i >= total_rows || j == 0 || j == N-1) 
                local_A[i * N + j] = 0.;
            else 
                local_A[i * N + j] = (1. + global_i + j);
        }
    }
}

void dop_rows(double *local_A, int local_rows, int total_rows)
{
    MPI_Status status;
    
    // Обмен дополнительными-строками с соседними процессами
    if (rank > 0) {
        // Обмен верхней строкой
        MPI_Sendrecv(&local_A[1 * N], N, MPI_DOUBLE, rank - 1, 0,
                    &local_A[0 * N], N, MPI_DOUBLE, rank - 1, 1, comm, &status);
    }
    
    if (rank < size - 1) {
        // Обмен нижней строкой
        MPI_Sendrecv(&local_A[local_rows * N], N, MPI_DOUBLE, rank + 1, 1,
                    &local_A[(local_rows + 1) * N], N, MPI_DOUBLE, rank + 1, 0, comm, &status);
    }
}

void relax(double *local_A, int local_rows, int total_rows)
{
    double local_eps = 0.;
    dop_rows(local_A, local_rows, total_rows);
    
    // Первый проход нечетных диагоналей
	int i;
    for(i = 1; i <= local_rows; i++)
    {
		int j;
        for(j = 1; j <= N-2; j++)
        {
            if (((i + j) % 2 == 1))
            {
                double b = w * ((local_A[(i-1) * N + j] + local_A[(i+1) * N + j] + 
                               local_A[i * N + j-1] + local_A[i * N + j+1]) / 4.0 - 
                               local_A[i * N + j]);
                double abs_b = fabs(b);
                if (abs_b > local_eps) 
                    local_eps = abs_b;
                local_A[i * N + j] += b;
            }
        }
    }
	
    dop_rows(local_A, local_rows, total_rows);
    
    // Второй проход четных диагоналей
    for(i = 1; i <= local_rows; i++)
    {
		int j;
        for(j = 1; j <= N-2; j++)
        {
            if (((i + j) % 2 == 0))
            {
                double b = w * ((local_A[(i-1) * N + j] + local_A[(i+1) * N + j] + 
                               local_A[i * N + j-1] + local_A[i * N + j+1]) / 4.0 - 
                               local_A[i * N + j]);
                local_A[i * N + j] += b;
            }
        }
    }
    
    // Обновление глобального eps
    MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, comm);
}

void verify(double *local_A, int local_rows, int total_rows)
{
    double local_s = 0.;
    int start_row = 0;
    int base_rows = total_rows / size;
    int remainder = total_rows % size;
    
	int i;
    for (i = 0; i < rank; i++) {
        start_row += base_rows + (i < remainder ? 1 : 0);
    }
    
    for(i = 1; i <= local_rows; i++)
    {
        int global_i = start_row + i - 1;
		int j;
        for(j = 0; j < N; j++)
        {
            local_s += local_A[i * N + j] * (global_i + 1) * (j + 1) / (N * N);
        }
    }
    
    // Добавление результата к 0-ому процессу
    double global_s;
    MPI_Reduce(&local_s, &global_s, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    
    if (rank == 0) {
        printf("  S = %f\n", global_s);
    }
}