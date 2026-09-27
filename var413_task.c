#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define N 4800 //(2*2*2*2*2*2+2)

double maxeps = 0.1e-7;
int itmax = 10;
double w = 0.5;
double eps;

void relax(double A[N][N], int task_size);
void init(double A[N][N], int task_size);
void verify(double A[N][N], int task_size);

int main(int an, char **as)
{
    int it;
    double A[N][N];
    
	// Получаем количество нитей и вычисляем оптимальный размер задачи
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
	
    int tasks_per_thread = 8;
    
    // Вычисляем оптимальный размер задачи
    int task_size = N / (num_threads * tasks_per_thread);
    
    if (task_size < 8) task_size = 8;
    if (task_size > 256) task_size = 256;
	
    double start, end;
    start = omp_get_wtime();
    
    init(A, task_size);

    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax(A, task_size);
        printf("it=%4i   eps=%f\n", it,eps);
        if (eps < maxeps) break;
    }

    printf("var13_t, N: %i\n", N);
    verify(A, task_size);
    
    end = omp_get_wtime();
    printf("Total time: %6f\n", end-start);

    return 0;
}

void init(double A[N][N], int task_size)
{
    int i, j;
    // ѕараллельна€ инициализаци€ с использованием задач
    #pragma omp parallel
    #pragma omp single
    {
        for(i=0; i<=N-1; i+=task_size)
        {
            #pragma omp task firstprivate(i, task_size) shared(A)
            {
                int i_end = i + task_size;
                if (i_end > N) i_end = N;
                for(int ii=i; ii<i_end; ii++)
                {
                    for(j=0; j<=N-1; j++)
                    {
                        if(ii==0 || ii==N-1 || j==0 || j==N-1) 
                            A[ii][j] = 0.;
                        else 
                            A[ii][j] = (1. + ii + j);
                    }
                }
            }
        }
        #pragma omp taskwait
    }
}

void relax(double A[N][N], int task_size)
{
    double local_eps = 0.;
    
    // pервый проход по нечетным диагонал€м
    #pragma omp parallel
    {
        double my_local_eps = 0.;
        
        #pragma omp single
        {
            for(int i=1; i<=N-2; i+=task_size)
            {
                #pragma omp task firstprivate(i, task_size) shared(A)
                {
                    int i_end = i + task_size;
                    if (i_end > N-1) i_end = N-1;
                    for(int ii=i; ii<i_end; ii++)
                    {
                        for(int j=1; j<=N-2; j++)
                        {
                            if ((ii + j) % 2 == 1)
                            {
                                double b = w * ((A[ii-1][j] + A[ii+1][j] + A[ii][j-1] + A[ii][j+1]) / 4.0 - A[ii][j]);
                                double abs_b = fabs(b);
                                if (abs_b > my_local_eps) 
                                    my_local_eps = abs_b;
                                A[ii][j] += b;
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
        
        #pragma omp critical
        {
            if (my_local_eps > local_eps) local_eps = my_local_eps;
        }
    }
    
    // ¬торой проход по четным диагонал€м
    #pragma omp parallel
    #pragma omp single
    {
        for(int i=1; i<=N-2; i+=task_size)
        {
            #pragma omp task firstprivate(i, task_size) shared(A)
            {
                int i_end = i + task_size;
                if (i_end > N-1) i_end = N-1;
                for(int ii=i; ii<i_end; ii++)
                {
                    for(int j=1; j<=N-2; j++)
                    {
                        if ((ii + j) % 2 == 0)
                        {
                            double b = w * ((A[ii-1][j] + A[ii+1][j] + A[ii][j-1] + A[ii][j+1]) / 4.0 - A[ii][j]);
                            A[ii][j] += b;
                        }
                    }
                }
            }
        }
        #pragma omp taskwait
    }
    
    eps = local_eps;
}

void verify(double A[N][N], int task_size)
{
    double s = 0.;
    
    // ѕараллельна€ верификаци€ с использованием задач
    #pragma omp parallel reduction(+:s)
    #pragma omp single
    {
        for(int i=0; i<=N-1; i+=task_size)
        {
            #pragma omp task firstprivate(i, task_size) shared(A)
            {
                double partial_sum = 0.;
                int i_end = i + task_size;
                if (i_end > N) i_end = N;
                for(int ii=i; ii<i_end; ii++)
                {
                    for(int j=0; j<=N-1; j++)
                    {
                        partial_sum += A[ii][j] * (ii+1) * (j+1) / (N*N);
                    }
                }
                s += partial_sum;
            }
        }
        #pragma omp taskwait
    }
    printf("  S = %f\n",s);
}