#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define N 160//(2*2*2*2*2*2+2)
double maxeps = 0.1e-7;
int itmax = 10;
double w = 0.5;
double eps;

void relax(double A[N][N]);
void init(double A[N][N]);
void verify(double A[N][N]);

int main(int an, char **as)
{
    int it;
    double A[N][N];
    
    double start, end;
    start = omp_get_wtime();
    
    init(A);

    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax(A);
        printf("it=%4i   eps=%f\n", it,eps);
        if (eps < maxeps) break;
    }

    printf("var13_for, N: %i\n", N);
    verify(A);
    
    end = omp_get_wtime();
    printf("Total time: %6f\n", end-start);

    return 0;
}

void init(double A[N][N])
{
    int i, j;
    // Параллельная инициализация
    #pragma omp parallel for private(j) schedule(static)
    for(i=0; i<=N-1; i++)
    {
        for(j=0; j<=N-1; j++)
        {
            if(i==0 || i==N-1 || j==0 || j==N-1) 
                A[i][j] = 0.;
            else 
                A[i][j] = (1. + i + j);
        }
    }
}

void relax(double A[N][N])
{
    int i, j;
    double local_eps = 0.;
    
    // for с reduction для вычисления максимума
    #pragma omp parallel for private(j) reduction(max:local_eps) schedule(static)
    for(i=1; i<=N-2; i++)
    {
        for(j=1; j<=N-2; j++)
        {
            if ((i + j) % 2 == 1)
            {
                double b = w * ((A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4.0 - A[i][j]);
                double abs_b = fabs(b);
                if (abs_b > local_eps) 
                    local_eps = abs_b;
                A[i][j] += b;
            }
        }
    }
    
    #pragma omp parallel for private(j) schedule(static)
    for(i=1; i<=N-2; i++)
    {
        for(j=1; j<=N-2; j++)
        {
            if ((i + j) % 2 == 0)
            {
                double b = w * ((A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4.0 - A[i][j]);
                A[i][j] += b;
            }
        }
    }
    
    eps = local_eps;
}

void verify(double A[N][N])
{
    double s = 0.;
    int i, j;
    
    // Параллельная верификация с reduction для суммирования
    #pragma omp parallel for private(j) reduction(+:s) schedule(static)
    for(i=0; i<=N-1; i++)
    {
        for(j=0; j<=N-1; j++)
        {
            s += A[i][j] * (i+1) * (j+1) / (N*N);
        }
    }
    printf("  S = %f\n",s);
}