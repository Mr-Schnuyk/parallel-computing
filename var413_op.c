#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define N 4048//(2*2*2*2*2*2+2)
double maxeps = 0.1e-7;
int itmax = 10;
double w = 0.5;
double eps;

void relax(double A [N][N]);
void init(double A [N][N]);
void verify(double A [N][N]); 

int main(int an, char **as)
{
	int it;
	double A [N][N];
	
	double start, end;
    start = omp_get_wtime();
	
	init(A);

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax(A);
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	printf("var13_op, N: %i\n", N);
	verify(A);
	
	end = omp_get_wtime();
    printf("Total time: %6f\n", end-start);

	return 0;
}


void init(double A [N][N])
{
	int i;
	for(i=0; i<=N-1; i++)
	{
		int j;
		for(j=0; j<=N-1; j++)
		{
			if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
			else A[i][j]= ( 1. + i + j) ;
		}
	}
	
} 


void relax(double A [N][N])
{
	int i;
	for(i=1; i<=N-2; i++)
	{
		int j;
		for(j=1; j<=N-2; j++)
		if ((i + j) % 2 == 1)
		{
			double b;
			b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
			eps =  Max(fabs(b),eps);
			A[i][j] = A[i][j] + b;
		}
	}
	for(i=1; i<=N-2; i++)
	{
		int j;
		for(j=1; j<=N-2; j++)
		if ((i + j) % 2 == 0)
		{
			double b;
			b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
			A[i][j] = A[i][j] + b;
		}
	}

}


void verify(double A [N][N])
{ 
	double s;

	s=0.;
	int i;
	for(i=0; i<=N-1; i++)
	{
		int j;
		for(j=0; j<=N-1; j++)
		{
			s=s+A[i][j]*(i+1)*(j+1)/(N*N);
		}
	}
	printf("  S = %f\n",s);
}


