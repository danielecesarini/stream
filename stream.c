#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <omp.h>

/*-----------------------------------------------------------------------
 * INSTRUCTIONS:
 *
 *	1) STREAM requires different amounts of memory to run on different
 *           systems, depending on both the system cache size(s) and the
 *           granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *           to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. I don't worry about the difference
 *           between 10^6 and 2^20, so in practice the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.  
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.  
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.  
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M elements.
 *
 *      Array size can be set at compile time without modifying the source
 *          code for the (many) compilers that support preprocessor definitions
 *          on the compile line.  E.g.,
 *                gcc -O -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream.100M
 *          will override the default size of 10M with a new size of 100M elements
 *          per array.
 */
#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	20000000
#endif

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */
#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

/*
 *	3) Compile the code with optimization.  Many compilers generate
 *       unreasonably bad code before the optimizer tightens things up.  
 *     If the results are unreasonably good, on the other hand, the
 *       optimizer might be too smart for me!
 *
 *     For a simple single-core version, try compiling with:
 *            cc -O stream.c -o stream
 *     This is known to work on many, many systems....
 *
 *     To use multiple cores, you need to tell the compiler to obey the OpenMP
 *       directives in the code.  This varies by compiler, but a common example is
 *            gcc -O -fopenmp stream.c -o stream_omp
 *       The environment variable OMP_NUM_THREADS allows runtime control of the 
 *         number of threads/cores used when the resulting "stream_omp" program
 *         is executed.
 *
 *     To run with single-precision variables and arithmetic, simply add
 *         -DSTREAM_TYPE=float
 *     to the compile line.
 *     Note that this changes the minimum array sizes required --- see (1) above.
 *
 *     The preprocessor directive "TUNED" does not do much -- it simply causes the 
 *       code to call separate functions to execute each kernel.  Trivial versions
 *       of these functions are provided, but they are *not* tuned -- they just 
 *       provide predefined interfaces to be replaced with tuned code.
 *
 *-----------------------------------------------------------------------*/

#define HLINE "--------------------------------------------------------------------\n"

#ifndef MIN
#   define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#   define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#ifndef STREAM_TYPE
#   define STREAM_TYPE double
#endif

#define SCALAR 3.0
static char	*label[4] = {
    "Copy:      ", 
    "Scale:     ",
    "Add:       ", 
    "Triad:     "
};

int checktick();
double mysecond();

int main(int argc, char *argv[])
{
    int BytesPerWord;
    int i, k, nthreads = 0;
    unsigned long stream_array_size;

    if(argc == 1 || argc > 2)
        stream_array_size = STREAM_ARRAY_SIZE;
    else
    {
        stream_array_size = strtoul(argv[1], &argv[1], 10);
        if(stream_array_size < 1)
        {
            printf("[ERROR] The input value is invalid!");
            return -1;
        }
    }

    printf(HLINE);

    printf("STREAM version by Daniele Cesarini\n");

    printf(HLINE);

    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n", BytesPerWord);

    printf(HLINE);

    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf("The *best* time for each kernel (excluding the first iteration)\n"); 
    printf("will be used to compute the reported bandwidth.\n");

    printf(HLINE);

    int ncpus = sysconf(_SC_NPROCESSORS_ONLN);

    #pragma omp parallel 
        #pragma omp master
            nthreads = omp_get_num_threads();

    printf("Total number of CPU: %d\n", ncpus);
    printf("Number of Threads requested = %d\n", nthreads);

    printf(HLINE);

    int cpu_id[nthreads];
    

    #pragma omp parallel shared(cpu_id)
    {
        int tid = omp_get_thread_num();
        cpu_id[tid] = sched_getcpu();
    }

    for(i = 0; i < nthreads; i++)
        printf("Threads ID %d pinned on CPU %d\n", i, cpu_id[i]);

    printf(HLINE);

    // Allocate arrays
    int array_size = (int) ((double) stream_array_size / (double) nthreads);
    STREAM_TYPE *a[nthreads], *b[nthreads], *c[nthreads];
    #pragma omp parallel shared(a, b, c)
    {
        int tid = omp_get_thread_num();
        a[tid] = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * array_size);
        b[tid] = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * array_size);
        c[tid] = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * array_size);
    }

    printf("Memory per array (a,b,c) = %.1f MiB (%.1f GiB).\n", 
            BytesPerWord * ( (double) array_size / 1024. / 1024.),
            BytesPerWord * ( (double) array_size / 1024. / 1024. / 1024.));
    printf("Memory required per threads = %.1f MiB (%.1f GiB).\n",
	        (3.0 * BytesPerWord) * ((double) array_size / 1024. / 1024.),
	        (3.0 * BytesPerWord) * ((double) array_size / 1024. / 1024. / 1024.));
    printf("Total memory required (%d threads) = %.1f MiB (%.1f GiB).\n", nthreads,
	        (3.0 * BytesPerWord * nthreads) * ((double) array_size / 1024. / 1024.),
	        (3.0 * BytesPerWord * nthreads) * ((double) array_size / 1024. / 1024. / 1024.));
    
    printf(HLINE);

    int quantum;
    if((quantum = checktick()) >= 1)
    {
        printf("Your clock granularity/precision appears to be "
            "%d microseconds.\n", quantum);
    }
    else 
    {
        printf("Your clock granularity appears to be "
            "less than one microsecond.\n");
	    quantum = 1;
    }

    printf(HLINE);

    printf("Initialize arrays...\n");
    #pragma omp parallel shared(a, b, c)
    {
        int tid = omp_get_thread_num();
        for (i = 0; i < array_size; i++)
        {
            a[tid][i] = 1.0;
            b[tid][i] = 2.0;
            c[tid][i] = 0.0;
        }
    }

    printf(HLINE);

    double t = mysecond();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (i = 0; i < array_size; i++)
		    a[tid][i] = 2.0E0 * a[tid][i];
    }
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	       " of %d microseconds.\n", (int) t);
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least %d microseconds per test.\n", 
            quantum*100);

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");

    printf(HLINE);

    double times[4][NTIMES];
    for(k = 0; k < NTIMES; k++)
	{
        times[0][k] = mysecond();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (i = 0; i < array_size; i++)
                c[tid][i] = a[tid][i];
        }
        times[0][k] = mysecond() - times[0][k];

        times[1][k] = mysecond();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (i = 0; i < array_size; i++)
                b[tid][i] = SCALAR * c[tid][i];
        }
        times[1][k] = mysecond() - times[1][k];

        times[2][k] = mysecond();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (i = 0; i < array_size; i++)
                c[tid][i] = a[tid][i]+b[tid][i];
        }
        times[2][k] = mysecond() - times[2][k];

        times[3][k] = mysecond();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (i = 0; i < array_size; i++)
                a[tid][i] = b[tid][i] + SCALAR * c[tid][i];
        }
        times[3][k] = mysecond() - times[3][k];
    }

    double avgtime[4];
    double mintime[4];
    double maxtime[4];
    for (i = 0; i < 4; i++)
    {
        avgtime[i] = times[i][1];
        mintime[i] = times[i][1];
        maxtime[i] = times[i][1];
    }
    for(k = 2; k < NTIMES; k++)
	{
        for (i = 0; i < 4; i++)
        {
            avgtime[i] = avgtime[i] + times[i][k];
            mintime[i] = MIN(mintime[i], times[i][k]);
            maxtime[i] = MAX(maxtime[i], times[i][k]);

        }
    }
    for (i = 0; i < 4; i++)
        avgtime[i] = avgtime[i] / (double)(NTIMES-1);

    static double bytes[4];
    bytes[0] = 2 * sizeof(STREAM_TYPE) * array_size * nthreads;
    bytes[1] = 2 * sizeof(STREAM_TYPE) * array_size * nthreads;
    bytes[2] = 3 * sizeof(STREAM_TYPE) * array_size * nthreads;
    bytes[3] = 3 * sizeof(STREAM_TYPE) * array_size * nthreads;

    printf("Function  Bandwidth (MB/s)  Avg time (s)  Min time (s)  Max time (s)\n");
    for (i = 0; i < 4; i++) 
    {
		printf("%s%8.0f  %16.6f  %13.6f  %12.6f\n",
           label[i],
	       (bytes[i] / 1024. / 1024.) / mintime[i],
	       avgtime[i],
	       mintime[i],
	       maxtime[i]);
    }

    printf(HLINE);

    return 0;
}

# define M 20
int checktick()
{
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

    /*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++)
    {
        t1 = mysecond();
        while(((t2 = mysecond()) - t1) < 1.0E-6);
        timesfound[i] = t1 = t2;
	}

    /*
    * Determine the minimum difference between these M values.
    * This result will be our estimate (in microseconds) for the
    * clock granularity.
    */

    minDelta = 1000000;
    for (i = 1; i < M; i++)
    {
        Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
        minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
}

double mysecond()
{
    struct timespec sample;
    clock_gettime(CLOCK_MONOTONIC, &sample); 
    return (double) sample.tv_sec + ((double) sample.tv_nsec / 1.0E9);
}