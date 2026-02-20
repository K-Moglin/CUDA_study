#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

// Simple matrix struct with row-major storage
typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

static void die(const char* msg) {
    perror(msg);
    exit(1);
}

/*----- Matrix utility layer-------*/
// Allocates a rows x cols matrix with aligned memory for better performance
static Matrix mat_alloc(int rows, int cols) {
    Matrix M;
    M.rows = rows;
    M.cols = cols;
#if defined(_ISOC11_SOURCE)
    M.data = aligned_alloc(64, (size_t)rows * (size_t)cols * sizeof(double));
    if (!M.data) die("aligned_alloc");
#else
    M.data = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (!M.data) die("malloc");
#endif
    return M;
}

// Frees matrix data and resets dimensions
static void mat_free(Matrix *M) {
    free(M->data);
    M->data = NULL;
    M->rows = M->cols = 0;
}

// Accessor functions for row-major storage
static inline double mat_get(const Matrix *M, int r, int c) {
    return M->data[(size_t)r * (size_t)M->cols + (size_t)c];
}

// Mutator function for row-major storage
static inline void mat_set(Matrix *M, int r, int c, double v) {
    M->data[(size_t)r * (size_t)M->cols + (size_t)c] = v;
}

// Fills matrix with deterministic pseudo-random values based on seed
static void mat_fill(Matrix *M, unsigned int seed) {
    //deterministic pseudo-random fill
    srand(seed);
    for (int i = 0; i < M->rows * M->cols; i++) {
        M->data[i] = (double)(rand() % 1000) / 100.0;
    }
}

// Sets all elements of the matrix to zero
static void mat_zero(Matrix *M) {
    memset(M->data, 0, (size_t)M->rows * (size_t)M->cols * sizeof(double));
}

// Compares two matrices for approximate equality within a given epsilon
static int mat_equal(const Matrix *A, const Matrix *B, double eps) {
    if (A->rows != B->rows || A->cols != B->cols) return 0;
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) {
        double diff = A->data[i] - B->data[i];
        if (diff < 0) diff = -diff;
        if (diff > eps) return 0;
    }
    return 1;
}

/*----------------Signle-threaded matmul----------------------
  C = A * B
  A: m * k, B: k * n, C: m * n
*/
static int matmul_single(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -2;

    const int m = A->rows, k = A->cols, n = B->cols;
    mat_zero(C);

    // i-k-j order improves locality for row-major C
    for (int i = 0; i < m; i++) {
        for (int kk = 0; kk < k; kk++) {
            const double a = mat_get(A, i, kk);
            const double *b_row = &B->data[(size_t)kk * (size_t)n];
            double * c_row = &C->data[(size_t)i * (size_t)n];
            for (int j = 0; j < n; j++) {
                c_row[j] += a * b_row[j];
            }
        }
    }
    return 0;
}

/*-----------------Mutil-threaded matmul with pthreads----------------
 Strategy: split rows of C across threads
*/
typedef struct {
    const Matrix *A;
    const Matrix *B;
    Matrix *C;
    int row_start;
    int row_end;
} WorkerArgs;

static void *worker_matmul_rows(void *arg) {
    WorkerArgs *w = (WorkerArgs*) arg;
    const Matrix *A = w->A;
    const Matrix *B = w->B;
    Matrix *C = w->C;

    const int k = A->cols;
    const int n = B->cols;

    for (int i = w->row_start; i < w->row_end; i++) {
        double *c_row = &C->data[(size_t)i * (size_t)n];
        //ensure row is zeroed (caller may already zero whole C, but safe here)
        for (int j = 0; j < n; j++) c_row[j] = 0.0;

        for (int kk = 0; kk < k; kk++) {
            const double a = mat_get(A, i, kk);
            const double *b_row = &B->data[(size_t)kk * (size_t)n];
            for (int j = 0; j < n; j++) {
                c_row[j] += a * b_row[j];
            }
        }
    }
    return NULL;
}

static int matmul_pthreads(const Matrix *A, const Matrix *B, Matrix *C, int num_threads) {
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -2;
    if (num_threads < 1) return -3;

    const int m = A->rows;

    //Cap thread count to rows
    if (num_threads > m) num_threads = m;

    pthread_t *threads = (pthread_t*)malloc((size_t)num_threads * sizeof(pthread_t));
    WorkerArgs *args = (WorkerArgs*)malloc((size_t)num_threads * sizeof(WorkerArgs));
    if (!threads || !args) die("malloc threads/args");

    //Split rows as evenly as possible
    int base = m / num_threads;
    int rem = m % num_threads;

    int cur = 0;
    for (int t = 0; t < num_threads; t++) {
        int take = base + (t < rem ? 1 : 0);
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].row_start = cur;
        args[t].row_end = cur + take;
        cur += take;

        int rc = pthread_create(&threads[t], NULL, worker_matmul_rows, &args[t]);
        if (rc != 0) {
            errno = rc;
            die("pthread_create");
        }
    }

    for (int t = 0; t < num_threads; t++) {
        int rc = pthread_join(threads[t], NULL);
        if (rc != 0) {
            errno = rc;
            die("pthread_join");
        }
    }

    free(args);
    free(threads);
    return 0;
}

/*----------------Timing utils----------------*/
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*----------------Test cases------------------*/
static int run_one_test(int m, int k, int n, int threads) {
    Matrix A = mat_alloc(m, k);
    Matrix B = mat_alloc(k, n);
    Matrix C1 = mat_alloc(m, n);
    Matrix C2 = mat_alloc(m, n);

    mat_fill(&A, 123u + (unsigned)m*31u + (unsigned)k*7u + (unsigned)n);
    mat_fill(&B, 999u + (unsigned)m*13u + (unsigned)k*17u + (unsigned)n*19u);

    if (matmul_single(&A, &B, &C1) != 0) return 0;
    if (matmul_pthreads(&A, &B, &C2, threads) != 0) return 0;

    int ok = mat_equal(&C1, &C2, 1e-9);

    mat_free(&A); mat_free(&B); mat_free(&C1); mat_free(&C2);
    return ok;
}

static void run_tests(void) {
    const int cases[][3] = {
        {1, 1, 1},
        {1, 1, 5},
        {2, 1, 3},
        {2, 2, 2},
        {3, 5, 1},
        {5, 3, 7},
        {7, 7, 7},
        {8, 16, 8},
        {31, 17, 29},
        {64, 64, 64},
        {128, 64, 256},
    };

    const int thread_list[] = {1, 2, 4, 8, 16, 32, 64, 128};

    int total = 0, passed = 0;

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        int m = cases[ci][0], k = cases[ci][1], n = cases[ci][2];
        for (size_t ti = 0; ti < sizeof(thread_list)/sizeof(thread_list[0]); ti++) {
            int th = thread_list[ti];
            total++;
            int ok = run_one_test(m, k, n, th);
            if (ok) passed++;
            else {
                printf("[FAIL] m=%d k=%d n=%d threads=%d\n", m, k, n, th);
            }
        }
    }

    srand(42);
    for (int t = 0; t < 50; t++) {
        int m = 1 + rand() % 25;
        int k = 1 + rand() % 25;
        int n = 1 + rand() % 25;
        for (size_t ti = 0; ti < sizeof(thread_list)/sizeof(thread_list[0]); ti++) {
            int th = thread_list[ti];
            total++;
            int ok = run_one_test(m, k, n, th);
            if (ok) passed++;
            else {
                printf("[FAIL] m=%d k=%d n=%d threads=%d\n", m, k, n, th);
            }
        }
    }

    printf("Test: %d %d passed\n", passed, total);
    if (passed != total) exit(2);
}

/*-------------------Benchmark-------------------*/
static void bench(int m , int k, int n) {
    Matrix A = mat_alloc(m, k);
    Matrix B = mat_alloc(k, n);
    Matrix C = mat_alloc(m, n);

    mat_fill(&A, 123);
    mat_fill(&B, 456);

    matmul_pthreads(&A, &B, &C, 1);

    const int thread_list[] = {1, 4, 16, 32, 64, 128};
    double t1 = 0.0;

    printf("Benchmark m=%d k=%d n=%d\n", m, k, n);
    for (size_t i = 0; i < sizeof(thread_list)/sizeof(thread_list[0]); i++) {
        int th = thread_list[i];

        double best = 1e100;
        for (int rep = 0; rep < 3; rep++) {
            double t0 = now_sec();
            int rc = matmul_pthreads(&A, &B, &C, th);
            double t = now_sec() - t0;
            if (rc != 0) {
                printf("matmul_pthreads failed rc=%d\n", rc);
                exit(3);
            }
            if (t < best) best = t;
        }

        if (th == 1) t1 = best;
        double speedup = t1 / best;
        printf("threads=%3d time=%.6f s speedup=%.2fx\n", th, best, speedup);
        fflush(stdout);
    }
    mat_free(&A); mat_free(&B); mat_free(&C);
}

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("   %s --test\n", prog);
    printf("   %s --bench m k n \n", prog);
    printf("\nExample: \n");
    printf("   %s --bench 2048 2048 2048\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--test") == 0) {
        run_tests();
        return 0;
    }
    if (strcmp(argv[1], "--bench") == 0) {
        if (argc != 5) { usage(argv[0]); return 1;}
        int m = atoi(argv[2]);
        int k = atoi(argv[3]);
        int n = atoi(argv[4]);
        if (m <= 0 || k <= 0 || n <= 0) {
            fprintf(stderr, "m,k,n must be positive\n");
            return 1;
        }
        bench(m, k, n);
        return 0;        
    }
    usage(argv[0]);
    return 1;
}