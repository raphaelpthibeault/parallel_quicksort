#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

void swap(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

void quicksort(int* arr, int start, int end) {
    if (start >= end)
        return;

    int mid = start + (end - start) / 2;
    int pivot = arr[mid];
    swap(arr, start, mid);

    int i = start + 1, j = end;
    while (i <= j) {
        while (i <= j && arr[i] <= pivot) i++;
        while (i <= j && arr[j] > pivot) j--;
        if (i < j) {
            swap(arr, i, j);
            i++;
            j--;
        }
    }
    swap(arr, start, j);

    quicksort(arr, start, j - 1);
    quicksort(arr, j + 1, end);
}

int* merge(int* arr1, int n1, int* arr2, int n2) {
    int* res = (int*)malloc((n1 + n2) * sizeof(int));
    int i = 0, j = 0, k = 0;

    while (i < n1 && j < n2) {
        if (arr1[i] < arr2[j]) {
            res[k++] = arr1[i++];
        } else {
            res[k++] = arr2[j++];
        }
    }

    while (i < n1) res[k++] = arr1[i++];
    while (j < n2) res[k++] = arr2[j++];

    return res;
}

int main(int argc, char* argv[]) {
    int N, P, rank, rc;
    int *arr = NULL, *subcube = NULL;
    int subcube_size, own_subcube_size;
    FILE* f = NULL;
    MPI_Status status;

    rc = MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    if (rc != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(-1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (argc != 3) {
            printf("Usage: %s <input file> <output file>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(-1);
        }

        f = fopen(argv[1], "r");
        if (f == NULL) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(-1);
        }

        fscanf(f, "%d", &N);
        subcube_size = (N + P - 1) / P;
        arr = (int*)malloc(sizeof(int) * P * subcube_size);

        for (int i = 0; i < N; i++) {
            fscanf(f, "%d", &arr[i]);
        }

        for (int i = N; i < P * subcube_size; i++) {
            arr[i] = INT_MAX;
        }

        fclose(f);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    subcube_size = (N + P - 1) / P;
    subcube = (int*)malloc(sizeof(int) * subcube_size);

    MPI_Scatter(arr, subcube_size, MPI_INT, subcube, subcube_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(arr);
    arr = NULL;

    own_subcube_size = (N >= subcube_size * (rank + 1)) ? subcube_size : (N - subcube_size * rank);
    quicksort(subcube, 0, own_subcube_size - 1);

    for (int step = 1; step < P; step *= 2) {
        if (rank % (2 * step) != 0) {
            MPI_Send(subcube, own_subcube_size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            break;
        }

        if (rank + step < P) {
            int received_chunk_size = (N >= subcube_size * (rank + 2 * step)) ? (subcube_size * step): (N - subcube_size * (rank + step));
            int* chunk_received = (int*)malloc(received_chunk_size * sizeof(int));
            MPI_Recv(chunk_received, received_chunk_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);

            int* merged_arr = merge(subcube, own_subcube_size, chunk_received, received_chunk_size);
            free(subcube);
            free(chunk_received);
            subcube = merged_arr;
            own_subcube_size += received_chunk_size;
        }
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        f = fopen(argv[2], "w");
        if (f == NULL) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(-1);
        }

        fprintf(f, "Elapsed time: %f seconds\n", elapsed_time);
        fprintf(f, "Number of processors: %d\n", P);
        fprintf(f, "Sorted array:\n");
        for (int i = 0; i < own_subcube_size; i++) {
            if (subcube[i] != INT_MAX) {
                fprintf(f, "%d ", subcube[i]);
            }
        }

        fclose(f);
    }

    MPI_Finalize();
    return 0;
}
