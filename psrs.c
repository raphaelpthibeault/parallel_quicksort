#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

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

int *find_pos(int *base, int size, int x) {
    int *low_bound = base;
    int *upper_bound = base + size - 1;

    while (low_bound <= upper_bound) {
        int *mid = low_bound + (upper_bound - low_bound) / 2;
        if (x < *mid)
            upper_bound = mid - 1;
        else
            low_bound = mid + 1;
    }

    return low_bound;
}

int main(int argc, char** argv) {
    int N, P, rank, rc, *arr = NULL;
    FILE* f = NULL;

    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        exit(-1);
    }

    f = fopen(argv[1], "r");
    if (f == NULL) {
        printf("Error opening file %s\n", argv[1]);
        exit(-1);
    }

    fscanf(f, "%d", &N);
    arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        fscanf(f, "%d", &arr[i]);
    }
    fclose(f);

    rc = MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();

    if (rc != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(-1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int segment_size = ceil((double)N / P);
    int* segment = (int*)malloc(segment_size * sizeof(int));

    MPI_Scatter(arr, segment_size, MPI_INT, segment, segment_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(arr);

    quicksort(segment, 0, segment_size - 1);

    int num_samples = P;
    int sample_dist = segment_size / (num_samples + 1);
    int* sample_buf = (int*)malloc(num_samples * sizeof(int));
    for (int i = 0; i < num_samples; i++) {
        sample_buf[i] = segment[(i + 1) * sample_dist];
    }

    int* all_samples = (int*)malloc(P * num_samples * sizeof(int));
    MPI_Gather(sample_buf, num_samples, MPI_INT, all_samples, num_samples, MPI_INT, 0, MPI_COMM_WORLD);
    free(sample_buf);
    int* pivots = (int*)malloc((P - 1) * sizeof(int));

    if (rank == 0) {
        quicksort(all_samples, 0, P * num_samples - 1);
        for (int i = 0; i < P - 1; i++) {
            pivots[i] = all_samples[(i + 1) * num_samples];
        }
        free(all_samples);
    }

    MPI_Bcast(pivots, P - 1, MPI_INT, 0, MPI_COMM_WORLD);

    int** partition_around = (int**)malloc((P + 1) * sizeof(int*));
    partition_around[0] = segment;
    for (int i = 1; i < P; i++) {
        partition_around[i] = find_pos(segment, segment_size, pivots[i - 1]);
    }
    partition_around[P] = segment + segment_size;

    int* send_counts = (int*)malloc(P * sizeof(int));
    int* recv_counts = (int*)malloc(P * sizeof(int));
    for (int i = 0; i < P; i++) {
        send_counts[i] = max(0, partition_around[i + 1] - partition_around[i]);
    }

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int* send_off = (int*)malloc(P * sizeof(int));
    int* recv_off = (int*)malloc(P * sizeof(int));
    send_off[0] = 0;
    recv_off[0] = 0;
    for (int i = 1; i < P; i++) {
        send_off[i] = send_off[i - 1] + send_counts[i - 1];
        recv_off[i] = recv_off[i - 1] + recv_counts[i - 1];
    }

    int trecv = recv_off[P - 1] + recv_counts[P - 1];
    int* recv_buf = (int*)malloc(trecv * sizeof(int));

    MPI_Alltoallv(segment, send_counts, send_off, MPI_INT, recv_buf, recv_counts, recv_off, MPI_INT, MPI_COMM_WORLD);

    int **list_ptr, *list_count;
    list_ptr = (int**)malloc(P * sizeof(int*));
    list_count = (int*)malloc(P * sizeof(int));

    for (int i = 0; i < P; i++) {
        list_ptr[i] = recv_buf + recv_off[i];
        list_count[i] = recv_counts[i];
    }

    int* sorted = (int*)malloc(N * sizeof(int));
    int sorted_size = 0;
    while (sorted_size < trecv) {
        int min_elem = INT_MAX, min_list = -1;
        for (int j = 0; j < P; j++) {
            if (list_count[j] > 0 && *(list_ptr[j]) < min_elem) {
                min_elem = *(list_ptr[j]);
                min_list = j;
            }
        }
        sorted[sorted_size++] = min_elem;
        list_ptr[min_list]++;
        list_count[min_list]--;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int *final_arr = NULL;
    int *all_recv_counts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        final_arr = (int*)malloc(N * sizeof(int));
        all_recv_counts = (int*)malloc(P * sizeof(int));
        displs = (int*)malloc(P * sizeof(int));
    }

    MPI_Gather(&sorted_size, 1, MPI_INT, all_recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < P; i++) {
            displs[i] = displs[i - 1] + all_recv_counts[i - 1];
        }
    }

    MPI_Gatherv(sorted, sorted_size, MPI_INT, final_arr, all_recv_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed_time = MPI_Wtime() - start_time;
        f = fopen(argv[2], "w");
        fprintf(f, "Elapsed time: %f seconds\n", elapsed_time);
        fprintf(f, "Number of processors: %d\n", P);
        fprintf(f, "Sorted array:\n");
        for (int k = 0; k < N; k++) {
            fprintf(f, "%d ", final_arr[k]);
        }
        fclose(f);
    }

    free(list_count);
    free(list_ptr);
    free(recv_buf);
    free(recv_off);
    free(send_off);
    free(recv_counts);
    free(send_counts);
    free(partition_around);
    free(segment);
    free(sorted);
    if (rank == 0) {
        free(final_arr);
        free(all_recv_counts);
        free(displs);
    }
    MPI_Finalize();
}
