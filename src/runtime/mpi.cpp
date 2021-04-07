#include "runtime_internal.h"
#include "HalideRuntimeMPI.h"
#include <mpi.h>
#include "printer.h"

extern "C" {

WEAK int halide_mpi_num_processors() {
    int num_processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    return num_processes;
}

WEAK int halide_mpi_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

WEAK bool halide_mpi_send(void *data,
                          int offset,
                          size_t num_bytes,
                          int destination,
                          int tag) {
    print(0) << "send, offset:" << offset << ", num_bytes:" << (int)num_bytes << ", dest:" << destination << ", data[0]:" << ((int*)data)[0] << "\n";
    MPI_Send((void*)((uint8_t*)data + offset), num_bytes, MPI_BYTE, destination, tag, MPI_COMM_WORLD);
    return true;
}

WEAK bool halide_mpi_recv(void *data,
                          int offset,
                          size_t num_bytes,
                          int source,
                          int tag) {
    MPI_Recv((void*)((uint8_t*)data + offset), num_bytes, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    print(0) << "recv, offset:" << offset << ", num_bytes:" << (int)num_bytes << ", src:" << source << ", data[0]:" << ((int*)data)[0] << "\n";
    return true;
}

} // extern "C"
