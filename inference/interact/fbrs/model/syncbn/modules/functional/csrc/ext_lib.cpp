#include "bn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("syncbn_sum_sqsum", &syncbn_sum_sqsum, "Sum and Sum^2 computation");
  m.def("syncbn_forward", &syncbn_forward, "SyncBN forward computation");
  m.def("syncbn_backward_xhat", &syncbn_backward_xhat,
        "First part of SyncBN backward computation");
  m.def("syncbn_backward", &syncbn_backward,
        "Second part of SyncBN backward computation");
}