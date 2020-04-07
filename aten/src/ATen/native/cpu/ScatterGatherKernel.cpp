#include <iostream>
#include <utility>
#include <vector>

#include <ATen/native/ScatterGatherShapeChecks.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>

namespace at { namespace native {

namespace {

enum class LoopSpecialization {
  ONE_DIMENSIONAL,
  ONE_DIMENSIONAL_CONTIGUOUS,
  BATCH_MAJOR,
  BATCH_MAJOR_CONTIGUOUS,
  FEATURE_MAJOR,
};

template <typename scalar_t, bool is_scatter_like>
struct _loop_helper {
  int64_t self_dim_stride;
  int64_t self_dim_size;

  int64_t index_dim_stride;
  int64_t index_dim_size;

  int64_t src_dim_stride;
  int64_t src_dim_size;

  int64_t dim;
  int64_t index_upper_bound;
  bool serial_exec;

  scalar_t* self_ptr;
  int64_t* index_ptr;
  scalar_t* src_ptr;

  int64_t force_heuristic;

  LoopSpecialization choose_specialization(
      const bool vector_subtask,
      const bool contiguous_subtask,
      const int64_t n) {
    if (vector_subtask) {
      return contiguous_subtask ? LoopSpecialization::ONE_DIMENSIONAL_CONTIGUOUS
                                : LoopSpecialization::ONE_DIMENSIONAL;
    }

    if (contiguous_subtask) return LoopSpecialization::BATCH_MAJOR_CONTIGUOUS;

    if (force_heuristic == 0) return LoopSpecialization::FEATURE_MAJOR;
    if (force_heuristic == 1) return LoopSpecialization::BATCH_MAJOR;

    return ((self_dim_stride == 1) || (n < index_dim_size))
      ? LoopSpecialization::FEATURE_MAJOR
      : LoopSpecialization::BATCH_MAJOR;
  }

  template <typename func_t>
  void batch_major(
    const func_t& f,
    const int64_t n,
    const int64_t self_iter_stride,
    const int64_t index_iter_stride,
    const int64_t src_iter_stride
  ) {

    run_loop(n, [&](int64_t start, int64_t end){
      for (int64_t i = start; i < end; ++i) {
        auto* self_data = self_ptr;
        auto* index_data = index_ptr + i * index_dim_stride;
        auto* src_data = src_ptr;
        for (int64_t nelem = 0; nelem < n; ++nelem) {
          int64_t idx_dim = *index_data;
          // we are not  pulling this check into a helper function because
          // it disables loop optimization in clang-7
          TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
            "index ", *index_data,
            " is out of bounds for dimension ", dim,
            " with size ", index_upper_bound
          );

          f(
            self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
            src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
          );

          self_data += self_iter_stride;
          index_data += index_iter_stride;
          src_data += src_iter_stride;
        }
      }
    });
  }

  template <typename func_t>
  void batch_major_contiguous(
    const func_t& f,
    const int64_t n
  ) {
    run_loop(n, [&](int64_t start, int64_t end){
      for (int64_t i = start; i < end; ++i) {
        int64_t idx_dim = *(index_ptr + i * index_dim_stride);
        // we are not putting idx_dim in the error message or pulling this check into
        // a helper function because it disables loop optimization in clang-7
        TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
          "index ", index_ptr[i * index_dim_stride],
          " is out of bounds for dimension ", dim,
          " with size ", index_upper_bound
        );

        auto* self_data = self_ptr + (is_scatter_like ? idx_dim : i) * self_dim_stride;
        auto* src_data = src_ptr + (is_scatter_like ? i : idx_dim) * src_dim_stride;

        for (int64_t nelem = 0; nelem < n; ++nelem) {
          f(self_data, src_data);
          ++self_data;
          ++src_data;
        }
      }
    });
  }

  template <typename func_t>
  void feature_major(
    const func_t& f,
    const int64_t n,
    const int64_t self_iter_stride,
    const int64_t index_iter_stride,
    const int64_t src_iter_stride
  ) {

    run_loop(n, [&](int64_t start, int64_t end){
      auto* self_data = self_ptr;
      auto* index_data = index_ptr;
      auto* src_data = src_ptr;
      for (int64_t nelem = 0; nelem < n; ++nelem) {
        // TODO(taylorrobie): measure if this needs to be extracted like `_cpu_scatter_gather_dim_loop`
        for (int64_t i = start; i < end; ++i) {
          int64_t idx_dim = index_data[i * index_dim_stride];
          // we are not putting idx_dim in the error message or pulling this check into
          // a helper function because it disables loop optimization in clang-7
          TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
            "index ", index_data[i * index_dim_stride],
            " is out of bounds for dimension ", dim,
            " with size ", index_upper_bound
          );

          f(
            self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
            src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
          );
        }

        self_data += self_iter_stride;
        index_data += index_iter_stride;
        src_data += src_iter_stride;
      }
    });
  }

  template <typename loop_func_t>
  void run_loop(int64_t n, loop_func_t inner_loop){
    auto grain_size = at::internal::GRAIN_SIZE / n;
    (serial_exec || index_dim_size < grain_size)
        ? inner_loop(0, index_dim_size)
        : at::parallel_for(0, index_dim_size, grain_size, inner_loop);
  }
};

template <bool broadcast_index, bool is_scatter_like>
struct cpu_scatter_gather_base_kernel_new {
  template <typename func_factory_t>
  void operator()(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const std::string& method_name,
    const func_factory_t& make_f,
    bool serial_exec = true,
    const int64_t force_heuristic = -1
  ) {
    auto self_dim = self.dim();
    dim = maybe_wrap_dim(dim, self_dim);

    TORCH_CHECK(dim == 0 || dim < self_dim, method_name, "(): Indexing dim ", dim, " is out of bounds of tensor");
    TORCH_CHECK(index.scalar_type() == ScalarType::Long, method_name, "(): Expected dtype int64 for index");
    TORCH_CHECK(self.scalar_type() == src.scalar_type(), method_name, "(): self and result must have the same scalar type");

    if (broadcast_index){
      TORCH_CHECK(index.is_contiguous(), "Internal error: when calling scatter/gather from index_* methods, "
                                         "index.contiguous() was not called.")
      TORCH_CHECK_INDEX(index.dim() <= 1, method_name, "(): Index is supposed to be a vector");
      TORCH_CHECK(is_scatter_like ? index.numel() > ensure_nonempty_size(src, dim) : true,
                  method_name, "(): Index size ", index.numel(), " does not match source size ",
                  ensure_nonempty_size(src, dim), " along dim ", dim, ".");
      // TODO: This is not complete as it does not check `self` and `src` agreement for the scatter (index_put) case.
      //       May be better to just reuse `scatter_shape_check`.
    } else {
      is_scatter_like ? scatter_shape_check(self, dim, index, src)
                      : gather_shape_check(self, dim, index, src);
    }

    auto iter = TensorIterator();
    iter.dont_compute_common_dtype();
    iter.dont_resize_outputs();
    iter.declare_static_shape(self.sizes(), /*squash_dim=*/dim);
    iter.add_output(self);
    iter.add_input(src, src.device(), src.scalar_type());
    if (!broadcast_index) iter.add_input(index);
    iter.build();

    const auto self_dim_stride = ensure_nonempty_stride(self, dim);
    const auto self_dim_size = ensure_nonempty_size(self, dim);

    const auto index_dim_stride = broadcast_index ? 1  : ensure_nonempty_stride(index, dim);
    const auto index_dim_size = broadcast_index ? index.numel() : ensure_nonempty_size(index, dim);

    const auto src_dim_stride = ensure_nonempty_stride(src, dim);
    const auto src_dim_size = ensure_nonempty_size(src, dim);

    AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, iter.dtype(),
      method_name, [&] {
        const auto slice_size = iter.numel();
        const bool vector_subtask = (slice_size == 1);
        int64_t* raw_index_ptr = broadcast_index ? index.data_ptr<int64_t>() : nullptr;
        const auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;
        const auto f = make_f((scalar_t*)(nullptr));

        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          constexpr auto SELF_ITER_STRIDE_IDX = 0;
          constexpr auto INDEX_ITER_STRIDE_IDX = 2;
          constexpr auto SRC_ITER_STRIDE_IDX = 1;

          // We convert from char* to scalar_t and int64_t to make it easier for the
          // compiler to vectorize certain tight loops, as well as to improve readability.
          TORCH_INTERNAL_ASSERT(!(strides[SELF_ITER_STRIDE_IDX] % sizeof(scalar_t)));
          TORCH_INTERNAL_ASSERT(!(strides[INDEX_ITER_STRIDE_IDX] % sizeof(int64_t)));
          TORCH_INTERNAL_ASSERT(!(strides[SRC_ITER_STRIDE_IDX] % sizeof(scalar_t)));

          auto* self_ptr = (scalar_t*)(data[SELF_ITER_STRIDE_IDX]);
          auto* index_ptr = broadcast_index ? raw_index_ptr : (int64_t*)(data[INDEX_ITER_STRIDE_IDX]);
          auto* src_ptr = (scalar_t*)(data[SRC_ITER_STRIDE_IDX]);

          auto self_iter_stride = strides[SELF_ITER_STRIDE_IDX] / sizeof(scalar_t);
          auto index_iter_stride = broadcast_index ? 0 : strides[INDEX_ITER_STRIDE_IDX] / sizeof(int64_t);
          auto src_iter_stride = strides[SRC_ITER_STRIDE_IDX] / sizeof(scalar_t);

          const bool contiguous_subtask = (
            !index_iter_stride && self_iter_stride == 1 &&
            src_iter_stride == 1);

          _loop_helper<scalar_t, is_scatter_like> loop_helper({
            self_dim_stride, self_dim_size,
            index_dim_stride, index_dim_size,
            src_dim_stride, src_dim_size,
            dim, index_upper_bound, serial_exec,
            self_ptr, index_ptr, src_ptr,
            force_heuristic
          });

          switch (loop_helper.choose_specialization(
              vector_subtask, contiguous_subtask, n)) {
            case LoopSpecialization::ONE_DIMENSIONAL_CONTIGUOUS:
              TORCH_INTERNAL_ASSERT(n == 1);
              loop_helper.batch_major_contiguous(f, /*n=*/1);
              break;
            case LoopSpecialization::ONE_DIMENSIONAL:
              TORCH_INTERNAL_ASSERT(n == 1);
              loop_helper.batch_major(
                  f, /*n=*/1, self_iter_stride,
                  index_iter_stride, src_iter_stride);
              break;
            case LoopSpecialization::BATCH_MAJOR_CONTIGUOUS:
              loop_helper.batch_major_contiguous(f, n);
              break;
            case LoopSpecialization::BATCH_MAJOR:
              loop_helper.batch_major(
                  f, n, self_iter_stride,
                  index_iter_stride, src_iter_stride);
              break;
            case LoopSpecialization::FEATURE_MAJOR:
              loop_helper.feature_major(
                  f, n, self_iter_stride,
                  index_iter_stride, src_iter_stride);
              break;
            default:
              TORCH_INTERNAL_ASSERT(false, "Unsupported specialization")
          }
        };

        serial_exec ? iter.serial_for_each(loop, {0, iter.numel()})
                    : iter.for_each(loop);
      }
    );
  }
}; // struct cpu_scatter_gather_base_kernel_new


static inline auto make_assign_f = [](auto*){  // Argument is strictly to infer scalar_t
  return [](auto* lhs, const auto* rhs) { *lhs = *rhs; };
};

static inline auto make_assign_add_f = [](auto*){  // Argument is strictly to infer scalar_t
  return [](auto* lhs, const auto* rhs) { *lhs += *rhs; };
};

// TODO(taylorrobie): Can we reduce the boilerplate?
void gather_cpu_kernel_new(
    Tensor& result, const Tensor& self, int64_t dim,
    const Tensor& index, int64_t force_heuristic = -1) {
  cpu_scatter_gather_base_kernel_new</*broadcast_index=*/false, /*is_scatter_like=*/false>()(
      result, dim, index, self, "index_select_out_cpu", make_assign_f,
      /*serial_exec=*/false, force_heuristic
  );
}

void scatter_cpu_kernel_new(
    Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
    int64_t force_heuristic = -1) {
  cpu_scatter_gather_base_kernel_new</*broadcast_index=*/false, /*is_scatter_like=*/true>()(
    self, dim, index, src, "scatter_cpu_", make_assign_f,
    /*serial_exec=*/false, force_heuristic
  );
}

void scatter_fill_cpu_kernel_new(Tensor& self, int64_t dim, const Tensor& index, Scalar src, int64_t force_heuristic = -1) {
  auto make_assign_fill_f = [src](auto* _){
    using scalar_t = typename std::remove_pointer<decltype(_)>::type;
    scalar_t fill_value = src.to<scalar_t>();
    return [fill_value](auto* lhs, const auto*) { *lhs = fill_value; };
  };

  cpu_scatter_gather_base_kernel_new</*broadcast_index=*/false, /*is_scatter_like=*/true>()(
    self, dim, index, self, "scatter_fill_cpu_", make_assign_fill_f,
    /*serial_exec=*/false, force_heuristic
  );
}

void scatter_add_cpu_kernel_new(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
                            int64_t force_heuristic = -1) {
  cpu_scatter_gather_base_kernel_new</*broadcast_index=*/false, /*is_scatter_like=*/true>()(
    self, dim, index, src, "scatter_add_cpu_", make_assign_add_f,
    /*serial_exec=*/true, force_heuristic
  );
}

void index_select_cpu_kernel_new(
    Tensor& result, const Tensor& self, int64_t dim,
    const Tensor& index, int64_t force_heuristic = -1) {
  cpu_scatter_gather_base_kernel_new</*broadcast_index=*/true, /*is_scatter_like=*/false>()(
      result, dim, index, self, "index_select_out_cpu", make_assign_f,
      /*serial_exec=*/false, force_heuristic
  );
}

void index_put_cpu_kernel_new(
    Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
    int64_t force_heuristic = -1) {
  cpu_scatter_gather_base_kernel_new</*broadcast_index=*/true, /*is_scatter_like=*/true>()(
    self, dim, index, src, "index_put_cpu_", make_assign_f,
    /*serial_exec=*/false, force_heuristic
  );
}



} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_cpu_kernel_new);
REGISTER_DISPATCH(scatter_stub, &scatter_cpu_kernel_new);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cpu_kernel_new);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel_new);

REGISTER_DISPATCH(index_select_kernel_stub, &index_select_cpu_kernel_new);


}} // namespace at::native
