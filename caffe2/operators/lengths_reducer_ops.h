#pragma once
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/perfkernels/embedding_lookup.h"
#include <algorithm>
#include <functional>

namespace caffe2 {

// A templated class that implements SparseLengths[Sum,WeightedSum,Mean].
template <
    typename T, // output type
    class InputTypes, // supported input types, such as TensorTypes<float>
    bool USE_WEIGHT = 0, // Whether it is SparseLengthsWeightedSum
    bool USE_MEAN = 0, // Whether this is SparseLengthsMean
    bool USE_POSITIONAL_WEIGHT = 0
    // USE_WEIGHT = 1 and USE_POSITIONAL_WEIGHT = 1
    // -> SparseLengthsPositionalWeightedSum
    >
class CPUSparseLengthsReductionOp : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  template <class... Args>
  explicit CPUSparseLengthsReductionOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {
    static_assert(
        !(USE_WEIGHT & USE_MEAN), "Cannot both specify weight and mean.");
  }

  ~CPUSparseLengthsReductionOp() {}

  // Currently, we support float and at::Half inputs for input data type, and
  // int32_t and int64_t for the index type.

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(DATA));
  }

  template <typename InputType>
  bool DoRunWithType() {
    return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
        this, Input(INDICES));
  }

  template <typename InputType, typename IndexType>
  bool DoRunWithType2() {
    auto& dataInput = Input(DATA);
    auto& indicesInput = Input(INDICES);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t N = dataInput.size(0);
    const int D = dataInput.size_from_dim(1);
    const int64_t M = lengthsInput.size(0);
    const int64_t indices_size = indicesInput.numel();

    auto shape = dataInput.sizes().vec();
    shape[0] = M;
    auto* output = Output(0, shape, at::dtype<T>());
    T* out_data = output->template mutable_data<T>();

    const InputType* in_data = dataInput.template data<InputType>();
    const IndexType* indices = indicesInput.template data<IndexType>();
    const int* lengths = lengthsInput.template data<int>();
    const T* in_weight = nullptr;

    if (USE_WEIGHT) {
      // static if
      auto& weightInput = Input(WEIGHT);
      CAFFE_ENFORCE_EQ(1, weightInput.dim(), "WEIGHT must be a vector");
      if (!USE_POSITIONAL_WEIGHT) {
        CAFFE_ENFORCE_EQ(
            weightInput.numel(),
            indices_size,
            "Weight should have the same length as indices.");
      }
      in_weight = weightInput.template data<T>();
    }

    // delegate work to perfkernel that branches based on architecture
    EmbeddingLookup<IndexType, InputType, T, USE_POSITIONAL_WEIGHT>(
        D,
        M,
        indices_size,
        N,
        in_data,
        indices,
        lengths,
        in_weight,
        nullptr, // scale_bias field is only used in SparseLengths8BitsRowwiseOp
        USE_MEAN,
        out_data);
    return true;
  }

  enum {
    DATA = 0, // Data input.
    WEIGHT = 1, // Weight input used in SparseLengthsWeightedSum
    INDICES = 1 + USE_WEIGHT, // 1 in SparseLengths[Sum,Mean] and
                              // 2 in SparseLengthsWeightedSum
    LENGTHS = 2 + USE_WEIGHT, // 2 in SparseLengths[Sum, Mean],
                              // 3 in SparseLengthsWeightedSum
  };
};

template <typename T, class Context, class Engine = DefaultEngine>
class TTSparseLengthsSumOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit TTSparseLengthsSumOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        factor_i(this->template GetRepeatedArgument<int>(
                  "factor_i", vector<int>{1, 1, 1})),
        factor_j(this->template GetRepeatedArgument<int>(
                  "factor_j", vector<int>{1, 1, 1})),
        ranks(this->template GetRepeatedArgument<int>(
                  "ranks", vector<int>{1, 1, 1, 1})),
        emb_size(this->template GetSingleArgument<int>(
                  "emb_size", 64)){
    //cumprod of i, used for index slice
    l_cumprod.push_back(1);
    for (size_t i = 1; i < factor_i.size(); ++i) {
      l_cumprod.push_back(l_cumprod[i-1]*factor_i[i-1]);
    }
  }

  ~TTSparseLengthsSumOp() {}

  void Ind2Sub(int64_t* out_factor_index, const int64_t* indices, int len) {
    //TODO: vectorization
    auto N = factor_i.size();
    for (int j = 0; j < len; j++) {
      auto idx = indices[j];
      for (int i = N; i > 0; i--) {
        out_factor_index[j * N + i - 1] = idx / l_cumprod[i-1];
        idx = idx % l_cumprod[i-1];
      }
    }
  }

  bool GetSlice(std::vector<std::vector<T>>& tgt_slice, const T* core, const vector<int64_t>& ind_slice, int bs, int idx) {
    //implement the functinality index_select(core, 1, ind_slice)
    auto num_of_elements = ranks[idx]*factor_j[idx]*ranks[idx+1];
    for (int i = 0; i < bs; i++) {
      memcpy(tgt_slice[i].data(), core+ind_slice[i]*num_of_elements, num_of_elements*sizeof(T));
    }
    return true;
  }

  bool GatherAllRows(int64_t* ind,
      int bs,
      int x_len,
      vector<const T*> cores,
      int segments,
      const int* lengths,
      T* out_data) {

      // compute the largest memory consumption of intermediate result
      // TODO: dynamic allocation size: cur_rows*factor_j[i]*ranks[i+1]
      // and also explore the contiguous memory storage for res and int_res
      int max_rank = *max_element(ranks.begin(), ranks.end());

      std::vector<std::vector<T>> res(bs, std::vector<T>(emb_size*max_rank, 0));
      std::vector<std::vector<T>> int_res(bs, std::vector<T>(emb_size*max_rank, 0));

      // Store the matrix A
      vector<T*> Y_ptr(bs);
      // Store the intermediate result in each layer
      vector<T*> Z_ptr(bs);

      for(int b = 0; b < bs; b++){
        Y_ptr[b] = res[b].data();
        Z_ptr[b] = int_res[b].data();
      }

      vector<int64_t> ind_slice(bs);
      int rows = 0;
      for (int i = 0; i < x_len; i++) {
        //slice cur
        for (int j = 0; j < bs; j++){
          ind_slice[j] = ind[x_len*j + i];
        }
        if (i == 0){
          GetSlice(res, cores[i], ind_slice, bs, i);
          rows = factor_j[0];
        } else {
          std::vector<std::vector<T>> slice(bs, std::vector<T>(ranks[i]*factor_j[i]*ranks[i+1], 0));
          vector<const T*> X_ptr(bs);
          for(int b = 0; b < bs; b++){
            X_ptr[b] = slice[b].data();
          }
          GetSlice(slice, cores[i], ind_slice, bs, i);

          math::GemmBatched<T, CPUContext>(
              CblasNoTrans,
              CblasNoTrans,
              bs,
              rows,
              factor_j[i]*ranks[i+1],
              ranks[i],
              1.0f,
              // ((i%2)? const_cast<const T**>(Y_ptr.data()) : const_cast<const T**>(Z_ptr.data())),
              const_cast<const T**>(Y_ptr.data()),
              X_ptr.data(),
              0.0f,
              Z_ptr.data(),
              // ((i%2)? Z_ptr.data() : Y_ptr.data()),
              &context_
              );
          for(int b = 0; b < bs; b++){
            std::memcpy(Y_ptr[b], Z_ptr[b], (emb_size*max_rank)*sizeof(T));
          }
          rows *= factor_j[i];
        }
        //save the intermediate output for backward path
        // shape for the core
        auto shape = vector<int64_t>({bs, rows, ranks[i+1]});
        auto* core_data = Output(i+1, shape, at::dtype<T>());
        T* out_core = core_data->template mutable_data<T>();
        for(int b = 0; b < bs; b++){
          std::memcpy(out_core+b*rows*ranks[i+1], Y_ptr[b], rows*ranks[i+1]*sizeof(T));
        }
      }

      // reduction and store back to output
      vector<int64_t> cum_lengths(segments);
      for (int seg = 0; seg < segments; seg++){
        cum_lengths[seg] = seg == 0 ? lengths[0] : lengths[seg] + cum_lengths[seg-1];
      }

      int length_idx = 0;
      vector<T> tmp_sum(emb_size, 0.0f);
      for (int i = 0; i <= bs; i++) {
        if (i == cum_lengths[length_idx]){
          //store the tmp_sum into output
          memcpy(&out_data[length_idx*emb_size], tmp_sum.data(), emb_size*sizeof(T));
          if (i == bs) {
            break;
          }
          length_idx++;
          //reset the tmp_sum;
          fill(tmp_sum.begin(), tmp_sum.end(), 0.0f);
        }
        transform(res[i].begin(), res[i].begin() + emb_size, tmp_sum.begin(), tmp_sum.begin(), std::plus<T>());
      }
      return true;
    }

  bool RunOnDevice() override {
    const auto& dataInput0 = Input(0);
    const auto& dataInput1 = Input(1);
    const auto& dataInput2 = Input(2);
    const auto& indicesInput = Input(3);
    const auto& lengthsInput = Input(4);

    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");

    int N = factor_i.size();
    const int64_t M = lengthsInput.size(0);

    auto shape = vector<int64_t>({M, emb_size});
    auto* output = Output(0, shape, at::dtype<T>());
    T* out_data = output->template mutable_data<T>();

    const T* core0 = dataInput0.template data<T>();
    const T* core1 = dataInput1.template data<T>();
    const T* core2 = dataInput2.template data<T>();

    const int* lengths = lengthsInput.template data<int>();

    vector<const T*> cores = {core0, core1, core2};

    const int64_t* indices = indicesInput.template data<int64_t>();

    // Store the factor index for backward path
    auto index_shape = vector<int64_t>({indicesInput.size(), N});
    auto* index_data = Output(4, index_shape, at::dtype<int64_t>());
    int64_t* out_factor_index = index_data->template mutable_data<int64_t>();

    // Store the factorized index for each core
    Ind2Sub(out_factor_index, indices, indicesInput.size());

    return GatherAllRows(
        out_factor_index,
        indicesInput.size(),
        N,
        cores,
        M,
        lengths,
        out_data);
  }

 protected:
   vector<int> factor_i;
   vector<int> factor_j;
   vector<int> ranks;
   vector<int> l_cumprod;
   int emb_size;
};

// template <
//     typename T, // output type
//     class InputTypes, // supported input types, such as TensorTypes<float>
//     bool USE_WEIGHT = 0, // Whether it is SparseLengthsWeightedSum
//     bool USE_MEAN = 0, // Whether this is SparseLengthsMean
//     bool USE_POSITIONAL_WEIGHT = 0
//     // USE_WEIGHT = 1 and USE_POSITIONAL_WEIGHT = 1
//     // -> SparseLengthsPositionalWeightedSum
//     >
// class CPUTTSparseLengthsReductionOp : public Operator<CPUContext> {
//  public:
//   USE_OPERATOR_FUNCTIONS(CPUContext);
//   template <class... Args>
//   explicit CPUTTSparseLengthsReductionOp(Args&&... args)
//       : Operator<CPUContext>(std::forward<Args>(args)...),
//         factor_i(this->template GetRepeatedArgument<int>(
//                     "factor_i", vector<int>{1, 1, 1})),
//         factor_j(this->template GetRepeatedArgument<int>(
//                     "factor_j", vector<int>{1, 1, 1})),
//         ranks(this->template GetRepeatedArgument<int>(
//                     "ranks", vector<int>{1, 1, 1})),
//         emb_size(this->template GetSingleArgument<int>(
//                     "emb_size", 64)){
//     static_assert(
//         !(USE_WEIGHT & USE_MEAN), "Cannot both specify weight and mean.");
//
//     //cumprod of i, used for index slice
//     l_cumprod.push_back(1);
//     for (size_t i = 1; i < factor_i.size(); ++i) {
//       l_cumprod.push_back(l_cumprod[i-1]*factor_i[i-1]);
//     }
//   }
//
//   ~CPUTTSparseLengthsReductionOp() {}
//
//   // Currently, we support float and at::Half inputs for input data type, and
//   // int32_t and int64_t for the index type.
//
//   bool RunOnDevice() override {
//     return DispatchHelper<InputTypes>::call(this, Input(CORE0));
//   }
//
//   template <typename IndexType>
//   void Ind2Sub(IndexType* out_factor_index, const IndexType* indices, int len) {
//     //TODO: vectorization
//     auto N = factor_i.size();
//     for (int j = 0; j < len; j++) {
//       auto idx = indices[j];
//       for (int i = N; i > 0; i--) {
//         out_factor_index[j * N + i - 1] = idx / l_cumprod[i-1];
//         idx = idx % l_cumprod[i-1];
//       }
//     }
//   }
//
//   template <typename InputType, typename IndexType>
//   bool GetSlice(std::vector<std::vector<T>>& tgt_slice, InputType* core, const vector<IndexType>& ind_slice, int bs, int idx) {
//     //implement the functinality index_select(core, 1, ind_slice)
//     auto num_of_elements = ranks[idx]*factor_j[idx]*ranks[idx+1];
//     for (int i = 0; i < bs; i++) {
//       memcpy(tgt_slice[i].data(), core+ind_slice[i]*num_of_elements, num_of_elements*sizeof(InputType));
//     }
//     return true;
//   }
//
//   template <typename InputType, typename IndexType>
//   bool GatherAllRows(IndexType* ind,
//     int bs,
//     int x_len,
//     vector<const InputType*> cores,
//     int segments,
//     const int* lengths,
//     T* out_data) {
//
//     // compute the largest memory consumption of intermediate result
//     // TODO: evaluate dynamic allocation size: cur_rows*factor_j[i]*ranks[i+1]
//     int max_rank = *max_element(ranks.begin(), ranks.end());
//
//     // TODO: explore the contiguous memory storage for res and int_res
//     std::vector<std::vector<T>> res(bs, std::vector<T>(emb_size*max_rank, 0));
//     std::vector<std::vector<T>> int_res(bs, std::vector<T>(emb_size*max_rank, 0));
//
//     // Store the matrix A
//     vector<T*> Y_ptr(bs);
//     // Store the intermediate result in each layer
//     vector<T*> Z_ptr(bs);
//     for(int b = 0; b < bs; b++){
//       Y_ptr[b] = res[b].data();
//       Z_ptr[b] = int_res[b].data();
//     }
//
//     vector<IndexType> ind_slice(bs);
//     int rows = 0;
//     for (int i = 0; i < x_len; i++) {
//       //slice cur
//       for (int j = 0; j < bs; j++){
//         ind_slice[j] = ind[x_len*j + i];
//       }
//       if (i == 0){
//         GetSlice(res, cores[i], ind_slice, bs, i);
//         rows = factor_j[0];
//       } else {
//         std::vector<std::vector<T>> slice(bs, std::vector<T>(ranks[i]*factor_j[i]*ranks[i+1], 0));
//         vector<const T*> X_ptr(bs);
//         for(int b = 0; b < bs; b++){
//           X_ptr[b] = slice[b].data();
//         }
//         GetSlice(slice, cores[i], ind_slice, bs, i);
//
//         math::GemmBatched<T, CPUContext>(
//             CblasNoTrans,
//             CblasNoTrans,
//             bs,
//             rows,
//             factor_j[i]*ranks[i+1],
//             ranks[i],
//             1.0f,
//             // ((i%2)? const_cast<const T**>(Y_ptr.data()) : const_cast<const T**>(Z_ptr.data())),
//             const_cast<const T**>(Y_ptr.data()),
//             X_ptr.data(),
//             0.0f,
//             Z_ptr.data(),
//             // ((i%2)? Z_ptr.data() : Y_ptr.data()),
//             &context_
//             );
//         for(int b = 0; b < bs; b++){
//           std::memcpy(Y_ptr[b], Z_ptr[b], (emb_size*max_rank)*sizeof(T));
//         }
//         rows *= factor_j[i];
//       }
//       //save the intermediate output for backward path
//       // shape for the core
//       // TODO: add is_test argument to selectively save for training
//       auto shape = vector<int64_t>({bs, rows, ranks[i+1]});
//       auto* core_data = Output(i+1, shape, at::dtype<T>());
//       T* out_core = core_data->template mutable_data<T>();
//       for(int b = 0; b < bs; b++){
//         std::memcpy(out_core+b*rows*ranks[i+1], Y_ptr[b], rows*ranks[i+1]*sizeof(T));
//       }
//     }
//     // if (x_len%2 == 0){
//     //   for(int b = 0; b < bs; b++){
//     //     std::memcpy(Y_ptr[b], Z_ptr[b], (emb_size*max_rank)*sizeof(T));
//     //   }
//     // }
//
//     // reduction and store back to output
//     vector<IndexType> cum_lengths(segments);
//     for (int seg = 0; seg < segments; seg++){
//       cum_lengths[seg] = seg == 0 ? lengths[0] : lengths[seg] + cum_lengths[seg-1];
//     }
//     // cum_lengths[segments] = bs;
//
//     int length_idx = 0;
//     vector<T> tmp_sum(emb_size, 0.0f);
//     for (int i = 0; i <= bs; i++) {
//       if (i == cum_lengths[length_idx]){
//         //store the tmp_sum into output
//         memcpy(&out_data[length_idx*emb_size], tmp_sum.data(), emb_size*sizeof(T));
//         if (i == bs) {
//           break;
//         }
//         length_idx++;
//         //reset the tmp_sum;
//         fill(tmp_sum.begin(), tmp_sum.end(), 0.0f);
//       }
//       transform(res[i].begin(), res[i].begin() + emb_size, tmp_sum.begin(), tmp_sum.begin(), std::plus<T>());
//     }
//     return true;
//   }
//
//   template <typename InputType>
//   bool DoRunWithType() {
//     return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
//         this, Input(INDICES));
//   }
//
//   template <typename InputType, typename IndexType>
//   bool DoRunWithType2() {
//     auto& dataInput0 = Input(CORE0);
//     auto& dataInput1 = Input(CORE1);
//     auto& dataInput2 = Input(CORE2);
//     auto& indicesInput = Input(INDICES);
//     auto& lengthsInput = Input(LENGTHS);
//
//     CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
//     CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
//
//     int N = factor_i.size();
//     const int64_t M = lengthsInput.size(0);
//
//     auto shape = vector<int64_t>({M, emb_size});
//     auto* output = Output(0, shape, at::dtype<T>());
//     T* out_data = output->template mutable_data<T>();
//
//     const InputType* core0 = dataInput0.template data<InputType>();
//     const InputType* core1 = dataInput1.template data<InputType>();
//     const InputType* core2 = dataInput2.template data<InputType>();
//     const int* lengths = lengthsInput.template data<int>();
//
//     vector<const InputType*> cores = {core0, core1, core2};
//
//     const IndexType* indices = indicesInput.template data<IndexType>();
//
//     // Store the factor index for backward path
//     auto index_shape = vector<int64_t>({indicesInput.size(), N});
//     auto* index_data = Output(4, shape, at::dtype<IndexType>());
//     IndexType* out_factor_index = index_data->template mutable_data<IndexType>();
//
//     // Store the factorized index for each core
//     Ind2Sub<IndexType>(out_factor_index, indices, indicesInput.size());
//
//     return GatherAllRows<InputType, IndexType>(
//         out_factor_index,
//         indicesInput.size(),
//         N,
//         cores,
//         M,
//         lengths,
//         out_data
//         );
//   }
//
//   enum {
//     CORE0 = 0, // Data input.
//     CORE1 = 1, // Data input.
//     CORE2 = 2, // Data input.
//     WEIGHT = 3, // Weight input used in SparseLengthsWeightedSum
//     INDICES = 3 + USE_WEIGHT, // 1 in SparseLengths[Sum,Mean] and
//                               // 2 in SparseLengthsWeightedSum
//     LENGTHS = 4 + USE_WEIGHT, // 2 in SparseLengths[Sum, Mean],
//                               // 3 in SparseLengthsWeightedSum
//   };
//   vector<int> factor_i;
//   vector<int> factor_j;
//   vector<int> ranks;
//   vector<int> l_cumprod;
//   int emb_size;
//   std::unique_ptr<Blob> Y_temp_;
// };

// template <typename T, class Context>
// class TTSparseLengthsSumGradientOp final : public Operator<Context> {
// public:
//   USE_OPERATOR_CONTEXT_FUNCTIONS;
//   template <class... Args>
//   explicit TTSparseLengthsSumGradientOp(Args&&... args)
//       : Operator<Context>(std::forward<Args>(args)...){
//       }
//   bool RunOnDevice() override;
//
//   ~TTSparseLengthsSumGradientOp() {}
// };

template <typename T, class Context>
class TTSparseLengthsSumGradientOp final : public Operator<Context> {
public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit TTSparseLengthsSumGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...){
      }
  bool RunOnDevice() override;

  ~TTSparseLengthsSumGradientOp() {}
};

//implement the graident op for TTLengthSumGradient op
template <typename T, class Context>
bool TTSparseLengthsSumGradientOp<T, Context>::RunOnDevice() {
  // // INDICES, LENGTHS, CORE0_output, CORE1_output, CORE2_output, dY
  // vector<string>{I(3), I(4), O(1), O(2), O(3), O(4), GO(0)},
  // // dCore0, dCore1, dCore2
  // vector<string>{GI(0), GI(1), GI(2)});

  const auto& core0 = Input(0);
  const auto& core1 = Input(1);
  const auto& core2 = Input(2);
  const auto& indices = Input(3); // unused
  const auto& lengths = Input(4);
  const auto& core0_out = Input(5);
  const auto& core1_out = Input(6);
  const auto& core2_out = Input(7);
  const auto& index_out = Input(8);
  const auto& dY = Input(9);

  const int* lengths_data = lengths.template data<int>();
  const T* dY_data = dY.template data<T>();

  // restore the arguments from shape
  const int64_t bs = index_out.size(0);
  // TODO: assert bs=dY.size(0)
  const int64_t emb_size = dY.size(1);
  const int64_t num_segments = lengths.size(0);

  vector<int> ranks{1, core0_out.size(2), core1_out.size(2), core2_out.size(2)};

  auto core0_shape = core0.sizes().vec();
  auto core1_shape = core1.sizes().vec();
  auto core2_shape = core2.sizes().vec();
  auto core0_out_shape = core0_out.sizes().vec();
  auto core1_out_shape = core1_out.sizes().vec();

  auto* dCore0 = Output(0, core0_shape, at::dtype<T>());
  auto* dCore1 = Output(1, core1_shape, at::dtype<T>());
  auto* dCore2 = Output(2, core2_shape, at::dtype<T>());

  T* dCore0_data = dCore0->template mutable_data<T>();
  T* dCore1_data = dCore1->template mutable_data<T>();
  T* dCore2_data = dCore2->template mutable_data<T>();

  memset(dCore0_data, 0.0f, sizeof(T)*accumulate(core0_shape.begin(), core0_shape.end(), 1, std::multiplies<T>()));
  memset(dCore1_data, 0.0f, sizeof(T)*accumulate(core1_shape.begin(), core1_shape.end(), 1, std::multiplies<T>()));
  memset(dCore2_data, 0.0f, sizeof(T)*accumulate(core2_shape.begin(), core2_shape.end(), 1, std::multiplies<T>()));

  int64_t* index_out_data = index_out.template mutable_data<int64_t>();

  vector<vector<int64_t>> index_slice(bs, vector<int64_t>(3, 0));
  for(int64_t b = 0; b < bs; b++){
    memcpy(index_slice[b].data(), index_out_data + b*3, 3*sizeof(int64_t));
  }

  vector<const T*> A_ptr(bs);
  vector<T*> B_ptr(bs);
  vector<T*> C_ptr(bs);
  // size of each batch
  int64_t num_of_elements = 0;

  // construct the ranks
  // expand the gradient into all indices
  vector<vector<T>> core2_out_grad(bs, vector<T>(emb_size, 0));
  int64_t data_index = 0;
  for (int64_t range_index = 0; range_index < num_segments; ++range_index) {
    for (int64_t start = data_index; data_index < start + lengths_data[range_index];
         ++data_index) {
      memcpy(core2_out_grad[data_index].data(), dY_data + range_index*emb_size, emb_size*sizeof(T));
    }
  }

  // =======================================================
  // Calculate dCore2_data:
  // 1) Transpose core1_out and multiply iwth core2_out_grad
  // 2)  add to dCore2_data
  vector<vector<T>> dCore2_data_slice_grad(bs, vector<T>(core2_shape[1]*core2_shape[2]*core2_shape[3], 0));
  const T* core1_out_data = core1_out.template data<T>();
  // const T* core1_out_p[bs];
  for (int64_t b = 0; b < bs; b++) {
    A_ptr[b] = core1_out_data + b*core1_out.size(1)*core1_out.size(2);
    B_ptr[b] = core2_out_grad[b].data();
    C_ptr[b] = dCore2_data_slice_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasTrans,
      CblasNoTrans,
      bs,
      core2.size(1), //M
      core2.size(2)*core2.size(3), //N
      core1_out.size(1),//K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_
      );

  // update the corresponding slice
  num_of_elements = core2_shape[1]*core2_shape[2]*core2_shape[3];

  T* core2_data = core2.template mutable_data<T>();
  vector<vector<T>> core2_slice(bs, vector<T>(core2_shape[1]*core2_shape[2]*core2_shape[3], 0));

  for (int64_t b = 0;  b < bs;  b++) {
    for(int i = 0; i < num_of_elements; i++) {
      dCore2_data[index_slice[b][2]*num_of_elements + i] += C_ptr[b][i];
    }
    memcpy(core2_slice[b].data(), core2_data + index_slice[b][2]*num_of_elements, sizeof(T)*num_of_elements);
  }

  // Calculate core1_out_grad
  vector<vector<T>> core1_out_grad(bs, vector<T>(core1_out_shape[1]*core1_out_shape[2], 0));

  for(int64_t b = 0;  b < bs;  b++) {
    A_ptr[b] = core2_out_grad[b].data();
    B_ptr[b] = core2_slice[b].data();
    C_ptr[b] = core1_out_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasNoTrans,
      CblasTrans,
      bs,
      core1_out.size(1), //M
      core2_shape[1], //N
      core2_shape[2]*core2_shape[3], //K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_
      );

  // =======================================================
  // Calcuate dCore1_data:
  // 1) Transpose core1_out_grad and multiply with core0_out
  // 2) Transpose the result and then add to dCore1_data
  vector<vector<T>> dCore1_data_slice_grad(bs, vector<T>(core1_shape[1]*core1_shape[2]*core1_shape[3], 0));
  const T* core0_out_data = core0_out.template data<T>();
  for (int64_t b = 0; b < bs; b++) {
    A_ptr[b] = core0_out_data + b*core0_out.size(1)*core0_out.size(2);
    B_ptr[b] = core1_out_grad[b].data();
    C_ptr[b] = dCore1_data_slice_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasTrans,
      CblasNoTrans,
      bs,
      core1.size(1), //M
      core1.size(2)*core1.size(3), //N
      core0_out.size(1), //K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_
      );

  // update the corresponding slice
  num_of_elements = core1_shape[1]*core1_shape[2]*core1_shape[3];
  T* core1_data = core1.template mutable_data<T>();
  vector<vector<T>> core1_slice(bs, vector<T>(core1_shape[1]*core1_shape[2]*core1_shape[3], 0));

  for (int64_t b = 0;  b < bs;  b++) {
    for(int i = 0; i < num_of_elements; i++) {
      dCore1_data[index_slice[b][1]*num_of_elements + i] += C_ptr[b][i];
    }
    memcpy(core1_slice[b].data(), core1_data + index_slice[b][1]*num_of_elements, sizeof(T)*num_of_elements);
  }

  // Calcuate core0_out_grad
  vector<vector<T>> core0_out_grad(bs, vector<T>(core0_out_shape[1]*core0_out_shape[2], 0));

  for(int64_t b = 0;  b < bs;  b++) {
    A_ptr[b] = core1_out_grad[b].data();
    B_ptr[b] = core1_slice[b].data();
    C_ptr[b] = core0_out_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasNoTrans,
      CblasTrans,
      bs,
      core0_out.size(1), //M
      core1_shape[1], //N
      core1_shape[2]*core1_shape[3], //K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_
      );

  num_of_elements = core0_shape[1]*core0_shape[2]*core0_shape[3];

  for (int64_t b = 0;  b < bs;  b++) {
    for(int i = 0; i < num_of_elements; i++) {
      dCore0_data[index_slice[b][0]*num_of_elements + i] += C_ptr[b][i];
    }
  }
  return true;
}
// template <typename T, class Context>
// class TTSparseLengthsSumGradientOp final : public Operator<Context> {
// public:
//   USE_OPERATOR_CONTEXT_FUNCTIONS;
//   template <class... Args>
//   explicit TTSparseLengthsSumGradientOp(Args&&... args)
//       : Operator<Context>(std::forward<Args>(args)...){
//       }
//   bool RunOnDevice() override;
//
//   ~TTSparseLengthsSumGradientOp() {}
// };
//
// //implement the graident op for TTLengthSumGradient op
// template <typename T, class Context>
// bool TTSparseLengthsSumGradientOp<T, Context>::RunOnDevice() {
//   // // INDICES, LENGTHS, CORE0_output, CORE1_output, CORE2_output, dY
//   // vector<string>{I(3), I(4), O(1), O(2), O(3), O(4), GO(0)},
//   // // dCore0, dCore1, dCore2
//   // vector<string>{GI(0), GI(1), GI(2)});
//
//   const auto& core0 = Input(0);
//   const auto& core1 = Input(1);
//   const auto& core2 = Input(2);
//   const auto& indices = Input(3);
//   const auto& lengths = Input(4);
//   const auto& core0_out = Input(5);
//   const auto& core1_out = Input(6);
//   const auto& core2_out = Input(7);
//   const auto& index_out = Input(8);
//   const auto& dY = Input(9);
//
//   const int* lengths_data = lengths.template data<int>();
//   const T* dY_data = dY.template data<T>();
//
//   // restore the arguments from shape
//   const int64_t bs = indices.size(0);
//   // TODO: assert bs=dY.size(0)
//   const int64_t emb_size = dY.size(1);
//   const int64_t num_segments = lengths.size(0);
//
//   vector<int> ranks{1, core0_out.size(2), core1_out.size(2), core2_out.size(2)};
//   vector<int> factor_i{core0.size(0), core1.size(0), core2.size(0)};
//   vector<int> factor_j{core0.size(2), core1.size(2), core2.size(2)};
//   auto core0_shape = core0.sizes().vec();
//   auto core1_shape = core1.sizes().vec();
//   auto core2_shape = core2.sizes().vec();
//   auto core0_out_shape = core0_out.sizes().vec();
//   auto core1_out_shape = core1_out.sizes().vec();
//
//   auto* dCore0 = Output(0, core0_shape, at::dtype<T>());
//   auto* dCore1 = Output(1, core1_shape, at::dtype<T>());
//   auto* dCore2 = Output(2, core2_shape, at::dtype<T>());
//
//   T* dCore0_data = dCore0->template mutable_data<T>();
//   T* dCore1_data = dCore1->template mutable_data<T>();
//   T* dCore2_data = dCore2->template mutable_data<T>();
//
//   memset(dCore0_data, 0.0f, sizeof(T)*accumulate(core0_shape.begin(), core0_shape.end(), 1, std::multiplies<T>()));
//   memset(dCore1_data, 0.0f, sizeof(T)*accumulate(core1_shape.begin(), core1_shape.end(), 1, std::multiplies<T>()));
//   memset(dCore2_data, 0.0f, sizeof(T)*accumulate(core2_shape.begin(), core2_shape.end(), 1, std::multiplies<T>()));
//
//   int64_t* index_out_data = index_out.template mutable_data<int64_t>();
//   vector<vector<int64_t>> index_slice(bs, vector<int64_t>(3, 0));
//   for(int64_t b = 0; b < bs; b++){
//     memcpy(index_slice[b].data(), index_out_data + b*3, 3*sizeof(int64_t));
//   }
//
//   vector<const T*> A_ptr(bs);
//   vector<T*> B_ptr(bs);
//   vector<T*> C_ptr(bs);
//
//   // construct the ranks
//   // fill the gradient into all indices
//   // vector<T> core2_out_grad(bs*emb_size, 0);
//   vector<vector<T>> core2_out_grad(bs, vector<T>(emb_size, 0));
//   int64_t data_index = 0;
//   for (int64_t range_index = 0; range_index < num_segments; ++range_index) {
//     for (int64_t start = data_index; data_index < start + lengths_data[range_index];
//          ++data_index) {
//       // copy emb_size data
//       memcpy(core2_out_grad[data_index].data(), dY_data + range_index*emb_size, emb_size*sizeof(T));
//     }
//   }
//
//   // =======================================================
//   // Calculate dCore2_data:
//   // 1) Transpose core1_out and multiply iwth core2_out_grad
//   // 2)  add to dCore2_data
//   vector<vector<T>> dCore2_data_slice_grad(bs, vector<T>(core2_shape[1]*core2_shape[2]*core2_shape[3], 0));
//   const T* core1_out_data = core1_out.template data<T>();
//   // const T* core1_out_p[bs];
//   for (int64_t b = 0; b < bs; b++) {
//     A_ptr[b] = core1_out_data + b*core1_out.size(1)*core1_out.size(2);
//     B_ptr[b] = core2_out_grad[b].data();
//     C_ptr[b] = dCore2_data_slice_grad[b].data();
//   }
//
//   math::GemmBatched<T, CPUContext>(
//       CblasTrans,
//       CblasNoTrans,
//       bs,
//       core2.size(1), //M
//       core2.size(2)*core2.size(3), //N
//       core1_out.size(1),//K
//       1.0f,
//       const_cast<const T**>(A_ptr.data()),
//       const_cast<const T**>(B_ptr.data()),
//       0.0f,
//       C_ptr.data(),
//       &context_
//       );
//
//   // update the corresponding slice
//   int64_t num_of_elements = core2_shape[1]*core2_shape[2]*core2_shape[3];
//   T* core2_data = core2.template mutable_data<T>();
//   vector<vector<T>> core2_slice(bs, vector<T>(core2_shape[1]*core2_shape[2]*core2_shape[3], 0));
//
//   for (int64_t b = 0;  b < bs;  b++) {
//     //index_slice[2][b]
//     // memcpy(dCore2_data + index_slice[b][2]*num_of_elements, dCore2_data_slice_grad[b].data(), sizeof(T)*num_of_elements);
//     // memcpy(dCore2_data + index_slice[b][2]*num_of_elements, C_ptr[b], sizeof(T)*num_of_elements);
//     for(int i = 0; i < num_of_elements; i++) {
//       dCore2_data[index_slice[b][2]*num_of_elements + i] += C_ptr[b][i];
//     }
//
//     memcpy(core2_slice[b].data(), core2_data + index_slice[b][2]*num_of_elements, sizeof(T)*num_of_elements);
//     // std::cout << "\n update core2 batch " << index_slice[b][2] << std::endl;
//     // for(int i = 0; i<num_of_elements; i++)
//     //   std::cout << C_ptr[b][i]*0.01 << ", ";
//     // std::cout << std::endl;
//   }
//
//   // Calculate core1_out_grad
//   vector<vector<T>> core1_out_grad(bs, vector<T>(core1_out_shape[1]*core1_out_shape[2], 0));
//
//   for(int64_t b = 0;  b < bs;  b++) {
//     A_ptr[b] = core2_out_grad[b].data();
//     B_ptr[b] = core2_slice[b].data();
//     C_ptr[b] = core1_out_grad[b].data();
//   }
//
//   math::GemmBatched<T, CPUContext>(
//       CblasNoTrans,
//       CblasTrans,
//       bs,
//       core1_out.size(1), //M
//       core2_shape[1], //N
//       core2_shape[2]*core2_shape[3], //K
//       1.0f,
//       const_cast<const T**>(A_ptr.data()),
//       const_cast<const T**>(B_ptr.data()),
//       0.0f,
//       C_ptr.data(),
//       &context_
//       );
//
//   // =======================================================
//   // Calcuate dCore1_data:
//   // 1) Transpose core1_out_grad and multiply with core0_out
//   // 2) Transpose the result and then add to dCore1_data
//   vector<vector<T>> dCore1_data_slice_grad(bs, vector<T>(core1_shape[1]*core1_shape[2]*core1_shape[3], 0));
//   const T* core0_out_data = core0_out.template data<T>();
//   //const T* core0_out_p[bs];
//   for (int64_t b = 0; b < bs; b++) {
//     A_ptr[b] = core0_out_data + b*core0_out.size(1)*core0_out.size(2);
//     B_ptr[b] = core1_out_grad[b].data();
//     C_ptr[b] = dCore1_data_slice_grad[b].data();
//   }
//
//   math::GemmBatched<T, CPUContext>(
//       CblasTrans,
//       CblasNoTrans,
//       bs,
//       core1.size(1), //M
//       core1.size(2)*core1.size(3), //N
//       core0_out.size(1), //K
//       1.0f,
//       const_cast<const T**>(A_ptr.data()),
//       const_cast<const T**>(B_ptr.data()),
//       0.0f,
//       C_ptr.data(),
//       &context_
//       );
//       // update the corresponding slice
//     num_of_elements = core1_shape[1]*core1_shape[2]*core1_shape[3];
//     T* core1_data = core1.template mutable_data<T>();
//     vector<vector<T>> core1_slice(bs, vector<T>(core1_shape[1]*core1_shape[2]*core1_shape[3], 0));
//
//     for (int64_t b = 0;  b < bs;  b++) {
//       //index_slice[2][b]
//       // memcpy(dCore1_data + index_slice[b][1]*num_of_elements, dCore1_data_slice_grad[b].data(), sizeof(T)*num_of_elements);
//       // memcpy(dCore1_data + index_slice[b][1]*num_of_elements, C_ptr[b], sizeof(T)*num_of_elements);
//       for(int i = 0; i < num_of_elements; i++) {
//         dCore1_data[index_slice[b][1]*num_of_elements + i] += C_ptr[b][i];
//       }
//       memcpy(core1_slice[b].data(), core1_data + index_slice[b][1]*num_of_elements, sizeof(T)*num_of_elements);
//       // std::cout << "\n update core1 batch " << index_slice[b][1] << std::endl;
//       // for(int i = 0; i<num_of_elements; i++)
//       //   std::cout << C_ptr[b][i]*0.01 << ", ";
//       // std::cout << std::endl;
//     }
//
//   // Calcuate core0_out_grad
//   vector<vector<T>> core0_out_grad(bs, vector<T>(core0_out_shape[1]*core0_out_shape[2], 0));
//
//   for(int64_t b = 0;  b < bs;  b++) {
//     A_ptr[b] = core1_out_grad[b].data();
//     B_ptr[b] = core1_slice[b].data();
//     C_ptr[b] = core0_out_grad[b].data();
//   }
//
//   math::GemmBatched<T, CPUContext>(
//       CblasNoTrans,
//       CblasTrans,
//       bs,
//       core0_out.size(1), //M
//       core1_shape[1], //N
//       core1_shape[2]*core1_shape[3], //K
//       1.0f,
//       const_cast<const T**>(A_ptr.data()),
//       const_cast<const T**>(B_ptr.data()),
//       0.0f,
//       C_ptr.data(),
//       &context_
//       );
//
//   num_of_elements = core0_shape[1]*core0_shape[2]*core0_shape[3];
//   for (int64_t b = 0;  b < bs;  b++) {
//     //index_slice[2][b]
//     // memcpy(dCore0_data + index_slice[b][0]*num_of_elements, core0_out_grad[b].data(), sizeof(T)*num_of_elements);
//     // memcpy(dCore0_data + index_slice[b][0]*num_of_elements, C_ptr[b], sizeof(T)*num_of_elements);
//     for(int i = 0; i < num_of_elements; i++) {
//       dCore0_data[index_slice[b][0]*num_of_elements + i] += C_ptr[b][i];
//     }
//     // std::cout << "\n update core0 batch " << index_slice[b][0] << std::endl;
//     // for(int i = 0; i<num_of_elements; i++)
//     //   std::cout << C_ptr[b][i]*0.01 << ", ";
//     // std::cout << std::endl;
//   }
//   return true;
// }

} // namespace caffe2
