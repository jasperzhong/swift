#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::autograd;

#define ASSERT_VARIABLE_EQ(a,b) ASSERT_TRUE(torch::allclose((a),(b)))
#define EXPECT_VARIABLE_EQ(a,b) EXPECT_TRUE(torch::allclose((a),(b)))

class MulConstant : public Function<MulConstant> {
 public:
  static Variable forward(AutogradContext *ctx, Variable variable, double constant) {
    ctx->saved_data["constant"] = constant;
    return variable * constant;
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
    return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), Variable()};
  }
};

TEST(AutogradAPITests, DEBUG_35736) {
  auto x = torch::randn({2}).requires_grad_();
  auto y = MulConstant::apply(x, 5.5);
  y.sum().backward();
  std::cout << x.grad() << std::endl;
}
