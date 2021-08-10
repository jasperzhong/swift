import dataclasses
import unittest

from tools.autograd import load_derivatives
import tools.codegen.model

class TestCreateDerivative(unittest.TestCase):

    def test_this_is_run(self):
        """Sanity check that this is run in CI: we expect a failure now."""
        # DO NOT MERGE
        raise AssertionError()

    def test_named_grads(self):
        schema = tools.codegen.model.FunctionSchema.parse(
            'func(Tensor a, Tensor b) -> (Tensor x, Tensor y)')
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        derivative = load_derivatives.create_derivative(
            native_function,
            formula='func_backward(grad_x, grad_y)',
            var_names=(),
            available_named_gradients=['grad_x', 'grad_y'])
        self.assertSetEqual(derivative.named_gradients, {'grad_x', 'grad_y'})

    def test_indexed_grads(self):
        schema = tools.codegen.model.FunctionSchema.parse(
            'func(Tensor a, Tensor b) -> (Tensor x, Tensor y)')
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        derivative = load_derivatives.create_derivative(
            native_function,
            formula='func_backward(grads[0], grads[1])',
            var_names=(),
            available_named_gradients=['grad_x', 'grad_y'])
        self.assertSetEqual(derivative.named_gradients, set())

    def test_named_grads_and_indexed_grads(self):
        specification = 'func(Tensor a, Tensor b) -> (Tensor x, Tensor y)'
        schema = tools.codegen.model.FunctionSchema.parse(specification)
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        with self.assertRaisesRegex(RuntimeError,
                                    'illegally mixes use of "grad_RETURN_NAME"'):
            load_derivatives.create_differentiability_info(
                defn={'name': specification,
                      # Uh-oh, the derivatives reference gradients by
                      # name and by index.
                      'a': 'grad_x',
                      'b': 'grads[1]',
                      },
                functions_by_signature={schema.signature(): [native_function]},
                functions_by_schema={specification: native_function},
            )


DEFAULT_NATIVE_FUNCTION, _ = tools.codegen.model.NativeFunction.from_yaml(
    {'func': 'func() -> bool'}, loc=tools.codegen.model.Location(__file__, 1))


if __name__ == '__main__':
    unittest.main()
