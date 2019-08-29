import unittest
import torch
from common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
    skipCUDANonDefaultStreamIf

# Defines device-specific "test bases" that allow "generic" or
# "device agnostic" tests to be specialized for a particular device type.
# To add a new device type you should inherit from DeviceTypeTestBase and
# add your class to device_type_test_bases when it's available.

class DeviceTypeTestBase(TestCase):
    device_type = "generic_device_type"
    tests_to_skip = {}

    # Marks all generated tests whose original names were in tests_to_skip
    # for skipping.
    @classmethod
    def add_skips(cls):
        replace_string = "_" + cls.device_type
        for test in [t for t in dir(cls) if t.startswith("test")]:
            orig_name = test.replace(replace_string, '')
            if orig_name in cls.tests_to_skip:
                @unittest.skip(cls.tests_to_skip[orig_name])
                def skipped_fun(self):
                    pass
                setattr(cls, test, skipped_fun)

class CPUTestBase(DeviceTypeTestBase):
    device_type = "cpu"

    @classmethod
    def setUpClass(cls):
        if not torch._C.has_lapack:
            reason = "PyTorch compiled without Lapack"
            tests = [
                "test_inverse",
                "test_inverse_many_batches",
                "test_pinverse",
                "test_matrix_rank",
                "test_matrix_power",
                "test_det_logdet_slogdet",
                "test_det_logdet_slogdet_batched",
                "test_solve",
                "test_solve_batched",
                "test_solve_batched_many_batches",
                "test_solve_batched_dims",
                "test_cholesky_solve",
                "test_cholesky_solve_batched",
                "test_cholesky_solve_batched_many_batches",
                "test_cholesky_solve_batched_dims",
                "test_cholesky_inverse",
                "test_cholesky",
                "test_cholesky_batched",
                "test_lu_solve_batched_many_batches",
                "test_symeig",
                "test_svd",
                "test_svd_no_singularvectors",
                "test_norm",
                "test_nuclear_norm_axes_small_brute_force",
                "test_geqrf",
                "test_triangular_solve"
            ]
            cls.tests_to_skip.update((t, reason) for t in tests)

        if not TEST_MKL:
            reason = "PyTorch is built without MKL support"
            tests = []
            cls.tests_to_skip.update((t, reason) for t in tests)

        cls.add_skips()

class CUDATestBase(DeviceTypeTestBase):
    device_type = "cuda"

    # Applies the skipCUDANonDefaultStreamIf(True) decorator to applicable
    # generated tests (skips)
    @classmethod
    def addSkipCUDANonDefaultStream(cls, skips):
        replace_string = "_" + cls.device_type
        for test in [t for t in dir(cls) if t.startswith("test")]:
            orig_name = test.replace(replace_string, '')
            if orig_name in skips:
                fun = getattr(cls, test)

                @skipCUDANonDefaultStreamIf(True)
                def wrapped_fun(self):
                    return fun(self)

                setattr(cls, test, wrapped_fun)

    @classmethod
    def setUpClass(cls):
        torch.ones(1).cuda()  # has_magma shows up after cuda is initialized
        if not torch.cuda.has_magma:
            reason = "no MAGMA library detected"
            tests = [
                "test_inverse",
                "test_inverse_many_batches",
                "test_pinverse",
                "test_matrix_rank",
                "test_matrix_power",
                "test_det_logdet_slogdet",
                "test_det_logdet_slogdet_batched",
                "test_solve",
                "test_solve_batched",
                "test_solve_batched_many_batches",
                "test_solve_batched_dims",
                "test_cholesky_solve",
                "test_cholesky_solve_batched",
                "test_cholesky_solve_batched_many_batches",
                "test_cholesky_solve_batched_dims",
                "test_cholesky_inverse",
                "test_cholesky",
                "test_cholesky_batched",
                "test_lu_solve_batched_many_batches",
                "test_symeig",
                "test_svd",
                "test_svd_no_singularvectors",
                "test_norm",
                "test_nuclear_norm_axes_small_brute_force",
                "test_nuclear_norm_exceptions",
                "test_geqrf",
                "test_triangular_solve"
            ]
            cls.tests_to_skip.update((t, reason) for t in tests)

        if TEST_WITH_ROCM:
            reason = "test doesn't currently work on the ROCm stack"
            tests = [
                "test_stft" # passes on ROCm w/ python 2.7, fails w/ python 3.6
            ]
            cls.tests_to_skip.update((t, reason) for t in tests)

        cls.add_skips()

        skipCUDANonDefaultStreamTests = [
            "test_multinomial_alias",
            "test_advancedindex",
            "test_norm",
            "test_nuclear_norm_axes_small_brute_force",
            "test_nuclear_norm_exceptions",
            "test_triangular_solve"
        ]
        cls.addSkipCUDANonDefaultStream(skipCUDANonDefaultStreamTests)

# Gathers available device types
device_type_test_bases = [CPUTestBase]

if torch.cuda.is_available():
    device_type_test_bases.append(CUDATestBase)

# Creates a device-type-specific test suite from the given generic test suite.
# All tests in the generic suite must begin with "test_" and have two required
# params, self and device.
# The test suites are put into the caller's provided scope to allow discovery.
def instantiate_device_type_tests(generic_test_class, scope):
    # Constructs a new class without the tests
    # Note: inheriting from the generic class is desirable because it may
    # container needed helper functions. Inherintg the "test_" functions,
    # however, would mean that generic tests were in the device-specific test
    # suite. This code creates a new class that is identical to the origin
    # generic class except the tests are removed.
    stripped_generic_test_class = type(generic_test_class.__name__ + "_base", (generic_test_class.__base__,), {})
    generic_funs = set(dir(generic_test_class)) - set(dir(stripped_generic_test_class))
    for generic_fun in generic_funs:
        if generic_fun.startswith('test_'):
            continue
        else:
            setattr(stripped_generic_test_class, generic_fun, getattr(generic_test_class, generic_fun))

    # Ports tests to device-specific classes and puts classes into scope
    generic_tests = [t for t in dir(generic_test_class) if t.startswith('test_')]
    for base in device_type_test_bases:
        class_name = generic_test_class.__name__ + base.device_type.upper()
        device_type_test_class = type(class_name, (base, stripped_generic_test_class), {})

        for generic in generic_tests:
            fun = getattr(generic_test_class, generic)
            fun_name = generic + '_' + base.device_type

            if generic in base.tests_to_skip:
                @unittest.skip(base.tests_to_skip[fun])
                def skipped_fun(self):
                    pass
                setattr(device_type_test_class, fun_name, skipped_fun)
            else:
                def wrapped_fun(self, func=fun, device=base.device_type):
                    return func(self, device)
                setattr(device_type_test_class, fun_name, wrapped_fun)

        scope[class_name] = device_type_test_class
