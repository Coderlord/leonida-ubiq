"""--int-dot-product: v_dot4_i32_i8 and v_dot4c_i32_i8 as one packed signed-byte dot product (OpSDotKHR with
PackedVectorFormat4x8BitKHR, SPV_KHR_integer_dot_product; Vulkan 1.3's shaderIntegerDotProduct).

Every case of test_alu_dot4_i8.py (the words measured on RDNA2 hardware: plain, op_sel 2 / op_sel_hi 5, the other
half selections, clamp's saturation, the refused forms) runs again with the switch on, through the interpreter's
OpSDot. The module then holds one OpSDotKHR per dot and no byte extract, declares the two capabilities and the
extension, and the index names the feature.

    python -m unittest discover -s tests/semantic -p test_alu_dot4_i8_native.py
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import test_alu_dot4_i8 as plain  # noqa: E402

rdna2spirv = support.rdna2spirv


class NativeSwitch:
    def setUp(self):
        self._saved = rdna2spirv.INT_DOT_PRODUCT
        rdna2spirv.INT_DOT_PRODUCT = True

    def tearDown(self):
        rdna2spirv.INT_DOT_PRODUCT = self._saved


class Dot4I8Native(NativeSwitch, plain.Dot4I8):
    """The measured cases of test_alu_dot4_i8.Dot4I8, translated with --int-dot-product."""


class Dot4I8NativeForm(NativeSwitch, unittest.TestCase):
    def test_one_packed_dot_per_instruction(self):
        a, b, addend, measured = plain.MEASURED[2]
        spv = support.translate_lines("ps_dot4_native", plain.program(a, b, addend))
        self.assertEqual(support.mrt0_words(spv), [measured] * 4)
        self.assertEqual(len(re.findall(r"= OpSDotKHR %uint \S+ \S+ PackedVectorFormat4x8BitKHR", spv)), 4)
        self.assertNotIn("OpBitFieldSExtract", spv)
        for line in ("OpCapability DotProductKHR", "OpCapability DotProductInput4x8BitPackedKHR",
                     'OpExtension "SPV_KHR_integer_dot_product"'):
            self.assertIn(line, spv)

    def test_a_swapped_source_is_repacked_first(self):
        lines = ["v_mov_b32_e32 v1, 0x04030201", "v_mov_b32_e32 v2, 0x08070605", "v_mov_b32_e32 v3, 0",
                 "v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
                 "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
        spv = support.translate_lines("ps_dot4_native_sel", lines)
        self.assertEqual(support.mrt0_words(spv), [62] * 4)
        self.assertEqual(spv.count("OpBitFieldInsert %uint"), 1)   # S1 only: S0 keeps its natural order

    def test_the_index_names_the_feature(self):
        a, b, addend, _ = plain.MEASURED[0]
        spv = support.translate_lines("ps_dot4_native_meta", plain.program(a, b, addend))
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
        self.assertIn("shaderIntegerDotProduct", meta["features"])
        self.assertIn("SPV_KHR_integer_dot_product", meta["extensions"])

    def test_off_by_default(self):
        rdna2spirv.INT_DOT_PRODUCT = False
        a, b, addend, _ = plain.MEASURED[0]
        spv = support.translate_lines("ps_dot4_plain", plain.program(a, b, addend))
        self.assertNotIn("OpSDot", spv)
        self.assertNotIn("DotProduct", spv)


if __name__ == "__main__":
    unittest.main()
