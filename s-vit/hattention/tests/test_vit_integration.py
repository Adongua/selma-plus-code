"""Fixed hAttention conversion and student/teacher isolation."""

import unittest

import torch
from torch import nn

from hattention.config import HAttentionConfig
from hattention.module import FixedHAttention
from hattention.vit_integration import convert_attention, make_fixed_student


class Attention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.num_heads = 4
        self.qkv = nn.Linear(16, 48, bias=False)
        self.proj = nn.Linear(16, 16)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(0.2)

    def forward(self, value):
        batch, tokens, dim = value.shape
        q, k, v = self.qkv(value).reshape(batch, tokens, 3, 4, 4).permute(2, 0, 3, 1, 4).unbind(0)
        weights = self.attn_drop(torch.softmax(q @ k.transpose(-2, -1) / 2, dim=-1))
        return self.proj_drop(self.proj((weights @ v).transpose(1, 2).reshape(batch, tokens, dim)))


class Block(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.self_attn = Attention(dropout)

    def forward(self, value):
        return value + self.self_attn(value)


class Model(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.classifier = nn.Module()
        self.classifier.seq_pool = False
        self.classifier.blocks = nn.ModuleList([Block(dropout), Block(dropout)])

    def forward(self, value):
        for block in self.classifier.blocks:
            value = block(value)
        return value


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = HAttentionConfig(num_heads=4, head_types=("rsa", "lsa", "mixer", "mixer"),
                                       patch_grid=(4, 4))

    def test_conversion_preserves_weights_dtype_grad_and_output_dropout(self):
        source = Attention().double().eval()
        source.qkv.weight.requires_grad_(False)
        converted = convert_attention(source, self.config)
        self.assertEqual(converted.qkv.weight.dtype, torch.float64)
        self.assertFalse(converted.qkv.weight.requires_grad)
        self.assertFalse(converted.training)
        self.assertEqual(converted.proj_drop.p, 0.2)
        self.assertTrue(torch.equal(converted.qkv.weight, source.qkv.weight))
        self.assertTrue(torch.equal(converted.proj.weight, source.proj.weight))
        self.assertNotEqual(converted.qkv.weight.data_ptr(), source.qkv.weight.data_ptr())
        result = converted(torch.randn(1, 17, 16, dtype=torch.float64))
        self.assertEqual(result.shape, (1, 17, 16))
        self.assertTrue(torch.isfinite(result).all())

    def test_student_is_independent_and_gradients_flow(self):
        teacher = Model().double().eval()
        student = make_fixed_student(teacher, [self.config, self.config])
        self.assertIsInstance(teacher.classifier.blocks[0].self_attn, Attention)
        self.assertIsInstance(student.classifier.blocks[0].self_attn, FixedHAttention)
        value = torch.randn(1, 17, 16, dtype=torch.float64)
        output = student(value)
        self.assertEqual(output.shape, value.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(student.classifier.blocks[0].self_attn.qkv.weight.grad).all())
        self.assertIsNone(teacher.classifier.blocks[0].self_attn.qkv.weight.grad)

    def test_dropout_and_cls_contract_are_explicit(self):
        teacher = Model(dropout=0.1)
        with self.assertRaisesRegex(ValueError, "attention-weight dropout"):
            make_fixed_student(teacher, [self.config, self.config])
        student = make_fixed_student(teacher, [self.config], layer_indices=[1],
                                     remove_attention_dropout=True)
        self.assertIsInstance(student.classifier.blocks[0].self_attn, Attention)
        self.assertIsInstance(student.classifier.blocks[1].self_attn, FixedHAttention)
        no_cls = HAttentionConfig(num_heads=4, head_types=self.config.head_types,
                                  has_cls_token=False, patch_grid=(4, 4))
        with self.assertRaisesRegex(ValueError, "seq_pool"):
            make_fixed_student(teacher, [no_cls], layer_indices=[0],
                               remove_attention_dropout=True)
        self.assertIsInstance(teacher.classifier.blocks[1].self_attn, Attention)


if __name__ == "__main__":
    unittest.main()
