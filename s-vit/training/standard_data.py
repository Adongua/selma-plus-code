"""Owner-local dataset using the standard ViT teacher's public image processor."""
from PIL import Image

from .data import PartyData


class StandardPartyData(PartyData):
    def __init__(self, party, spec, root=None, manifest=None, teacher_dir=None):
        super().__init__(party, spec, root, manifest)
        self.processor = None
        if not spec['fixture']:
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(
                teacher_dir, local_files_only=True, trust_remote_code=False, use_fast=False)

    def batch(self, step, size, fixed=False):
        if self.rows is None:
            return super().batch(step, size, fixed)
        indices = [(0 if fixed else step*size) + i for i in range(size)]
        indices = [i % len(self) for i in indices]
        images = []
        for i in indices:
            with Image.open(self.root / self.rows[i]['path']) as image:
                images.append(image.convert('RGB').copy())
        import torch
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].double()
        labels = torch.tensor([self.rows[i]['label'] for i in indices])
        return pixels, labels
