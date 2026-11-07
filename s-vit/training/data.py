"""Owner-local image manifests, or explicitly public overfit fixtures."""
import json
from pathlib import Path
import torch


class PartyData:
    def __init__(self, party, spec, root=None, manifest=None, seed=91):
        self.party, self.spec = party, spec
        if spec['fixture']:
            generator = torch.Generator().manual_seed(seed+party)
            self.images = torch.randn(4, *spec['image_shape'], generator=generator, dtype=torch.float64)*.2
            self.labels = torch.zeros(4, dtype=torch.long)
            self.rows = None
        else:
            self.root = Path(root).resolve()
            self.rows = json.loads(Path(manifest).read_text())
            if not isinstance(self.rows, list) or not self.rows:
                raise ValueError("manifest must be a nonempty JSON list")
            seen = set()
            for row in self.rows:
                path = (self.root / row['path']).resolve()
                if not path.is_relative_to(self.root) or path in seen:
                    raise ValueError("manifest path escapes dataset root or repeats")
                seen.add(path)
                if not path.is_file() or type(row['label']) is not int or not 0 <= row['label'] < spec['classes']:
                    raise ValueError("missing image or invalid label")

    def __len__(self):
        return len(self.rows) if self.rows is not None else len(self.labels)

    def batch(self, step, size, fixed=False):
        # Deterministic sequential coverage; +/- always reuse this exact batch.
        start = 0 if fixed else step*size
        indices = [(start+i) % len(self) for i in range(size)]
        if self.rows is None:
            return self.images[indices].clone(), self.labels[indices].clone()
        from hattention.benchmark_vit_cuda import image_tensor
        images = [image_tensor(self.root/self.rows[i]['path'])[0].double() for i in indices]
        return torch.stack(images), torch.tensor([self.rows[i]['label'] for i in indices])
