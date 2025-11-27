from pathlib import Path

lines = []
lines.append('Part D: Four VAE Variants, latent_dim 256, 256x256, Tanh outputs')
lines.append('Validation first 1000 images; Epochs 20; AMP on; no LPIPS')
lines.append('Loss: L1 reconstruction plus beta * KL normalized by latent_dim')
lines.append('')

archs = [
    ('VAEAttnShallow', [
        'Encoder: Conv4x4 s2 3->32; Conv4x4 s2 32->64 with ResBlock; Conv4x4 s2 64->128 with ResBlock; Conv4x4 s2 128->256 with ResBlock; Self-attention on 256x16x16; flatten to mu logvar 256',
        'Decoder: Linear to 256x16x16 with self-attention; ConvT4x4 s2 256->128 with ResBlock; ConvT4x4 s2 128->64 with ResBlock; ConvT4x4 s2 64->32 with ResBlock; ConvT4x4 s2 32->3; Tanh',
        'Hyper: batch 12; lr 2e-4; beta 0.001'
    ]),
    ('VAEAttnDeep', [
        'Encoder: Conv4x4 s2 3->48; Conv4x4 s2 48->96 with ResBlock; Conv4x4 s2 96->192 with ResBlock; Conv4x4 s2 192->384 with ResBlock; Conv4x4 s2 384->512 with ResBlock; Self-attention on 512x8x8; flatten to mu logvar 256',
        'Decoder: Linear to 512x8x8 with self-attention; ConvT4x4 s2 512->384 with ResBlock; ConvT4x4 s2 384->192 with ResBlock; ConvT4x4 s2 192->96 with ResBlock; ConvT4x4 s2 96->48 with ResBlock; ConvT4x4 s2 48->3; Tanh',
        'Hyper: batch 10; lr 2e-4; beta 0.002'
    ]),
    ('VAEResWide', [
        'Encoder: Conv4x4 s2 3->64; Conv4x4 s2 64->128 with ResBlock; Conv4x4 s2 128->256 with ResBlock; Conv4x4 s2 256->512 with ResBlock; flatten to mu logvar 256',
        'Decoder: Linear to 512x16x16; ConvT4x4 s2 512->256 with ResBlock; ConvT4x4 s2 256->128 with ResBlock; ConvT4x4 s2 128->64 with ResBlock; ConvT4x4 s2 64->3; Tanh',
        'Hyper: batch 10; lr 2e-4; beta 0.001'
    ]),
    ('VAEPlainDeep', [
        'Encoder: Conv4x4 s2 3->48; Conv4x4 s2 48->96; Conv4x4 s2 96->192; Conv4x4 s2 192->384; Conv4x4 s2 384->512; flatten to mu logvar 256',
        'Decoder: Linear to 512x8x8; ConvT4x4 s2 512->384; ConvT4x4 s2 384->192; ConvT4x4 s2 192->96; ConvT4x4 s2 96->48; ConvT4x4 s2 48->3; Tanh',
        'Hyper: batch 12; lr 2e-4; beta 0.001'
    ]),
]
for name, details in archs:
    lines.append(name)
    for d in details:
        lines.append('  - ' + d)
    lines.append('')

lines.append('Training settings common: Adam betas 0.9 0.999; KL per-dim; save validation recon grids; log total L1 KL PSNR')

content_lines = []
y = 780
leading = 14
content_lines.append('BT')
content_lines.append('/F1 12 Tf')
for ln in lines:
    if y < 60:
        break
    ln = ln.replace('(', '[').replace(')', ']')
    content_lines.append(f'72 {y} Td ({ln}) Tj')
    content_lines.append('T*')
    y -= leading
content_lines.append('ET')
content_stream = '\n'.join(content_lines)
length = len(content_stream.encode('latin-1'))

objects = []
objects.append('1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj')
objects.append('2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj')
objects.append('3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj')
objects.append(f'4 0 obj\n<< /Length {length} >>\nstream\n{content_stream}\nendstream\nendobj')
objects.append('5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj')

pdf_parts = ['%PDF-1.4\n']
offsets = []
current_pos = len(pdf_parts[0])
for obj in objects:
    offsets.append(current_pos)
    pdf_parts.append(obj + '\n')
    current_pos += len(obj) + 1
xref_pos = current_pos
xref_lines = ['xref', f'0 {len(objects)+1}', '0000000000 65535 f ']
for off in offsets:
    xref_lines.append(f{off:010d}
