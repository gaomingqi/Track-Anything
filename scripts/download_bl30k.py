import os
import gdown
import tarfile


LICENSE = """
This dataset is a derivative of ShapeNet.
Please read and respect their licenses and terms before use. 
Textures and skybox image are obtained from Google image search with the "non-commercial reuse" flag.
Do not use this dataset for commercial purposes.
You should cite both ShapeNet and our paper if you use this dataset.
"""

print(LICENSE)
print('Datasets will be downloaded and extracted to ../BL30K')
print('The script will download and extract the segment one by one')
print('You are going to need ~1TB of free disk space')
reply = input('[y] to confirm, others to exit: ')
if reply != 'y':
    exit()

links = [
    'https://drive.google.com/uc?id=1z9V5zxLOJLNt1Uj7RFqaP2FZWKzyXvVc',
    'https://drive.google.com/uc?id=11-IzgNwEAPxgagb67FSrBdzZR7OKAEdJ',
    'https://drive.google.com/uc?id=1ZfIv6GTo-OGpXpoKen1fUvDQ0A_WoQ-Q',
    'https://drive.google.com/uc?id=1G4eXgYS2kL7_Cc0x3N1g1x7Zl8D_aU_-',
    'https://drive.google.com/uc?id=1Y8q0V_oBwJIY27W_6-8CD1dRqV2gNTdE',
    'https://drive.google.com/uc?id=1nawBAazf_unMv46qGBHhWcQ4JXZ5883r',
]

names = [
    'BL30K_a.tar',
    'BL30K_b.tar',
    'BL30K_c.tar',
    'BL30K_d.tar',
    'BL30K_e.tar',
    'BL30K_f.tar',
]

for i, link in enumerate(links):
    print('Downloading segment %d/%d ...' % (i, len(links)))
    gdown.download(link, output='../%s' % names[i], quiet=False)
    print('Extracting...')
    with tarfile.open('../%s' % names[i], 'r') as tar_file:
        tar_file.extractall('../%s' % names[i])
    print('Cleaning up...')
    os.remove('../%s' % names[i])


print('Done.')