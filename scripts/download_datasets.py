import os
import gdown
import zipfile
from scripts import resize_youtube


LICENSE = """
These are either re-distribution of the original datasets or derivatives (through simple processing) of the original datasets. 
Please read and respect their licenses and terms before use. 
You should cite the original papers if you use any of the datasets.

For BL30K, see download_bl30k.py

Links:
DUTS: http://saliencydetection.net/duts
HRSOD: https://github.com/yi94code/HRSOD
FSS: https://github.com/HKUSTCV/FSS-1000
ECSSD: https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html
BIG: https://github.com/hkchengrex/CascadePSP

YouTubeVOS: https://youtube-vos.org
DAVIS: https://davischallenge.org/
BL30K: https://github.com/hkchengrex/MiVOS
Long-Time Video: https://github.com/xmlyqing00/AFB-URR
"""

print(LICENSE)
print('Datasets will be downloaded and extracted to ../YouTube, ../YouTube2018, ../static, ../DAVIS, ../long_video_set')
reply = input('[y] to confirm, others to exit: ')
if reply != 'y':
    exit()


"""
Static image data
"""
os.makedirs('../static', exist_ok=True)
print('Downloading static datasets...')
gdown.download('https://drive.google.com/uc?id=1wUJq3HcLdN-z1t4CsUhjeZ9BVDb9YKLd', output='../static/static_data.zip', quiet=False)
print('Extracting static datasets...')
with zipfile.ZipFile('../static/static_data.zip', 'r') as zip_file:
    zip_file.extractall('../static/')
print('Cleaning up static datasets...')
os.remove('../static/static_data.zip')


"""
DAVIS dataset
"""
# Google drive mirror: https://drive.google.com/drive/folders/1hEczGHw7qcMScbCJukZsoOW4Q9byx16A?usp=sharing
os.makedirs('../DAVIS/2017', exist_ok=True)

print('Downloading DAVIS 2016...')
gdown.download('https://drive.google.com/uc?id=198aRlh5CpAoFz0hfRgYbiNenn_K8DxWD', output='../DAVIS/DAVIS-data.zip', quiet=False)

print('Downloading DAVIS 2017 trainval...')
gdown.download('https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d', output='../DAVIS/2017/DAVIS-2017-trainval-480p.zip', quiet=False)

print('Downloading DAVIS 2017 testdev...')
gdown.download('https://drive.google.com/uc?id=1fmkxU2v9cQwyb62Tj1xFDdh2p4kDsUzD', output='../DAVIS/2017/DAVIS-2017-test-dev-480p.zip', quiet=False)

print('Downloading DAVIS 2017 scribbles...')
gdown.download('https://drive.google.com/uc?id=1JzIQSu36h7dVM8q0VoE4oZJwBXvrZlkl', output='../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip', quiet=False)

print('Extracting DAVIS datasets...')
with zipfile.ZipFile('../DAVIS/DAVIS-data.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/')
os.rename('../DAVIS/DAVIS', '../DAVIS/2016')

with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-trainval-480p.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
os.rename('../DAVIS/2017/DAVIS', '../DAVIS/2017/trainval')

with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-test-dev-480p.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
os.rename('../DAVIS/2017/DAVIS', '../DAVIS/2017/test-dev')

print('Cleaning up DAVIS datasets...')
os.remove('../DAVIS/2017/DAVIS-2017-trainval-480p.zip')
os.remove('../DAVIS/2017/DAVIS-2017-test-dev-480p.zip')
os.remove('../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip')
os.remove('../DAVIS/DAVIS-data.zip')


"""
YouTubeVOS dataset
"""
os.makedirs('../YouTube', exist_ok=True)
os.makedirs('../YouTube/all_frames', exist_ok=True)

print('Downloading YouTubeVOS train...')
gdown.download('https://drive.google.com/uc?id=13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4', output='../YouTube/train.zip', quiet=False)
print('Downloading YouTubeVOS val...')
gdown.download('https://drive.google.com/uc?id=1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t', output='../YouTube/valid.zip', quiet=False)
print('Downloading YouTubeVOS all frames valid...')
gdown.download('https://drive.google.com/uc?id=1rWQzZcMskgpEQOZdJPJ7eTmLCBEIIpEN', output='../YouTube/all_frames/valid.zip', quiet=False)

print('Extracting YouTube datasets...')
with zipfile.ZipFile('../YouTube/train.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube/')
with zipfile.ZipFile('../YouTube/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube/')
with zipfile.ZipFile('../YouTube/all_frames/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube/all_frames')

print('Cleaning up YouTubeVOS datasets...')
os.remove('../YouTube/train.zip')
os.remove('../YouTube/valid.zip')
os.remove('../YouTube/all_frames/valid.zip')

print('Resizing YouTubeVOS to 480p...')
resize_youtube.resize_all('../YouTube/train', '../YouTube/train_480p')

# YouTubeVOS 2018
os.makedirs('../YouTube2018', exist_ok=True)
os.makedirs('../YouTube2018/all_frames', exist_ok=True)

print('Downloading YouTubeVOS2018 val...')
gdown.download('https://drive.google.com/uc?id=1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr', output='../YouTube2018/valid.zip', quiet=False)
print('Downloading YouTubeVOS2018 all frames valid...')
gdown.download('https://drive.google.com/uc?id=1yVoHM6zgdcL348cFpolFcEl4IC1gorbV', output='../YouTube2018/all_frames/valid.zip', quiet=False)

print('Extracting YouTube2018 datasets...')
with zipfile.ZipFile('../YouTube2018/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube2018/')
with zipfile.ZipFile('../YouTube2018/all_frames/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube2018/all_frames')

print('Cleaning up YouTubeVOS2018 datasets...')
os.remove('../YouTube2018/valid.zip')
os.remove('../YouTube2018/all_frames/valid.zip')


"""
Long-Time Video dataset
"""
os.makedirs('../long_video_set', exist_ok=True)
print('Downloading long video dataset...')
gdown.download('https://drive.google.com/uc?id=100MxAuV0_UL20ca5c-5CNpqQ5QYPDSoz', output='../long_video_set/LongTimeVideo.zip', quiet=False)
print('Extracting long video dataset...')
with zipfile.ZipFile('../long_video_set/LongTimeVideo.zip', 'r') as zip_file:
    zip_file.extractall('../long_video_set/')
print('Cleaning up long video dataset...')
os.remove('../long_video_set/LongTimeVideo.zip')


print('Done.')