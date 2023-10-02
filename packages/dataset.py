# URL example
# https://lesc.dinfo.unifi.it/VISION/dataset/D01_Samsung_GalaxyS3Mini/videos/flatWA/D01_V_flatWA_panrot_0001.mp4

import os

def parse_url(url):
    structure = url.split('https://lesc.dinfo.unifi.it/VISION/dataset/')[1].split('/')
    device = structure[0]
    media_type = structure[1]
    compression = structure[2]
    name = structure[3]

    return device, media_type, compression, name

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.system(f'wget -P {path} {url}')

def get_videos_structure(filename):
    videos = []
    with open(filename) as f:
        urls = f.readlines()

        for url in urls:
            url = url.split('\n')[0]
            device, media_type, compression, name = parse_url(url)

            if media_type == 'videos':
                videos.append((url, device, media_type, compression, name))
        
    return videos

if __name__ == '__main__':
    from packages.video_utils import H264Extractor, VideoHandler, Gop
    from packages.constants import GOP_SIZE, FRAME_WIDTH, FRAME_HEIGHT
    import pickle

    FILENAME = 'datasets/VISION_files.txt'
    DATASET_ROOT = 'datasets/VISION'

    project_path = os.getcwd()

    bin_path = os.path.abspath(os.path.join(project_path, 'h264-extractor', 'bin'))
    h264_ext_bin = os.path.join(bin_path, 'h264dec_ext_info')
    
    videos = get_videos_structure(FILENAME)

    for url, device, media_type, compression, name in videos:
        filename = os.path.join(DATASET_ROOT, device, media_type, compression, name)
        if not os.path.exists(filename):
            download(url, os.path.join(DATASET_ROOT, device, media_type, compression))
        

        extractor = H264Extractor(h264_ext_bin, os.path.join(project_path, '.cache'))

        h264_filename = extractor.convert_to_h264(filename)
        yuv_filename, coded_data_filename = extractor.extract_yuv_and_codes(h264_filename)

        video_handler = VideoHandler(filename, h264_filename, yuv_filename, coded_data_filename)

        gop = Gop(video_handler, GOP_SIZE, FRAME_WIDTH, FRAME_HEIGHT)

        (device, media_type, compression, name, gop)

        # delete unnecessary data