import os
import subprocess
import skimage.transform
import numpy as np
import struct
from slice_pb2 import Slice
from slice_pb2 import SliceType


class H264Extractor():
    def __init__(self, bin_filename, cache_dir):
        if not os.path.exists(bin_filename):
            raise FileNotFoundError(f'cannot locate the binary file "{bin_filename}", was it built?')
        self.bin_filename = bin_filename

        if not os.path.exists(cache_dir):
            os.makedirs(os.path.join(cache_dir))
        self.cache_dir = cache_dir


    def convert_to_h264(self, video_filename):
        """converts a given mp4 video file to an h264 annex b file.
        The h264 file will be saved in the cache folder provided in the constructor

        Args:
            video_filename (str): full path to the mp4 video file

        Raises:
            FileNotFoundError: if the h264 is not generated correctly

        Returns:
            str: full path to the h264 file
        """
        
        video_name = os.path.basename(video_filename).split('.')[0]

        h264_filename = os.path.join(self.cache_dir, video_name + '.h264')

        # extract h264 from the mp4 file using ffmpeg
        cp = subprocess.run(
                ['ffmpeg', '-y', '-i', video_filename, '-vcodec', 'copy', '-an', 
                '-bsf:v', 'h264_mp4toannexb', h264_filename],
                check=True
            )
        
        if not os.path.exists(h264_filename):
            raise FileNotFoundError(f'cannot locate the h264 file "{h264_filename}", it probably hasn\'t been generated')

        return h264_filename
    
    def extract_yuv_and_codes(self, h264_filename):
        # compute the filenames
        video_name = os.path.basename(h264_filename).split('.')[0]
        yuv_filename = os.path.join(self.cache_dir, video_name+ '.yuv') # YUV frames
        coded_data_filename = os.path.join(self.cache_dir, video_name + '.msg') # encoding parameters

        # run the extractor to get the yuv and coded data files
        cp = subprocess.run(
                [self.bin_filename, h264_filename, '--yuv_out', yuv_filename, '--info_out', coded_data_filename, '--n_threads', '0'],
                # for now, only setting threads to 0 is allowed; using other values can result in 
                # unexpected behaviors
                check=True
            )
        
        # remove the h264 file since it's not needed anymore
        # os.remove(h264_filename)
        # if os.path.exists(h264_filename):
        #     print(f'WARNING: could not remove the h264 file "{h264_filename}"')

        # raise exception if the files haven't been generated
        if not os.path.exists(yuv_filename) or not os.path.exists(coded_data_filename):
            raise FileNotFoundError(f'cannot locate the yuv and/or coded data files, they probably haven\'t been generated')
        return (yuv_filename, coded_data_filename)
    
    def clean_cache(self):
        if os.path.exists(self.cache_dir):
            for files in os.listdir(self.cache_dir):
                os.remove(self.cache_dir, files)
            os.rmdir(self.cache_dir)


class VideoHandler():   
    def __init__(self, video_filename, h264_filename, yuv_filename, coded_data_filename):
        if not os.path.exists(video_filename):
            raise FileNotFoundError(f'cannot locate the video file "{video_filename}"')
        # Original file properties
        self.filename = video_filename
        self.path = os.path.dirname(video_filename)
        self.name = os.path.basename(video_filename).split('.')[0]
        self.extension = os.path.basename(video_filename).split('.')[1]

        # Elaborated files properties
        if not os.path.exists(h264_filename):
            raise FileNotFoundError(f'cannot locate the h264 file "{h264_filename}"')
        self.h264_filename = h264_filename

        if not os.path.exists(yuv_filename):
            raise FileNotFoundError(f'cannot locate the yuv file "{yuv_filename}"')
        self.yuv_filename = yuv_filename

        if not os.path.exists(coded_data_filename):
            raise FileNotFoundError(f'cannot locate the coded data file "{coded_data_filename}"')
        self.coded_data_filename = coded_data_filename
    
    def get_rgb_frame(self, frame_number, width, height):
        # In YUV420 format, each pixel of the Y (luma) component is represented by 1 byte, while the U and V (chroma) components are subsampled, so each of them is represented by 0.25 bytes. Hence, the total size is (width * height * 1.5).
        width = 1280
        height = 720
        frame_size = int(width * height * 1.5)
        
        # Color space conversion constants
        U_MAX = 0.436
        V_MAX = 0.615

        with open(self.yuv_filename, 'rb') as yuv_sequence:
            # Read the frame at the specified frame number
            yuv_sequence.seek(frame_number * frame_size)
            
            y = np.frombuffer(yuv_sequence.read(width * height), dtype=np.uint8).reshape((height, width))
            u = np.frombuffer(yuv_sequence.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))
            v = np.frombuffer(yuv_sequence.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))

            # Rescale subsampled chroma components to the same size as the luma component
            y = skimage.img_as_float32(y)
            u = skimage.transform.rescale(u, 2.0, 1, anti_aliasing=False)
            v = skimage.transform.rescale(v, 2.0, 1, anti_aliasing=False)

            # Color space conversion
            u = (u * 2 * U_MAX) - U_MAX
            v = (v * 2 * V_MAX) - V_MAX

            # Convert to RGB
            yuv_frame = np.dstack([y, u, v])
            rgb_frame = skimage.color.yuv2rgb(yuv_frame)
        return rgb_frame
    
class Gop():
    def __init__(self, video_handler: VideoHandler, gop_length: int, width: int = 0, height: int = 0):

        self.video_handler = video_handler

        self.gop = []
        self.length = 0

        self.frame_height = 0
        self.frame_width = 0

        self.intra_frame = None
        self.inter_frames = []
        self.frame_types = []
        self.mb_types = []
        self.luma_qps = []

        self.extract_gop(gop_length, width, height)

    def _get_ep_file_iterator(self):
        with open(self.video_handler.coded_data_filename, 'rb') as file:
            file_size = os.stat(self.video_handler.coded_data_filename).st_size
            while file.tell() < file_size:
                length_bytes = file.read(4)
                # Interpret data as little-endian unsigned int to convert from C layer to Python
                length = struct.unpack('<I', length_bytes)[0]
                yield file.read(length)

    def _get_slice_iterator(self):
        iterator = self._get_ep_file_iterator()
        for bytes in iterator:
            slice = Slice()
            slice.ParseFromString(bytes)
            yield slice

    def _extract_features(self):
        if self.length == 0 or self.gop is None:
            raise ValueError('GOP not extracted yet')
        
        for slice, frame_number in self.gop:
            self.frame_types.append(slice.type)
            if slice.type == SliceType.I:
                self.intra_frame = self.video_handler.get_rgb_frame(frame_number, slice.width, slice.height)
                # include difference between I frame and itself (zeros)
                self.inter_frames.append(self.intra_frame - self.intra_frame)
            else:
                self.inter_frames.append(self.video_handler.get_rgb_frame(frame_number, slice.width, slice.height) - self.intra_frame) # abs()?
            for mb in slice.mbs:
                self.mb_types.append(mb.type)
                self.luma_qps.append(mb.luma_qp)

        return self

    def extract_gop(self, target_length: int, width: int = 0, height: int = 0) -> list:
        # TODO: how to crop?
        slice_iterator = self._get_slice_iterator()

        slice = next(slice_iterator)
        frame_index = 0
        if slice.type != SliceType.I:
            # The first slice is expected to be of type Intra. Find next I slice to start gop
            while(slice.type != SliceType.I):
                try:
                    slice = next(slice_iterator)
                    frame_index += 1
                except StopIteration:
                    raise ValueError('No Intra slice found')

        self.gop.append((slice, frame_index))

        while len(self.gop) < target_length:
            try:
                slice = next(slice_iterator)
                frame_index += 1
                if slice.type == SliceType.I:
                    # GOP is over
                    break
                else:
                    self.gop.append((slice, frame_index))
            except StopIteration:
                if len(self.gop) < target_length:
                    raise ValueError(f'Unable to reach desired GOP length of {target_length}, actual gop length is {len(self.gop)}')
                else:
                    break
                
        self.length = len(self.gop)
        self._extract_features()

        return self
    
    def get_rgb_frame(self, frame_number):
        if self.length == 0:
            raise ValueError('GOP not extracted yet')
        
        if frame_number == 0:
            return self.intra_frame
        elif frame_number > 0 and frame_number < self.length:
            return self.inter_frames[frame_number - 1]
        else:
            raise ValueError(f'Frame number {frame_number} is out of range.')