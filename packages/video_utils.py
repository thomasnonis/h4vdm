import os
import subprocess
import skimage.transform
import skimage.color
import numpy as np
import struct
import torch
import logging
import pickle
# import torchvision.transforms as transforms
from slice_pb2 import Slice as SliceProto
from slice_pb2 import SliceType as SliceTypeProto

from packages.constants import MACROBLOCK_SIZE
from packages.common import create_custom_logger

# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================

def get_crop_dimensions(frame_width, frame_height, crop_width, crop_height, position = 'center'):
    if position != 'center':
        raise NotImplementedError('Only center crop is supported for now')
    
    left_crop = (frame_width - crop_width) // 2
    right_crop = left_crop + crop_width
    top_crop = (frame_height - crop_height) // 2
    bottom_crop = top_crop + crop_height

    return (left_crop, right_crop, top_crop, bottom_crop)

def crop_frame(frame, crop_width, crop_height):
        # TODO: verify correct shape indeces
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # crop to center of frame
        (left_crop, right_crop, top_crop, bottom_crop) = get_crop_dimensions(frame_width, frame_height, crop_width, crop_height)

        if crop_width <= 0 or crop_height <= 0:
            raise ValueError('Crop dimensions must be positive')
        
        if crop_width > frame_width or crop_height > frame_height:
            raise ValueError('Crop dimensions are bigger than frame dimensions')
        
        # TODO: verify correct positions of x, y, channel
        if len(frame.shape) == 3:
            return frame[top_crop:bottom_crop, left_crop:right_crop, :]
        elif len(frame.shape) == 2:
            return frame[top_crop:bottom_crop, left_crop:right_crop]
        else:
            raise ValueError('Frame has an unexpected number of dimensions')
        
# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================

class H264Extractor():
    log = create_custom_logger('H264Extractor', logging.DEBUG)

    def __init__(self, bin_filename, cache_dir):
        if not os.path.exists(bin_filename):
            raise FileNotFoundError(f'cannot locate the binary file "{bin_filename}", was it built?')
        self.bin_filename = bin_filename

        cache_dir = os.path.join(cache_dir, 'h264_cache')
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
        video_name = os.path.basename(video_filename)

        h264_filename = os.path.join(self.cache_dir, video_name.split('.')[0] + '.h264')

        if os.path.exists(h264_filename):
            H264Extractor.log.info(f'H264 file for video {video_name} already exists at {h264_filename}, skipping conversion')
            return h264_filename
        
        H264Extractor.log.debug(f'Converting video {video_name} to h264')

        # extract h264 from the mp4 file using ffmpeg
        cp = subprocess.run(
                ['ffmpeg', '-y', '-i', video_filename, '-vcodec', 'copy', '-an', 
                '-bsf:v', 'h264_mp4toannexb', h264_filename, '-loglevel', 'panic'],
                check=True
            )
        
        if not os.path.exists(h264_filename):
            raise FileNotFoundError(f'cannot locate the h264 file "{h264_filename}", it probably hasn\'t been generated')

        H264Extractor.log.debug(f'H264 file for video {video_name} generated successfully at {h264_filename}')
        return h264_filename
    
    def extract_yuv_and_codes(self, h264_filename):
        # compute the filenames
        video_name = os.path.basename(h264_filename).split('.')[0]
        h264_name = os.path.basename(h264_filename)
        yuv_filename = os.path.join(self.cache_dir, video_name+ '.yuv') # YUV frames
        coded_data_filename = os.path.join(self.cache_dir, video_name + '.msg') # encoding parameters

        if os.path.exists(yuv_filename) and os.path.exists(coded_data_filename):
            H264Extractor.log.info(f'YUV and coded data files for h264 file {h264_name} already exist at {yuv_filename} and {coded_data_filename}, skipping extraction')
            return (yuv_filename, coded_data_filename)

        H264Extractor.log.debug(f'Extracting YUV and coded data from h264 file {h264_name}')


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
        
        H264Extractor.log.debug(f'YUV and coded data files for h264 file {h264_name} generated successfully at {yuv_filename} and {coded_data_filename}')
        return (yuv_filename, coded_data_filename)
    
    def clean_cache(self):
        H264Extractor.log.debug(f'Cleaning cache directory {self.cache_dir}')
        if os.path.exists(self.cache_dir):
            for files in os.listdir(self.cache_dir):
                os.remove(self.cache_dir, files)
            os.rmdir(self.cache_dir)

# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================

class Video():
    h264_extractor = None
    log = create_custom_logger('Video', logging.DEBUG)

    def __init__(self, filename: str, device: str, crop_width: int, crop_height: int, target_n_gops: int, target_gop_length: int, extract_gops_on_init: bool = False):
        if Video.h264_extractor == None:
            raise RuntimeError('H264 extractor not set, call Video.set_h264_extractor() before instantiating a Video object')
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f'cannot locate the video file "{filename}"')
        
        # Original file properties
        self.filename = filename
        self.path = os.path.dirname(filename)
        self.name = os.path.basename(filename).split('.')[0]
        self.extension = os.path.basename(filename).split('.')[1]
        self.device = device

        self.h264_filename = None
        self.yuv_filename = None
        self.coded_data_filename = None

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.target_n_gops = target_n_gops
        self.target_gop_length = target_gop_length
        self.gops = []

        if extract_gops_on_init:
            self._extract_gops()

    @staticmethod
    def set_h264_extractor(h264_extractor):
        Video.h264_extractor = h264_extractor
        Video.log.debug(f'H264 extractor set to {Video.h264_extractor.bin_filename}')
        return Video.h264_extractor
    
    @staticmethod
    def save(video, path):
        path = os.path.join(path, video.name)

        if not os.path.exists(path):
            os.makedirs(path)

        filename = os.path.join(path, video.name + '.video')
        Video.log.info(f'Saving video to {filename}')

        gops_paths = []
        for gop in video.gops:
            gops_paths.append(Gop.save(gop, os.path.join(path, 'gops')))

        video_data = (
            video.filename,
            video.device,
            video.crop_width,
            video.crop_height,
            video.target_n_gops,
            video.target_gop_length,
            gops_paths,
        )

        pickle.dump(video_data, open(filename, 'wb'))
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Error while saving gop {filename}')        
        return filename
    
    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f'cannot locate the video file "{filename}"')
        
        Video.log.info(f'Loading video from {filename}')

        video_data = pickle.load(open(filename, 'rb'))
        video = Video(video_data[0], video_data[1], video_data[2], video_data[3], video_data[4], video_data[5], extract_gops_on_init=False)

        for gop_path in video_data[6]:
            video.gops.append(Gop.load(gop_path, video))

        return video

    def _decode(self):     
        if self.coded_data_filename == None or not os.path.exists(self.coded_data_filename) or self.yuv_filename == None or not os.path.exists(self.yuv_filename):
            self.h264_filename = Video.h264_extractor.convert_to_h264(self.filename)

        (self.yuv_filename, self.coded_data_filename) = Video.h264_extractor.extract_yuv_and_codes(self.h264_filename)
        # try:
        #     os.remove(self.h264_filename)
        #     Video.log.info(f'Removed {self.h264_filename}')
        # except Exception as e:
        #     Video.log.critical(f'Unable to remove {self.h264_filename}. Continuing, but beware of your disk space!')    

    def _extract_gops(self):
        if len(self.gops) == self.target_n_gops:
            Video.log.debug(f'Video {self.name} already has {self.target_n_gops} GOPs, skipping extraction')
            return self.gops
        
        self._decode()

        slice_iterator = Slice.get_slice_iterator(self.coded_data_filename)

        current_frame_number = 0

        while len(self.gops) < self.target_n_gops:
            try:
                Video.log.debug(f'Extracting GOP {len(self.gops) + 1}/{self.target_n_gops} from video {self.name}')
                gop = Gop(self, current_frame_number, slice_iterator, extract_features_on_init=False)
                # update iterator with progress of last GOP
                slice_iterator = gop.get_slice_iterator()
                current_frame_number = gop.get_current_frame_number()
                self.gops.append(gop)
            except StopIteration:
                raise ValueError(f'Unable to reach desired number of GOPs for video {self.name}. Target was {self.target_n_gops}, actual is {len(self.gops)}')

        return self.gops
    
    def _get_rgb_frame(self, frame_number, width, height):
        """Extracts a frame from the YUV sequence and converts it to RGB.
        It assumes a YUV420 subsampling.

        Args:
            frame_number (_type_): Index of the frame to be extracted
            width (_type_): Width of the original frame, not the desired crop
            height (_type_): Height of the original frame, not the desired crop

        Returns:
            _type_: The desired frame in RGB format
        """
        self._decode()

        # In YUV420 format, each pixel of the Y (luma) component is represented by 1 byte, while the U and V (chroma) components are subsampled, so each of them is represented by 0.25 bytes. Hence, the total size is (width * height * 1.5).
        # width = 1280
        # height = 720
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
    
    def get_gops(self):
        if len(self.gops) == 0:
            self._extract_gops()

        return self.gops
    
# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
class Gop():
    log = create_custom_logger('Gop', logging.DEBUG)

    def __init__(self, video_ref: Video, current_frame_number: int, slice_iterator = None, extract_features_on_init: bool = False, extract_slices_on_init: bool = True):

        self.video_ref = video_ref
        self.current_frame_number = current_frame_number

        self.crop_width = video_ref.crop_width
        self.crop_height = video_ref.crop_height

        self.intra_frame = None
        self.inter_frames = []
        self.frame_types = []
        self.mb_types = []
        self.luma_qps = []

        if slice_iterator == None:
            self.slice_iterator = Slice.get_slice_iterator(self.video_ref.coded_data_filename)
        else:
            self.slice_iterator = slice_iterator

        if extract_slices_on_init:
            self.slices = self._extract_slices(video_ref.target_gop_length)
        else:
            self.slices = []

        if extract_features_on_init:
            self._extract_features(self.slices)

    @staticmethod
    def save(gop, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        filename = os.path.join(path, gop.video_ref.name) + f'_{gop.current_frame_number - gop.video_ref.target_gop_length + 1}.gop'
        Gop.log.info(f'Saving GOP to {filename}')

        slices_paths = []
        for slice in gop.slices:
            slices_paths.append(Slice.save(slice, os.path.join(path, 'slices')))

        gop_data = (
            gop.current_frame_number,
            slices_paths,
        )

        pickle.dump(gop_data, open(filename, 'wb'))

        if not os.path.exists(filename):
            raise FileNotFoundError(f'Error while saving gop {filename}')
        
        return filename

    @staticmethod
    def load(path: str, video_ref: Video):
        if not os.path.exists(path):
            raise FileNotFoundError(f'cannot locate the video file "{path}"')
        
        Gop.log.info(f'Loading GOP from {path}')
        gop_data = pickle.load(open(path, 'rb'))
        gop = Gop(video_ref, gop_data[0], extract_features_on_init=False, extract_slices_on_init=False)
        
        for slice_path in gop_data[1]:
            gop.slices.append(Slice.load(slice_path, gop))

        return gop

    def __eq__(self, other):
        #TODO: must check also GOP index if multiple GOPs are extracted from the same video
        return self.video_properties['filename'] == other.video_properties['filename']
    
    def __len__(self):
        return len(self.slices)

    def __getitem__():
        pass

    def _extract_slices(self, target_length: int) -> list:
        # each video has x gops and each gop has y slices
        slice_raw = next(self.slice_iterator)
        if self.current_frame_number != 0:
            self.current_frame_number += 1

        slices = []
        # The first slice is expected to be of type Intra. Find next I slice to start gop
        while(slice_raw.type != SliceTypeProto.I):
            try:
                Gop.log.debug(f'Found slice of type {slice_raw.type} at frame {self.current_frame_number} while looking for type {SliceTypeProto.I}, skipping')
                slice_raw = next(self.slice_iterator)
                self.current_frame_number += 1
            except StopIteration:
                raise ValueError('No Intra slice found')

        Gop.log.debug(f'Found Intra slice at frame {self.current_frame_number}')
        slices.append(Slice(slice_raw, self.current_frame_number, self))

        while len(slices) < target_length:
            try:
                slice_raw = next(self.slice_iterator)
            except StopIteration:
                Gop.log.critical(f'next() raised StopIteration, Video is over at frame {self.current_frame_number}')
                break
            
            self.current_frame_number += 1
            
            if slice_raw.type == SliceTypeProto.I and len(slices) < target_length:
                # GOP is over
                # TODO: find next GOP, don't raise exception (careful of very long/infinite loops)
                # TODO: will need to go to previous(iterator) in next gop
                Gop.log.critical(f'Found slice of type I before reaching target length of {target_length}, GOP is over with length {len(slices)}')
                break

            Gop.log.debug(f'Appending Inter slice {self.current_frame_number}')
            slices.append(Slice(slice_raw, self.current_frame_number, self))

        if len(slices) != target_length:
            raise ValueError(f'Unable to reach desired GOP length of {target_length}, actual gop length is {len(slices)}')
        
        Gop.log.debug(f'GOP succesfully extracted from frames {self.current_frame_number - target_length + 1} - {self.current_frame_number}')

        return slices

    def _extract_features(self):
        if len(self) == 0 or self.slices is None:
            raise ValueError('GOP not extracted yet')
        
        Gop.log.debug(f'Extracting features for GOP of video {self.video_ref.name} starting from frame {self.current_frame_number}')

        for slice in self.slices:
            # append frame type
            self.frame_types.append(slice.get_type())

            # append intra frame
            if slice.is_intra():
                self.intra_frame = slice.get_rgb_frame()
                # include difference between I frame and itself (zeros)
                self.inter_frames.append(self.intra_frame - self.intra_frame)
            # append inter frame
            else:
                self.inter_frames.append(slice.get_rgb_frame() - self.intra_frame) # abs()?
            
            # create macroblock image, where each pixel is the type of the macroblock it belongs to
            self.mb_types.append(slice.get_macroblock_image())
            
            # create luma quantization parameter image, where each pixel is the luma qp of the macroblock it belongs to
            # TODO: needs testing
            self.luma_qps.append(slice.get_luma_qp_image())

        return None
    
    def get_slice_iterator(self):
        return self.slice_iterator
    
    def get_current_frame_number(self):
        return self.current_frame_number
    
    def get_first_frame_number(self):
        return self.current_frame_number - len(self.slices) + 1
    
    def get_rgb_frame(self, frame_number: int):
        if frame_number < self.get_first_frame_number() or frame_number > self.get_current_frame_number():
            raise ValueError(f'Frame number {frame_number} is not in the GOP range [{self.get_first_frame_number()}, {self.get_current_frame_number()}]')
        
        return self.video_ref._get_rgb_frame(frame_number, self.slices[0].original_width, self.slices[0].original_height)
    
    def get_intra_frame(self):
        if self.intra_frame is None:
            self._extract_features()
        return self.intra_frame
    
    def get_inter_frames(self):
        if len(self.inter_frames) == 0:
            self._extract_features()
        return self.inter_frames
    
    def get_frame_types(self):
        if len(self.frame_types) == 0:
            self._extract_features()
        return self.frame_types
    
    def get_macroblock_images(self):
        if len(self.mb_types) == 0:
            self._extract_features()
        return self.mb_types
    
    def get_luma_qp_images(self):
        if len(self.luma_qps) == 0:
            self._extract_features()
        return self.luma_qps
        
    def is_same_device(self, other_gop) -> bool:
        return self.video_properties['device'] == other_gop.video_properties['device']
    
# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
    
class Slice:

    log = create_custom_logger('Slice', logging.DEBUG)

    def __init__(self, slice, frame_number: int, gop_ref: Gop):
        if slice is None:
            self.type = None
            self.macroblocks = None
            self.original_width = None
            self.original_height = None
        else:
            self.type = slice.type
            self.macroblocks = slice.mbs
            self.original_width = slice.width
            self.original_height = slice.height
        
        self.crop_width = gop_ref.crop_width
        self.crop_height = gop_ref.crop_height
        
        self.frame_number = frame_number
        self.video_ref = gop_ref.video_ref

        self.rgb_frame = None
        self.luma_qp_image = None
        self.macroblock_image = None

    @staticmethod
    def _get_ep_file_iterator(coded_data_filename: str):
        with open(coded_data_filename, 'rb') as file:
            file_size = os.stat(coded_data_filename).st_size
            while file.tell() < file_size:
                length_bytes = file.read(4)
                # Interpret data as little-endian unsigned int to convert from C layer to Python
                length = struct.unpack('<I', length_bytes)[0]
                yield file.read(length)

    @staticmethod
    def get_slice_iterator(coded_data_filename: str):
        iterator = Slice._get_ep_file_iterator(coded_data_filename)
        for bytes in iterator:
            slice = SliceProto()
            slice.ParseFromString(bytes)
            yield slice
    
    @staticmethod
    def save(slice, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        slice._extract_features()

        filename = os.path.join(path, slice.video_ref.name) + f'_{slice.frame_number}.slice'
        Slice.log.info(f'Saving slice to {filename}')

        slice_data = (
            slice.type,
            slice.original_width,
            slice.original_height,
            slice.crop_width,
            slice.crop_height,
            slice.frame_number,
            slice.rgb_frame,
            slice.luma_qp_image,
            slice.macroblock_image
            )

        pickle.dump(slice_data, open(filename, 'wb'))
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Error while saving gop {filename}')  
        return filename
    
    @staticmethod
    def load(filename, gop_ref: Gop):
        if not os.path.exists(filename):
            raise FileNotFoundError(f'cannot locate the video file "{filename}"')
        
        Slice.log.info(f'Loading slice from {filename}')
        slice_data = pickle.load(open(filename, 'rb'))
        slice = Slice(None, slice_data[5], gop_ref)
        slice.type = slice_data[0]
        slice.original_width = slice_data[1]
        slice.original_height = slice_data[2]
        slice.crop_width = slice_data[3]
        slice.crop_height = slice_data[4]
        slice.rgb_frame = slice_data[6]
        slice.luma_qp_image = slice_data[7]
        slice.macroblock_image = slice_data[8]
        return slice


    def get_type(self):
        return self.type

    def is_intra(self) -> bool:
        return self.get_type() == SliceTypeProto.I

    def get_rgb_frame(self):
        # TODO: remember to include also intra-intra in the list of inter frames
        if self.rgb_frame is None:
            self.rgb_frame = self.video_ref._get_rgb_frame(self.frame_number, self.original_width, self.original_height)
            self.rgb_frame = crop_frame(self.rgb_frame, self.crop_width, self.crop_height)
        
        return self.rgb_frame
    
    def _extract_features(self):
        luma_qp_image = np.zeros((self.original_height, self.original_width))
        macroblock_image = np.zeros((self.original_height, self.original_width))
        for mb in self.macroblocks:
            luma_qp_image[mb.y * MACROBLOCK_SIZE:mb.y * MACROBLOCK_SIZE + MACROBLOCK_SIZE, mb.x * MACROBLOCK_SIZE:mb.x * MACROBLOCK_SIZE + MACROBLOCK_SIZE] = mb.luma_qp
            macroblock_image[mb.y * MACROBLOCK_SIZE:mb.y * MACROBLOCK_SIZE + MACROBLOCK_SIZE, mb.x * MACROBLOCK_SIZE:mb.x * MACROBLOCK_SIZE + MACROBLOCK_SIZE] = mb.type

        self.luma_qp_image = crop_frame(luma_qp_image, self.crop_width, self.crop_height)
        self.macroblock_image = crop_frame(macroblock_image, self.crop_width, self.crop_height)
        return
    
    def get_luma_qp_image(self):
        # create luma quantization parameter image, where each pixel is the luma qp of the macroblock it belongs to
        if self.luma_qp_image is None:
            self._extract_features()
        
        return self.luma_qp_image
    
    def get_macroblock_image(self):
        # create macroblock image, where each pixel is the type of the macroblock it belongs to
        if self.macroblock_image is None:
            self._extract_features()

        return self.macroblock_image

    def get_rgb_frame_as_tensor(self):
        frame = self.get_rgb_frame()

        frame_tensor = torch.tensor((), dtype=torch.float)
        frame_tensor = frame_tensor.new_zeros((1, frame.shape[2], frame.shape[0], frame.shape[1]))
        frame_tensor[0] = torch.from_numpy(self.intra_frame).permute(2, 0, 1) # (h,w,c) -> (c,h,w)
        return frame_tensor
    
    # def get_frame_types_as_tensor(self):
    #     if self.frame_types is None:
    #         raise ValueError('GOP not extracted yet')
        
    #     frame_types_tensor = torch.tensor(self.frame_types)
    #     return frame_types_tensor

    def get_macroblock_image_as_tensor(self):
        macroblock_image = self.get_macroblock_image()
        
        mb_image_tensor = torch.tensor((), dtype=torch.float)
        mb_image_tensor = mb_image_tensor.new_zeros((1, 3, macroblock_image.shape[0], macroblock_image.shape[1]))
        converted = torch.from_numpy(macroblock_image)

        # image is grayscale, but we need 3 channels for the network. Copy the grayscale image to all 3 channels
        mb_image_tensor[0][0] = converted
        mb_image_tensor[0][1] = converted
        mb_image_tensor[0][2] = converted

        return mb_image_tensor
    
    def get_luma_qps_as_tensor(self):
        luma_qps_image = self.get_luma_qp_image()
        
        luma_qps_tensor = torch.tensor((), dtype=torch.float)
        luma_qps_tensor = luma_qps_tensor.new_zeros((1, 3, luma_qps_image.shape[0], luma_qps_image.shape[1]))
        converted = torch.from_numpy(luma_qps_image)

        # image is grayscale, but we need 3 channels for the network. Copy the grayscale image to all 3 channels
        luma_qps_tensor[0][0] = converted
        luma_qps_tensor[0][1] = converted
        luma_qps_tensor[0][2] = converted

        return luma_qps_tensor