#### IMPORTS ####
from typing import BinaryIO

#### CONSTANTS ####
SEEK_FROM_BEGINNING = 0
SEEK_FROM_CURRENT_POS = 1
SEEK_FROM_END = 2

VIDEO_PATH = '../data/night-sky.h264'

#### FUNCTIONS ####

# The implementation with shift == None is stupidly inefficient, but it's nice so I'll leave it here
def mask(word: int, mask: int = None, shift: int = None) -> int:
	if mask is None:
		mask = 0xFFFF
		shift = 0
	
	elif shift is None:
		shift = 0
		tmp_mask = mask
		while tmp_mask >> 1 == tmp_mask / 2:
			tmp_mask >>= 1
			shift += 1

			if shift > 15:
				raise Exception(f'Automatic shift calculation failed. Input is {word} and shift is {shift}')

	return (word & mask) >> shift

def read_bytes(file: BinaryIO, num_bytes: int) -> tuple:
	"""Wrapper function for reading bytes from a file and returning the read data, the location where the read started
		and the location where the read ended (the byte after the last read byte)

	Args:
		file (BinaryIO): binary file handle
		num_bytes (int): number of bytes to read

	Returns:
		tuple: tuple containing the read data, the location where the read started and the location where the read ended (the byte after the last read byte)
	"""	
	start_ptr = file.tell()
	data = file.read(num_bytes)
	end_ptr = file.tell()
	# print('Read {} bytes from {} to {} - {}'.format(end_ptr - start_ptr, start_ptr, end_ptr, data))
	return data, start_ptr, end_ptr

def find_start_code_locations(file: BinaryIO, n_limit: int = -1) -> list:
	"""Finds the start code locations in an h264 file and saves the start and end pointers of each start code in a list.
		The implemetation is quite stupid and inefficient, but it works :).

	Args:
		file (BinaryIO): file handle to the h264 file
		n_limit (int, optional): number of desired start codes to limit the execution, set to -1 to disable the limit. Defaults to -1.

	Returns:
		list: list of tuples containing the start and end pointers of each start code
	"""	

	start_code_locations = []

	#TODO: make more efficient
	while True:
		data, start_ptr, end_ptr = read_bytes(file, 4)
		if data == '':
			break

		if  n_limit > 0 and len(start_code_locations) > n_limit:
			break

		if data == b'\x00\x00\x00\x01':
			# print('Found 4-byte start code at {}'.format(start_ptr))
			start_code_locations.append((start_ptr, end_ptr))
		elif data[:-2] == b'\x00\x00\x00':
			next_byte, _, end_ptr = read_bytes(file, 1)
			if next_byte == b'\x01':
				# print('Found 5-byte start code at {}'.format(start_ptr))
				start_code_locations.append((start_ptr, end_ptr))
			else:
				# print('No start code found')
				file.seek(-3, SEEK_FROM_CURRENT_POS) #go forward one byte
		else:
			# print('No start code found')
			file.seek(-3, SEEK_FROM_CURRENT_POS) #go forward one byte
	#TODO: fix redundant else

	return start_code_locations

def extract_nalus(file: BinaryIO, start_code_locations: list) -> list:
	"""Extracts N-1 NALUs from an h264 file, where N is the number of start code locations.
		The last start code location is only used to determine the length of the last NALU.

	Args:
		file (BinaryIO): file handle to the h264 file
		start_code_locations (list): list of tuples containing the start and end pointers of each start code

	Returns:
		list: list of raw NALUs
	"""
	nalus = []
	for idx, start_code_location in enumerate(start_code_locations):
		if idx < len(start_code_locations) - 1:
			length = start_code_locations[idx + 1][0] - start_code_location[1] # exclude start code of next NALU
			file.seek(start_code_location[1])
			data, start_ptr, end_ptr = read_bytes(file, length)
			nalus.append(data)

	return nalus

#### CLASSES ####

class NALU(object):
	def from_list(raw_nalus: list):
		return [NALU(raw_nalu) for raw_nalu in raw_nalus]
	
	def __init__(self, raw_data: bytes):
		self.header = int.from_bytes(raw_data[:1], byteorder='big')
		self.rbsp = raw_data[1:] # payload

		self.__parse_header()
		
	def __parse_header(self):
		self.forbidden_zero_bit = mask(self.header, 0b1000_0000, 7)
		if self.forbidden_zero_bit != 0:
			raise ValueError('Forbidden zero bit is not zero')
		
		self.nal_ref_idc = mask(self.header, 0b0110_0000, 5)
		self.nal_unit_type = mask(self.header, 0b0001_1111, 0)

	def __find_rbsp_stop_bit(self):
		last_byte = self.rbsp[-1]

		for i in range(0, 8, 1):
			if mask(last_byte, 1 << i, i) == 1:
				print('Found stop bit at bit {} from end ({} trailing zeros)'.format(i, i))
				return
			
	def get_nal_structure_description(self):
		if self.nal_unit_type in range(1, 24): # [1, 23]
			return 'Single NAL Unit Packets'
		if self.nal_unit_type in range(24, 28): # [24, 27]
			return 'Aggregation Packet'
		if self.nal_unit_type in [20, 29]:
			return 'Fragmentation Unit'
		

	def get_nal_unit_type_description(self):
		if self.nal_unit_type == 0:
			return 'Unspecified'
		if self.nal_unit_type == 1:
			return 'Coded slice of a non-IDR picture'
		if self.nal_unit_type == 2:
			return 'Coded slice data partition A'
		if self.nal_unit_type == 3:
			return 'Coded slice data partition B'
		if self.nal_unit_type == 4:
			return 'Coded slice data partition C'
		if self.nal_unit_type == 5:
			return 'Coded slice of an IDR picture'
		if self.nal_unit_type == 6:
			return 'Supplemental enhancement information (SEI)'
		if self.nal_unit_type == 7:
			return 'Sequence parameter set'
		if self.nal_unit_type == 8:
			return 'Picture parameter set'
		if self.nal_unit_type == 9:
			return 'Access unit delimiter'
		if self.nal_unit_type == 10:
			return 'End of sequence'
		if self.nal_unit_type == 11:
			return 'End of stream'
		if self.nal_unit_type == 12:
			return 'Filler data'
		if self.nal_unit_type == 13:
			return 'Sequence parameter set extension'
		if self.nal_unit_type == 14:
			return 'Prefix NAL unit'
		if self.nal_unit_type == 15:
			return 'Subset sequence parameter set'
		if self.nal_unit_type >= 16 and self.nal_unit_type <= 18:
			return 'Reserved'
		if self.nal_unit_type == 19:
			return 'Coded slice of an auxiliary coded picture without partitioning'
		if self.nal_unit_type == 20:
			return 'Coded slice extension'
		if self.nal_unit_type == 21:
			return 'Coded slice extension for depth view components'
		if self.nal_unit_type >= 22 and self.nal_unit_type <= 23:
			return 'Reserved'
		# RTP packets
		if self.nal_unit_type == 24:
			return 'Single-Time Aggregation Packet type A (STAP-A)'
		if self.nal_unit_type == 25:
			return 'Single-Time Aggregation Packet type B (STAP-B)'
		if self.nal_unit_type == 26:
			return 'Multi-Time Aggregation Packet with 16-bit offset (MTAP16)'
		if self.nal_unit_type == 27:
			return 'Multi-Time Aggregation Packet with 24-bit offset (MTAP24)'
		if self.nal_unit_type == 28:
			return 'Fragmentation unit A'
		if self.nal_unit_type == 29:
			return 'Fragmentation unit B'
		if self.nal_unit_type >= 30 and self.nal_unit_type <= 31:
			return 'Reserved'
		
		raise ValueError('Invalid NAL unit type')

	def get_nal_transport_priority(self):
		return self.nal_ref_idc


	def is_reference(self) -> bool:
		return self.nal_ref_idc != 0
	
	def is_vcl(self) -> bool:
		return self.nal_unit_type in range(1, 6) # [1 - 5] inclusive

#### MAIN ####

if __name__ == '__main__':
	with open(VIDEO_PATH, 'rb') as f:
		start_code_locations = find_start_code_locations(f, 10)

		raw_nalus = extract_nalus(f, start_code_locations)

		nalus = NALU.from_list(raw_nalus)