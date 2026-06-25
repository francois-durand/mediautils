"""Top-level package for Media Utils."""

from importlib.metadata import metadata

from mediautils.file_name import file_name_to_datetime as file_name_to_datetime
from mediautils.file_name import wa_file_name_to_date as wa_file_name_to_date
from mediautils.image import get_all_jpeg_files as get_all_jpeg_files
from mediautils.image import get_orientation as get_orientation
from mediautils.image import process_images as process_images
from mediautils.image import resize_image as resize_image
from mediautils.image import set_time_image as set_time_image
from mediautils.video import rotate_video as rotate_video
from mediautils.media import process_directory_files as process_directory_files
from mediautils.media import process_standard_files as process_standard_files
from mediautils.media import process_wa_files as process_wa_files
from mediautils.media import set_time as set_time


infos = metadata(__name__)
__version__ = infos["Version"]
__author__ = "François Durand"
__email__ = "fradurandpub@gmail.com"
