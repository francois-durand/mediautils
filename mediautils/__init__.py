"""Top-level package for Media Utils."""

from importlib.metadata import metadata

from mediautils.file_name import file_name_to_datetime as file_name_to_datetime
from mediautils.file_name import replace_datetime_prefix as replace_datetime_prefix
from mediautils.file_name import wa_file_name_to_date as wa_file_name_to_date
from mediautils.image import get_all_jpeg_files as get_all_jpeg_files
from mediautils.image import get_offset_image as get_offset_image
from mediautils.image import get_orientation as get_orientation
from mediautils.image import get_time_image as get_time_image
from mediautils.image import process_images as process_images
from mediautils.image import resize_image as resize_image
from mediautils.image import set_file_creation_time as set_file_creation_time
from mediautils.image import set_time_image as set_time_image
from mediautils.image import set_time_raw as set_time_raw
from mediautils.video import rotate_video as rotate_video
from mediautils.media import gather_yes_no as gather_yes_no
from mediautils.media import process_camera_timezone as process_camera_timezone
from mediautils.media import process_clock_shift as process_clock_shift
from mediautils.media import process_date_subdirs as process_date_subdirs
from mediautils.media import process_datetime_filenames as process_datetime_filenames
from mediautils.media import process_timezone_shift as process_timezone_shift
from mediautils.media import process_whatsapp_files as process_whatsapp_files
from mediautils.media import set_time as set_time


infos = metadata(__name__)
__version__ = infos["Version"]
__author__ = "François Durand"
__email__ = "fradurandpub@gmail.com"
