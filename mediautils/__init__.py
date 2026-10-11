"""Top-level package for Media Utils."""

from importlib.metadata import metadata

from mediautils.audiosync import audio_offset as audio_offset
from mediautils.audiosync import read_audio as read_audio
from mediautils.audiosync import sync_videos as sync_videos
from mediautils.audiosync import SyncResult as SyncResult
from mediautils.shotcut import project_files as project_files
from mediautils.shotcut import relink_project as relink_project

from mediautils.file_name import file_name_to_datetime as file_name_to_datetime
from mediautils.file_name import replace_datetime_prefix as replace_datetime_prefix
from mediautils.file_name import standardize_name as standardize_name
from mediautils.file_name import wa_file_name_to_date as wa_file_name_to_date
from mediautils.image import (
    dispatch_images_by_orientation as dispatch_images_by_orientation,
)
from mediautils.image import get_all_image_files as get_all_image_files
from mediautils.image import get_offset_image as get_offset_image
from mediautils.image import get_orientation as get_orientation
from mediautils.image import get_time_image as get_time_image
from mediautils.image import resize_image as resize_image
from mediautils.image import set_file_creation_time as set_file_creation_time
from mediautils.image import set_time_image as set_time_image
from mediautils.image import set_time_raw as set_time_raw
from mediautils.video import rotate_video as rotate_video
from mediautils.video import set_all_dates_video as set_all_dates_video
from mediautils.video import write_precise_start as write_precise_start
from mediautils.media import gather_yes_no as gather_yes_no
from mediautils.media import process_camera_timezone as process_camera_timezone
from mediautils.media import process_clock_shift as process_clock_shift
from mediautils.media import process_date_subdirs as process_date_subdirs
from mediautils.media import process_datetime_filenames as process_datetime_filenames
from mediautils.media import process_timezone_shift as process_timezone_shift
from mediautils.media import process_videos as process_videos
from mediautils.media import process_whatsapp_files as process_whatsapp_files
from mediautils.media import set_time as set_time


infos = metadata(__name__)
__version__ = infos["Version"]
__author__ = "François Durand"
__email__ = "fradurandpub@gmail.com"
