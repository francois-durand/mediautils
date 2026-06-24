"""Top-level package for Media Utils."""

from importlib.metadata import metadata

from mediautils.file_name import file_name_to_datetime as file_name_to_datetime
from mediautils.image import set_time_image as set_time_image
from mediautils.sub_package_1.my_class_1 import MyClass1 as MyClass1
from mediautils.sub_package_2.my_class_2 import MyClass2 as MyClass2
from mediautils.sub_package_2.my_class_3 import MyClass3 as MyClass3


infos = metadata(__name__)
__version__ = infos["Version"]
__author__ = "François Durand"
__email__ = "fradurandpub@gmail.com"
