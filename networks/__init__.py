import importlib
import os
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import arch modules for registry
# scan all the files under the 'networks' folder and collect files ending with '_arch.py'
arch_root_folder = osp.dirname(osp.abspath(__file__))
arch_folder_list = [v for v in os.scandir(arch_root_folder) if osp.isdir(v)]
arch_filename_list = [[folder.name, osp.splitext(osp.basename(v))[0]] for folder in arch_folder_list for v in scandir(folder)
                      if (v.startswith('generator') or v.startswith('discriminator'))]
# import all the arch modules
_arch_modules = [importlib.import_module(f'networks.{file_name[0]}.{file_name[1]}') for file_name in arch_filename_list]

