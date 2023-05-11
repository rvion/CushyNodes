# __init__.py
from .Cushy_Nodes import NODE_CLASS_MAPPINGS
import sys
import os
import importlib
import glob

#Change this if you just want the nodes and do not plan on using Cushy Studio
activate_apis = True

# Get the absolute path of the __init__.py file
init_file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(init_file_path)

if (activate_apis):
    from . import Api

# Import and execute all installation files in the /install directory
install_directory = os.path.join(base_directory, "install")
install_files = glob.glob(os.path.join(install_directory, "*.py"))

for install_file in install_files:
    install_module_name = os.path.splitext(os.path.basename(install_file))[0]
    install_module = importlib.import_module(f".install.{install_module_name}", package=__name__)
    install_module.install()

__all__ = ['NODE_CLASS_MAPPINGS']
