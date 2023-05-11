# __init__.py
import sys
import os
import importlib.util
import glob


# Get the absolute path of the __init__.py file
init_file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(init_file_path)
install_directory = os.path.join(base_directory, "install")


# Import and execute all installation files in the /install directory
install_files = glob.glob(os.path.join(install_directory, "*.py"))

print(str(install_files))
for install_file in install_files:
    print(f"Installing from {install_file} ")
    install_module_name = os.path.splitext(os.path.basename(install_file))[0]
    
    spec = importlib.util.spec_from_file_location(install_module_name, install_file)
    install_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(install_module)

    install_module.install()


#Change this if you just want the nodes and do not plan on using Cushy Studio
activate_apis = True
if (activate_apis):
    #from . import Api
    pass

from .Cushy_Nodes import NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']
