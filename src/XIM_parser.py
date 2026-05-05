from pylinac.core.image import XIM
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

""" 
We assume that the XIM files are stored in a folder with the following structure:
Folder
|--- TopLayer
|--- BottomLayer
and that the corresponding XIM files are named the same in both folders 
"""

def XIM_to_JLD2_file(folder_path, image_name):

    top_folder =  os.path.join(folder_path, "TopLayer")
    bottom_folder = os.path.join(folder_path, "BottomLayer")

    # Read the XIM file using pylinac
    top_image = XIM(os.path.join(top_folder, image_name))
    bottom_image = XIM(os.path.join(bottom_folder, image_name))
    
    # Save the image array to a JLD2 file using h5py with metadata in groups
    with h5py.File(os.path.join(top_folder, image_name.replace('.xim', '') + ".jld2"), 'w') as f:
        f.create_dataset("TopLayer", data=top_image.array)
        metadata_group = f.create_group("metadata")
        for key, value in top_image.properties.items():
            try:
                metadata_group[key] = value
            except (TypeError, ValueError):
                metadata_group[key] = str(value)
        
    with h5py.File(os.path.join(bottom_folder, image_name.replace('.xim', '') + ".jld2"), 'w') as f:
        f.create_dataset("BottomLayer", data=bottom_image.array)
        metadata_group = f.create_group("metadata")
        for key, value in bottom_image.properties.items():
            try:
                metadata_group[key] = value
            except (TypeError, ValueError):
                metadata_group[key] = str(value)


def XIM_to_JLD2(folder_path):
    
        top_folder =  os.path.join(folder_path, "TopLayer") # could use the bottom folder as well
        
        images_names = os.listdir(top_folder)
        images_names = [f for f in images_names if f.endswith('.xim')]
        #image_name = images_names[0]

        for image_name in images_names:
            XIM_to_JLD2_file(folder_path, image_name)

"""
folder_path_main = "/Users/fd86/Documents/julia_code/Prism/data/Aquisitions_3_20_2026/Pelvic_phantom_outputs/trig_noshift_beam1_output/"
folder_list = os.listdir(folder_path_main)
#folder_list = [os.path.join(folder_path_main, f) for f in folder_list if f.startswith("output")]
folder_list = [os.path.join(folder_path_main, f) for f in folder_list]
for folder_path in folder_list:
    XIM_to_JLD2(folder_path)
 """

"""
Be careful, the function applies the negative logarithm to the pixel values and normalizes the images between 0 and 1.
"""
def XIM_to_png_file(folder_path, image_name):

    top_folder =  os.path.join(folder_path, "TopLayer")
    bottom_folder = os.path.join(folder_path, "BottomLayer")

    # Read the XIM file using pylinac
    top_image = XIM(os.path.join(top_folder, image_name))
    bottom_image = XIM(os.path.join(bottom_folder, image_name))

    top_image = -np.log(top_image.array) 
    bottom_image = -np.log(bottom_image.array)

    top_image = (top_image - np.min(top_image)) / (np.max(top_image) - np.min(top_image))
    bottom_image = (bottom_image - np.min(bottom_image)) / (np.max(bottom_image) - np.min(bottom_image))
    
    # Save the image array as PNG files using matplotlib

    plt.imsave(os.path.join(top_folder, image_name.replace('.xim', '') + ".png"), top_image, cmap='gray')
    plt.imsave(os.path.join(bottom_folder, image_name.replace('.xim', '') + ".png"), bottom_image, cmap='gray')


def XIM_to_png(folder_path):
    
        top_folder =  os.path.join(folder_path, "TopLayer") # could use the bottom folder as well
        
        images_names = os.listdir(top_folder)
        images_names = [f for f in images_names if f.endswith('.xim')]

        for image_name in images_names:
            XIM_to_png_file(folder_path, image_name)

""" folder_path = "/Users/fd86/Downloads/kV_Leeds_yellow button/___Processed/866827012/"
image_name = "Frame00818.xim"
file_path_tor = "/Users/fd86/Downloads/kV_Leeds_yellow button/___Processed/866827012/BottomLayer/Frame00818.xim"
file_tor = XIM(file_path_tor)
vars(file_tor)

file_tor.plot()
file_tor.properties["AcquisitionNotes"]

file_path_tor2 = "/Users/fd86/Downloads/kV_Leeds_yellow button/___Processed/866827015/BottomLayer/Frame01938.xim"
file_tor2 = XIM(file_path_tor2)
vars(file_tor2)

folder_path = "/Users/fd86/Downloads/Thorax Patient1/"
image_name = "Frame05138.xim"
file_path = "/Users/fd86/Downloads/Thorax Patient1/DLI L1 125kVp 270mAs HalfFan (Top)/Frame05138.xim"
file = XIM(file_path)
vars(file)


folder_path_1 = "../data/Patient_21/output_20260205170203/881753812/"
folder_path_2 = "../data/Patient_21/output_20260205170231/881753813/"
XIM_to_JLD2(folder_path_1)
XIM_to_JLD2(folder_path_2) """