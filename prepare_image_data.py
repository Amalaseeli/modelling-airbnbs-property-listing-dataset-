from PIL import Image
import os
from os import listdir

folder_dir='C:/Users/admin/Documents/amala/Career_amala/Aicore/modelling-airbnbs-property-listing-dataset-/images'
def get_image_folders(folder_dir):
    folders=os.listdir(folder_dir)
    return folders

def get_image_file_path(imagedir):
    image_file_list=[]
    root=f"images/{imagedir}"
    pattern="*.png"   
    print(os.walk(root))
    for path,subdir,files in os.walk(root):
        # print(os.walk(root))
        for name in files:
            image_file_list.append(os.path.join(''.join(path.split('/')[1:]),name))
    return image_file_list        
    print( image_file_list)

  

def resize_images(fp):
    filepath='C:/Users/admin/Documents/amala/Career_amala/Aicore/modelling-airbnbs-property-listing-dataset-/images/'+fp
    img=Image.open(filepath)
    img.show()


if __name__=='__main__':
   imagedir= get_image_folders(folder_dir)
   for folder in imagedir:
       image_list=get_image_file_path(folder)
       for image in image_list:
           resize_images(image)
    

