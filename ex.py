import pandas as pd
from PIL import Image
# import image
def Image_openn():
    
    img=Image.open('images/f14f73e9-143c-44f2-bc3e-0f30d85c9b84/f14f73e9-143c-44f2-bc3e-0f30d85c9b84-a.png')
    # img.show()
    print(img.format)
    print(img.size)
    resized_image=img.resize((360,240))
    resized_image.show()
    r1=img.transpose(Image.Transpose.ROTATE_180)
    r1.show()
   
    print(resized_image.size)
   
if __name__=='__main__':
    Image_openn()
