import numpy as np
import sys

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    from Tkinter import *
else:
    from tkinter import *

from PIL import ImageTk
from PIL import Image

class SensoryVisualization(object):
    def __init__(self,width=320,height=240):
        self.width = width
        self.height = height
        
        self.root = Tk()
        self.root.wm_title("Sensory Input")
        self.root_frame = Frame(self.root)
        self.canvas = Canvas(self.root_frame, borderwidth=0, highlightthickness=0,
                        width=self.width, height=self.height, bg="black")
        self.canvas.config( width=self.width, height=self.height )
        self.canvas.pack(padx=5, pady=5)
        self.root_frame.pack()
        self.reset()

    def reset(self):
        self.canvas.delete("all")
        self.image = Image.new('RGB', (self.width, self.height))
        self.photo = None
        self._image_handle = None

    def display_data(self,data):
        """ Data should be a 2D array of values (width x height) that lie between 0 and 1."""
        cmap = Image.fromarray(np.uint8(data*255))
        cmap.load()
        self.image.paste(cmap)
        # Convert to a photo for canvas use:
        self.photo = ImageTk.PhotoImage(self.image)

        # And update/create the canvas image:
        if self._image_handle is None:
            self._image_handle = self.canvas.create_image(self.width/2,
                                                          self.height/2,
                                                          image=self.photo)
        else:
            self.canvas.itemconfig(self._image_handle, image=self.photo)
        self.root.update()

        
