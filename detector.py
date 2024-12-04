## command line tool to execute program
import os
import sys
from BoundingBox import BoundingBox
from PackageExtractor import PackageExtractor
from PIL import Image
from matplotlib import pyplot as plt

# default paremeters
default_model = "./783-Pin-Detection/runs/detect/train56/weights/best.pt"

def help():
    print("python3 detector.py <filename> <args>\n"
          "  Available arguments:\n"
          "    -save                          : Saves both output image and text; default filenames are output.png and output.txt\n"
          "    -ifile <filename>              : Saves the output image to the filename given. -save option must be called\n"
          "    -tfile <filename>              : Saves the text output in plaintext to the filename given\n"
          "    -show                          : Shows the output as a pyplot\n"
          "    -model <model>                 : Runs a different model to the default\n"
          "    -no-text                       : Returns no text\n"
          "    -no-image                      : Returns no image\n"
          "    -overlay-good-pin <color code> : Overlays good pins with some color\n"
          "    -overlay-bent-pin <color code> : Overlays bent pins with some color\n"
          "    -overlay-package  <color code> : Overlays the package with some color\n"
          "    -overlay-text     <color code> : Overlays text with some color\n"
          "    -help                          : Brings up the help menu (this menu)\n\n"
          "\n"
          "  Color codes: c1, c2, c3, c4, c5, c6\n\n"
          "  Example:\n"
          "    The following example displays the bounding box overlayed on the image.\n"
          "    It does not save the example.\n\n"
          "        python3 detector.py image.png\n\n"
          "  Example:\n"
          "    The following example displays the bounding box overlayed on the image.\n"
          "    It does not save the example.\n\n"
          "        python3 detector.py image.png -show\n\n"
          "  Example:\n"
          "    The following example saves the image, no text, to output.png.\n\n"
          "        python3 detector.py image.png -save -no-text\n\n"
          "  Example:\n"
          "    The following example shows the image and saves the text file to some filename, no image.\n\n"
          "        python3 detector.py image.png -show -save -no-image -tfile somefilename.txt\n"
          )

class Commands:
    def __init__(self, cmd : str, inputs : int):
        self.cmd = cmd
        self.inputs = inputs

colors = {
    'c0' : [255, 0  , 0  ],
    'c1' : [0  , 255, 0  ],
    'c2' : [0  , 0  , 255],
    'c3' : [255, 255, 0  ],
    'c4' : [255, 0  , 255],
    'c5' : [0  , 255, 255],
    'c6' : [255, 255, 255]
}

commands = [
    Commands("-save", 0),
    Commands("-ifile", 1),
    Commands("-tfile", 1),
    Commands("-show", 0),
    Commands("-model", 1),
    Commands("-help", 0),
    Commands("-no-text", 0),
    Commands("-no-image", 0),
    Commands("-overlay-good-pin", 1),
    Commands("-overlay-bent-pin", 1),
    Commands("-overlay-package", 1),
    Commands("-overlay-text", 1)
]

def is_a_command(txt)->bool:
    for cmd in commands:
        if txt == cmd.cmd:
            return True
    return False

def num_parameters(txt)->bool:
    for cmd in commands:
        if txt == cmd.cmd:
            return cmd.inputs
    return 0

class Detector:
    def __init__(self)->None:
        self.command_tree = {}

        self.active_commands = {
            '-model'    : default_model,
            '-show'     : False,
            '-save'     : False,
            '-ifile'    :  "output.png",
            '-tfile'    : "output.txt",
            '-no-image' : False,
            '-no-text'  : False,
            '-overlay-good-pin' : None,
            '-overlay-bent-pin' : 'c0',
            '-overlay-package'  : 'c5',
            '-overlay-text'     : None
        }

    def add_to_command_tree(self, cmd : str, input : list):
        self.command_tree[cmd] = input

    def print_command_tree(self):
        print(self.command_tree)

    def __bb_core(self, image : Image.Image)->Image.Image:
        image = image.copy()

        # start of model
        bb = BoundingBox(self.active_commands['-model'])

        # compute which cl to use
        cls = []
        ccs = []
        if not (self.active_commands['-overlay-bent-pin'] is None):
            cls.append(0)
            ccs.append(colors[self.active_commands['-overlay-bent-pin']])

        if not (self.active_commands['-overlay-good-pin'] is None):
            cls.append(1)
            ccs.append(colors[self.active_commands['-overlay-good-pin']])

        if not (self.active_commands['-overlay-package'] is None):
            cls.append(2)
            ccs.append(colors[self.active_commands['-overlay-package']])

        if not (self.active_commands['-overlay-text'] is None):
            cls.append(3)
            ccs.append(colors[self.active_commands['-overlay-text']])

        print(cls, ccs)
        overlay = bb.overlay(image, 10, cls, ccs)

        return overlay
    
    def __text_core(self, image : Image.Image)->list:
        pe = PackageExtractor(self.active_commands['-model'])
        cropped = pe.extract_package(image)

        # apply OCR here
        return []

    def viewer(self, overlay : Image.Image, path : str, text : str | list)->None:
        if not self.active_commands['-show']:
            return
        
        plt.figure(figsize=[10, 5])
        plt.imshow(overlay)
        plt.title(f"Image from file {path}")
        plt.show()

    def save_text(text : str | list, path : str)->None:
        if isinstance(text, str):
            text = [text]

        with open(path, "w") as file:
            for t in text:
                file.write(t + "\n")

    def saver(self, overlay : Image.Image, text : str | list, image_save_path : str, text_save_path : str)->None:
        if not self.active_commands['-save']:
            return
        
        if not self.active_commands['-no-image']:
            overlay.save(image_save_path)

        if not self.active_commands['-no-text']:
            self.save_text(text, text_save_path)

    def parser(self):
        # if overlay commands  are in command tree, set all overlays to None
        for v in ['-overlay-good-pin', '-overlay-bent-pin', '-overlay-package', '-overlay-text']:
            if v in self.command_tree:
                self.active_commands['-overlay-good-pin'] = None
                self.active_commands['-overlay-bent-pin'] = None
                self.active_commands['-overlay-package'] = None
                self.active_commands['-overlay-text'] = None

        for key, value in self.command_tree.items():
            if len(value) == 0:
                self.active_commands[key] = True
            else:
                if value[0] in ["None", "none"]:
                    self.active_commands[key] = None
                else:
                    self.active_commands[key] = value[0]

        if len(self.command_tree) == 0: # default behavior if no commands are given
            self.active_commands['-show'] = True

    def execute(self, path : str):
        self.parser()

        image = Image.open(path)
        overlay = self.__bb_core(image)
        text    = self.__text_core(image)

        self.viewer(overlay, path, text)
        self.saver (overlay, text, self.active_commands['-ifile'], self.active_commands['-tfile'])

if __name__ == '__main__':
    # cmdline parser
    args = sys.argv

    # if it is less than 2 arguments,
    # then return the help menu to the user
    if len(args) < 2:
        help()
        exit(-1)

    filename = args[1]

    de = Detector()
    i = 2
    while i < len(args):
        v = args[i]
        if is_a_command(v):
            cmd = v
            i  += 1 # everytime we get a command or parameter, we increment by 1

            num_inputs = num_parameters(cmd)
            inputs = []
            for j in range(num_inputs):
                inputs.append(args[i])
                i += 1
            
            de.add_to_command_tree(cmd, inputs)
        else:
            print(f"{v} is not a valid command...")
            print("Use -help to see help menu")
            exit(-1)

    de.execute(filename)        # run program