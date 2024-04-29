# -*- coding: utf-8 -*-
"""
A simple GUI wrapper for the cal_readnoise function.

Uses TkinterDnD2 by Philippe Gagné for drag-and-drop functionality. Install with
pip install tkinterdnd2

Image processing based on NanoImagingPack
https://gitlab.com/bionanoimaging/nanoimagingpack/

"""
import os
import pathlib
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import NanoImagingPack as nip

import tkinter
import tkinter.messagebox
import tkinter.filedialog
from tkinterdnd2 import *
try:
    from Tkinter import *
    from ScrolledText import ScrolledText
except ImportError:
    from tkinter import *
    from tkinter.scrolledtext import ScrolledText

ABOUT_MESSAGE = """Calibration tool for inhomogenous image stacks

Written by David McFadden, FSU Jena

NanoImagingPack library: https://gitlab.com/bionanoimaging/nanoimagingpack/

Questions, bugs and requests to: david.mcfadden777@gmail.com
heintzmann@gmail.com
"""

root = TkinterDnD.Tk()
root.withdraw()
root.title('Inhomogenous Stack Calibration')
root.grid_rowconfigure(2, weight=1, minsize=250)
root.grid_columnconfigure(0, weight=1, minsize=300)
root.grid_columnconfigure(1, weight=1, minsize=300)

def print_event_info(event):
    print('\nAction:', event.action)
    print('Supported actions:', event.actions)
    print('Mouse button:', event.button)
    print('Type codes:', event.codes)
    print('Current type code:', event.code)
    print('Common source types:', event.commonsourcetypes)
    print('Common target types:', event.commontargettypes)
    print('Data:', event.data)
    print('Event name:', event.name)
    print('Supported types:', event.types)
    print('Modifier keys:', event.modifiers)
    print('Supported source types:', event.supportedsourcetypes)
    print('Operation type:', event.type)
    print('Source types:', event.sourcetypes)
    print('Supported target types:', event.supportedtargettypes)
    print('Widget:', event.widget, '(type: %s)' % type(event.widget))
    print('X:', event.x_root)
    print('Y:', event.y_root, '\n')

def process(exportpath=None):
    # private functions
    def stack_frompaths(fullpaths):
        stack = []
        for file in fullpaths:
            img = tifffile.imread(file).squeeze()
            err_msg = "If input consists of multiple files, each must be a 2D image."
            try:
                assert img.ndim == 2
            except:
                if GUI:
                    tkinter.messagebox.showinfo("Shape Error", err_msg)
                print(err_msg)
                return
            stack.append(img[:])
        stack = np.array(stack)
        return stack

    for box in (bg_box, fg_box):
        fullpaths = box.get(0,box.size())
        fullpaths = [pathlib.Path(f) for f in fullpaths]
        fullpaths.sort()

        # debug
        # if box is bg_box:
        #     fullpaths = list(pathlib.Path(r"").glob("*.tiff"))
        # else:
        #     fullpaths = list(pathlib.Path(r"").glob("*.tiff"))

        if len(fullpaths) == 0: #error
            err_msg = "File list is empty"
            if GUI:
                tkinter.messagebox.showinfo("Error", err_msg)
            return
        elif len(fullpaths) == 1: #could be dir or 3D file
            fullpath = fullpaths[0]
            if fullpath.is_dir():
                files = list(fullpath.glob("*"))
                file_varieties = len(set([a.suffix for a in files]))
                try:
                    assert file_varieties == 1
                except:
                    err_msg = "Directory must contain a single file type"
                    if GUI:
                        tkinter.messagebox.showinfo("More than one file type", err_msg)
                    print(err_msg)
                    return
                stack = stack_frompaths(files)
            else:
                try:
                    img = tifffile.imread(fullpath).squeeze()
                    # img = nip.readim(str(fullpath)).squeeze()
                except tifffile.tifffile.TiffFileError as e:
                    print(e, "trying numpy.load")
                    img = np.load(fullpath).squeeze()
                try:
                    assert img.ndim == 3, err_msg
                except:
                    err_msg = "If input is a single file, it must be a 3D stack."
                    if GUI: tkinter.messagebox.showinfo("Shape Error", err_msg)
                    print(err_msg)
                    return
                stack = img
        else: # needs to be list of files
            stack = stack_frompaths(fullpaths)
            # print("Stack shape: ", stack.shape)

        skip_first_str = skip_first_var.get().strip()
        if skip_first_str == "":
            initial_index = 0
        else:
            initial_index = int(skip_first_str)
        box.stack = stack.astype(float)[initial_index:]   
        
    # fit range

    validRange = "empty"
    for var in [lower_bound_var, upper_bound_var]:
        varstr = var.get().strip()
        if varstr=="":
            validRange = None
            break
        try:
            var.value = float(varstr)
        except:
            err_msg = "Incorrect format for fit range."
            if GUI: tkinter.messagebox.showinfo("Range Error", err_msg)
            print(err_msg)
            return
        
    if validRange == "empty":
        validRange = (lower_bound_var.value, upper_bound_var.value)

    # same for hist range

    histRange = "empty"
    for var in [hist_lower_bound_var, hist_upper_bound_var]:
        varstr = var.get().strip()
        if varstr=="":
            histRange = None
            break
        try:
            var.value = float(varstr)
        except:
            err_msg = "Incorrect format for histogram range."
            if GUI: tkinter.messagebox.showinfo("Range Error", err_msg)
            print(err_msg)
            return
        
    if histRange == "empty":
        histRange = (hist_lower_bound_var.value, hist_upper_bound_var.value)

    # hist bins
    hist_bins_str = hist_bins_var.get().strip()
    if hist_bins_str == "":
        numBins = 100
    else:
        numBins = int(hist_bins_str)

    exportFormat = format_var.get().lower()


    #
     #  check bg is not implemented 
    # try:
    #     raise AssertionError
    #     (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, Text) = nip.cal_readnoise(
    #         fg_box.stack, bg_box.stack, validRange=validRange, histRange=histRange, numBins=numBins, exportpath=exportpath, exportFormat=exportFormat,
    #         brightness_blurring=scmos_filter_var.get(), plotHist=showHist_var.get(), saturationImage=saturationImage_var.get())
    # except AssertionError as err:
    #     err_msg = err
    #     if GUI: tkinter.messagebox.showinfo("Failed analysis", err_msg)
    #     print(err_msg)
    #     # tkinter.askyesno("Foo", "Bar")
    (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, Text) = nip.cal_readnoise(
        fg_box.stack, bg_box.stack, validRange=validRange, histRange=histRange, numBins=numBins, exportpath=exportpath, exportFormat=exportFormat,
        brightness_blurring=scmos_filter_var.get(), plotHist=showHist_var.get(), saturationImage=saturationImage_var.get(), check_bg=False)        
        # return




    # Chunking evaluation
    def getint_fromvar(strvar, default=None):
        var_str = strvar.get().strip()
        if var_str == "":
            val = default
        else:
            val = int(var_str)
        return val
    xchunks = getint_fromvar(xchunks_var, default=1)
    ychunks = getint_fromvar(ychunks_var, default=1)

    plt.ioff()

    if (xchunks != 1) or (ychunks != 1):
        gain_profile = np.empty((ychunks, xchunks))
        fg, bg = fg_box.stack, bg_box.stack
        # with array_split, we don't need to worry about the array not being divisible
        print(fg.shape)

        exportpath_parent = exportpath
        for ix, (fgxc, bgxc) in enumerate(zip(np.array_split(fg, xchunks, axis=-1), np.array_split(bg, xchunks, axis=-1))):
            print(fgxc.shape)
            for iy, (fgyc, bgyc) in enumerate(zip(np.array_split(fgxc, ychunks, axis=-2), np.array_split(bgxc, ychunks, axis=-2))):
                
                if exportpath_parent is not None:
                    exportpath = pathlib.Path(exportpath_parent/f"chunk_X{ix}Y{iy}")
                    exportpath.mkdir(parents=True, exist_ok=True)
                else:
                    exportpath = None
                (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, Text) = nip.cal_readnoise(
                    fgyc, bgyc, validRange=validRange, histRange=histRange, numBins=numBins, exportpath=exportpath, exportFormat=exportFormat,
                    brightness_blurring=scmos_filter_var.get(), plotHist=showHist_var.get())
                gain_profile[iy, ix] = gain
                # Debug: save the image
                # nip.imsave(nip.image(fgyc.mean((0))), str(exportpath/"sub_image.tif"))
        if exportpath_parent is not None:
            np.savetxt(exportpath_parent/"gain_profile.csv", gain_profile)

    plt.ion()
    plt.show()
    return

def process_and_save():
    output_dir = pathlib.Path(tkinter.filedialog.askdirectory())
    process(output_dir)
    return

def clear_boxes():
    bg_box.delete(0,bg_box.size())
    fg_box.delete(0,fg_box.size())

def close_plots():
    plt.close("all")


def about():
    tkinter.messagebox.showinfo("About", ABOUT_MESSAGE)

Label(root, text='Drag and drop image files. These must either be single 3D stacks or a sequence of 2D images.').grid(
                    row=0, columnspan=2, column=0, padx=10, pady=5)
Label(root, text='Dark images:').grid(
                    row=1, column=0, padx=10, pady=5)
Label(root, text='Bright images:').grid(
                    row=1, column=1, padx=10, pady=5)
button_frame = Frame(root)
button_frame.grid(row=3, column=0, columnspan=2, pady=5)
button_frame_line2 = Frame(root)
button_frame_line2.grid(row=4, column=0, columnspan=2, pady=5)

advanced_frame = LabelFrame(root, text="Advanced options")
advanced_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky="we")

# ------------------------------------------------------------------
# Button_frame
# ------------------------------------------------------------------
scmos_filter_var = IntVar(value=True)
Checkbutton(button_frame, text="sCMOS filter", variable=scmos_filter_var).pack(
                    side=LEFT, padx=5)
saturationImage_var = IntVar()
Checkbutton(button_frame, text="Saturation Image", variable=saturationImage_var).pack(
                    side=LEFT, padx=5)
showHist_var = IntVar()
Checkbutton(button_frame, text="Show histogram", variable=showHist_var).pack(
                    side=LEFT, padx=5)

Label(button_frame, text='Skip first').pack(
                    side=LEFT, padx=5)
skip_first_var = StringVar()
tkinter.Spinbox(button_frame, from_=0, to_=9999, textvariable=skip_first_var, width=5).pack(
                    side=LEFT, padx=5)

# ------------------------------------------------------------------
# Button_frame line 2
# ------------------------------------------------------------------
Label(button_frame_line2, text='Fit from').pack(
                    side=LEFT, padx=5)
lower_bound_var = StringVar()
Entry(button_frame_line2, textvariable=lower_bound_var, width=5).pack(
                    side=LEFT, padx=5)
Label(button_frame_line2, text='to').pack(
                    side=LEFT, padx=5)  
upper_bound_var = StringVar()                                      
Entry(button_frame_line2, textvariable=upper_bound_var, width=5).pack(
                    side=LEFT, padx=5)

Label(button_frame_line2, text='Histogram from').pack(
                    side=LEFT, padx=5)
hist_lower_bound_var = StringVar()
Entry(button_frame_line2, textvariable=hist_lower_bound_var, width=5).pack(
                    side=LEFT, padx=5)
Label(button_frame_line2, text='to').pack(
                    side=LEFT, padx=5)  
hist_upper_bound_var = StringVar()                                      
Entry(button_frame_line2, textvariable=hist_upper_bound_var, width=5).pack(
                    side=LEFT, padx=5)

Label(button_frame_line2, text='Histogram bins').pack(
                    side=LEFT, padx=5)
hist_bins_var = StringVar(value=100)
tkinter.Spinbox(button_frame_line2, from_=10, to_=9999, textvariable=hist_bins_var, width=5).pack(
                    side=LEFT, padx=5)

# ------------------------------------------------------------------
# Processing buttons
# ------------------------------------------------------------------
Button(button_frame, text='Clear file list', command=clear_boxes).pack(
                    side=LEFT, padx=5)
Button(button_frame, text='Process', command=process).pack(
                    side=LEFT, padx=5)
format_var = StringVar()
# Dictionary with options
format_choices = { 'PNG','SVG'}
format_var.set('PNG') # set the default option
OptionMenu(button_frame, format_var, *format_choices).pack(
                    side=LEFT, padx=5)  

Button(button_frame, text='Process and save', command=process_and_save).pack(
                    side=LEFT, padx=5)
Button(button_frame, text='Close plots', command=close_plots).pack(
                    side=LEFT, padx=5)
Button(button_frame, text='Quit', command=root.quit).pack(
                    side=LEFT, padx=5)

Button(button_frame, text='About', command=about).pack(
                    side=LEFT, padx=10)

# ------------------------------------------------------------------
# Advanced_frame
# ------------------------------------------------------------------

Label(advanced_frame, text='X Chunks').pack(
                    side=LEFT, padx=5)
xchunks_var = StringVar()
Entry(advanced_frame, textvariable=xchunks_var, width=5).pack(
                    side=LEFT, padx=5)                                      

Label(advanced_frame, text='Y Chunks').pack(
                    side=LEFT, padx=5)
ychunks_var = StringVar()
Entry(advanced_frame, textvariable=ychunks_var, width=5).pack(
                    side=LEFT, padx=5)                                      
# ------------------------------------------------------------------
# Boxes
# ------------------------------------------------------------------
bg_box = Listbox(root, name='bg_box',
                    selectmode='extended', width=1, height=1)
bg_box.grid(row=2, column=0, padx=5, pady=5, sticky='news')

fg_box = Listbox(root, name='fg_box',
                    selectmode='extended', width=1, height=1)
fg_box.grid(row=2, column=1, padx=5, pady=5, sticky='news')



# bg_box.insert(END, os.path.abspath(__file__))

# Drop callbacks can be shared between the Listbox and Text;
# according to the man page these callbacks must return an action type,
# however they also seem to work without

def drop_enter(event):
    event.widget.focus_force()
    print('Entering widget: %s' % event.widget)
    #print_event_info(event)
    return event.action

def drop_position(event):
    # print('Position: x %d, y %d' %(event.x_root, event.y_root))
    #print_event_info(event)
    return event.action

def drop_leave(event):
    print('Leaving %s' % event.widget)
    #print_event_info(event)
    return event.action

def drop(event):
    if event.data:
        print('Dropped data:\n', event.data)
        #print_event_info(event)
        # event.data is a list of filenames as one string;
        # if one of these filenames contains whitespace characters
        # it is rather difficult to reliably tell where one filename
        # ends and the next begins; the best bet appears to be
        # to count on tkdnd's and tkinter's internal magic to handle
        # such cases correctly; the following seems to work well
        # at least with Windows and Gtk/X11

        file_box = event.widget
        files = file_box.tk.splitlist(event.data)
        for f in files:
            if os.path.exists(f):
                print('Dropped file: "%s"' % f)
                file_box.insert('end', f)
            else:
                print('Not dropping file "%s": file does not exist.' % f)
    return event.action

# now make the Listbox and Text drop targets
bg_box.drop_target_register(DND_FILES, DND_TEXT)
fg_box.drop_target_register(DND_FILES, DND_TEXT)
# text.drop_target_register(DND_TEXT)

for widget in (bg_box, fg_box):
    widget.dnd_bind('<<DropEnter>>', drop_enter)
    widget.dnd_bind('<<DropPosition>>', drop_position)
    widget.dnd_bind('<<DropLeave>>', drop_leave)
    widget.dnd_bind('<<Drop>>', drop)
    #widget.dnd_bind('<<Drop:DND_Files>>', drop)
    #widget.dnd_bind('<<Drop:DND_Text>>', drop)

# define drag callbacks

def drag_init_listbox(event):
    print_event_info(event)
    # use a tuple as file list, this should hopefully be handled gracefully
    # by tkdnd and the drop targets like file managers or text editors
    data = ()
    file_box = event.widget
    if file_box.curselection():
        data = tuple([file_box.get(i) for i in file_box.curselection()])
        print('Dragging :', data)
    # tuples can also be used to specify possible alternatives for
    # action type and DnD type:
    return ((ASK, COPY), (DND_FILES, DND_TEXT), data)


def drag_end(event):
    #print_event_info(event)
    # this callback is not really necessary if it doesn't do anything useful
    print('Drag ended for widget:', event.widget)

# finally make the widgets a drag source

bg_box.drag_source_register(1, DND_TEXT, DND_FILES)
fg_box.drag_source_register(1, DND_TEXT, DND_FILES)

bg_box.dnd_bind('<<DragInitCmd>>', drag_init_listbox)
bg_box.dnd_bind('<<DragEndCmd>>', drag_end)

fg_box.dnd_bind('<<DragInitCmd>>', drag_init_listbox)
fg_box.dnd_bind('<<DragEndCmd>>', drag_end)
# skip the useless drag_end() binding for the text widget

root.update_idletasks()
root.deiconify()
root.mainloop()

