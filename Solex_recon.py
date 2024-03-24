# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
contributors: Valerie Desnoux, Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 24 September 2023

------------------------------------------------------------------------
Reconstruction of an image from the deviations between the minimum of the line and a reference line
-------------------------------------------------------------------------

"""

from solex_util import *
from video_reader import *
from ellipse_to_circle import ellipse_to_circle, correct_image
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import PySimpleGUI as sg # for progress bar
from scipy.ndimage import gaussian_filter1d

'''
process files: call solex_read and solex_proc to process a list of files with specified options
input: tasks: list of tuples (file, option)
'''

def solex_do_work(tasks, flag_command_line = False):
    multi = True
    if not multi:
        print("WARNING: multithreading is off")
    with Pool(4) as p:
        results = []
        for i, (file, options) in enumerate(tasks):
            print('file %s is processing'%file)
            if len(tasks) > 1 and not flag_command_line:  
                sg.one_line_progress_meter('Progress Bar', i, len(tasks), '','Reading file...')
            disk_list, backup_bounds, hdr = solex_read(file, options)
            if multi:
                result = p.apply_async(solex_process, args = (options, disk_list, backup_bounds, hdr)) # TODO: prints won't be visible inside new thread, can this be fixed?
                results.append(result)
            else:
                solex_process(options, disk_list, backup_bounds, hdr)
        [result.get() for result in results]
        if len(tasks) > 1 and not flag_command_line:  
            sg.one_line_progress_meter('Progress Bar', len(tasks), len(tasks), '','Done.')
        
'''
read a solex file and return a list of numpy arrays representing the raw result
'''
def solex_read(file, options):
    basefich0 = os.path.splitext(file)[0] # file name without extension
    options['basefich0'] = basefich0
    clearlog(basefich0 + '_log.txt', options)
    logme(basefich0 + '_log.txt', options, 'Pixel shift : ' + str(options['shift']))
    options['shift_requested'] = options['shift']
    options['shift'] = list(dict.fromkeys([options['ellipse_fit_shift'], 0] + options['shift']))  # options['ellipse_fit_shift'], 0 are "fake", but if they are requested, then don't double count
    rdr = video_reader(file)
    hdr = make_header(rdr)
    ih = rdr.ih
    iw = rdr.iw

    mean_img, fit, backup_y1, backup_y2 = compute_mean_return_fit(video_reader(file), options, hdr, iw, ih, basefich0)

    disk_list, ih, iw, FrameCount = read_video_improved(video_reader(file), fit, options)
    
    hdr['NAXIS1'] = iw  # note: slightly dodgy, new width for subsequent fits file


    
    # sauve fichier disque reconstruit

    if options['flag_display']:
        cv2.destroyAllWindows()

    for i in range(len(disk_list)):
        if options['flip_x']:
            disk_list[i] = np.flip(disk_list[i], axis = 1)
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
        flag_requested = options['shift'][i] in options['shift_requested']
        
        if options['save_fit'] and flag_requested:
            DiskHDU = fits.PrimaryHDU(disk_list[i], header=hdr)
            DiskHDU.writeto(output_path(basefich + '_raw.fits', options), overwrite='True')
    return disk_list, (backup_y1, backup_y2), hdr
    
'''
process the raw disks: circularise, detransversalium, crop, and adjust contrast

inputs: disk_list : list of images as np arrays
backup_bounds: tuple of numbers for disk upper and lower bounds (backup for case of no ellipse-fit)
hdr: an hdr header for fits files

'''
def solex_process(options, disk_list, backup_bounds, hdr):
    basefich0 = options['basefich0']
    if options['transversalium']:
        logme(basefich0 + '_log.txt', options, 'Transversalium correction : ' + str(options['trans_strength']))
    else:
        logme(basefich0 + '_log.txt', options, 'Transversalium disabled')
    logme(basefich0 + '_log.txt', options, 'Mirror X : ' + str(options['flip_x']))
    logme(basefich0 + '_log.txt', options, 'Post-rotation : ' + str(options['img_rotate']) + ' degrees')
    logme(basefich0 + '_log.txt', options, f'Protus adjustment : {options["delta_radius"]}')
    logme(basefich0 + '_log.txt', options, f'de-vignette : {options["de-vignette"]}')
    borders = [0,0,0,0]
    cercle0 = (-1, -1, -1)
    for i in range(len(disk_list)):
        flag_requested = options['shift'][i] in options['shift_requested']
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
        """
        We now apply ellipse_fit to apply the geometric correction

        """
        # disk_list[0] is always shift = 10, for more contrast for ellipse fit
        if options['ratio_fixe'] is None and options['slant_fix'] is None:
            frame_circularized, cercle0, options['ratio_fixe'], phi, borders = ellipse_to_circle(
                disk_list[i], options, basefich)
            # in options angles are stored as degrees (slightly annoyingly)
            options['slant_fix'] = math.degrees(phi)

        else:
            ratio = options['ratio_fixe'] if not options['ratio_fixe'] is None else 1.0
            phi = math.radians(options['slant_fix']) if not options['slant_fix'] is None else 0.0
            if flag_requested:
                frame_circularized = correct_image(disk_list[i] / 65536, phi, ratio, np.array([-1.0, -1.0]), -1.0, options, print_log=i == 0)[0]  # Note that we assume 16-bit
                if options['de-vignette']:
                    if cercle0 == (-1, -1, -1):
                        print("WARNING: cannot de-vignette without ellipse fit")
                    else:
                        frame_circularized = removeVignette(frame_circularized, cercle0)
        if not flag_requested:
            continue # skip processing if shift is not desired
        
        single_image_process(frame_circularized, hdr, options, cercle0, borders, basefich, backup_bounds)
        write_complete(basefich0 + '_log.txt', options)


def single_image_process(frame_circularized, hdr, options, cercle0, borders, basefich, backup_bounds):
    if options['save_fit']:  # first two shifts are not user specified
        DiskHDU = fits.PrimaryHDU(frame_circularized, header=hdr)
        DiskHDU.writeto(output_path(basefich + '_circular.fits', options), overwrite='True')


    if options['transversalium']:
        if not cercle0 == (-1, -1, -1):
            detransversaliumed = correct_transversalium2(frame_circularized, cercle0, borders, options, 0, basefich)
        else:
            detransversaliumed = correct_transversalium2(frame_circularized, (0,0,99999), [0, backup_bounds[0]+20, frame_circularized.shape[1] -1, backup_bounds[1]-20], options, 0, basefich)
    else:
        detransversaliumed = frame_circularized

    if options['save_fit'] and options['transversalium']:  # first two shifts are not user specified
        DiskHDU = fits.PrimaryHDU(detransversaliumed, header=hdr)
        DiskHDU.writeto(output_path(basefich + '_detransversaliumed.fits', options), overwrite='True')

    cercle = cercle0
    if not options['fixed_width'] == None or options['crop_width_square']:
        h, w = detransversaliumed.shape
        nw = h if options['fixed_width'] == None else options['fixed_width'] # new width
        nw2 = nw // 2
        cx = w // 2 if cercle == (-1, -1, -1) else int(cercle[0])
        tx = nw2 - cx
        new_img = np.full((h, nw), detransversaliumed[0, 0], dtype=detransversaliumed.dtype)

        new_img[:, :min(cx + nw2, detransversaliumed.shape[1]) - max(0, cx - nw2)] = detransversaliumed[:, max(0, cx - nw2) : min(cx + nw2, detransversaliumed.shape[1])]
        
        if tx > 0:
            new_img = np.roll(new_img, tx, axis = 1)
            new_img[:, :tx] = detransversaliumed[0, 0]

        if not cercle == (-1, -1, -1):
            cercle = (nw2, cercle[1], cercle[2])
        detransversaliumed = new_img    
        
        #--- MattC start of resize code
        #https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
        def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
            img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                     -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)

            y2 += -min(0, y1)
            y1 += -min(0, y1)
            x2 += -min(0, x1)
            x1 += -min(0, x1)
            return img, x1, x2, y1, y2

        def imcrop(img, bbox):
            x1, y1, x2, y2 = bbox
            if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
            return img[y1:y2, x1:x2]

        #if options['radiusgoal'] > 0 :
        if 1: #force this to be done to test.  Want a 900 pixel radius solar disc AND 2400 pixel wide canvas
            print("Circle ",cercle)
            scalefactor = 900/ cercle[2] #options['radiusgoal'] / cercle[2]
            new_frame_width = int(detransversaliumed.shape[1] * scalefactor)
            new_frame_height = int(detransversaliumed.shape[0] * scalefactor)
            new_frame_dimension = (new_frame_width, new_frame_height)
            y = list(cercle)
            y[2] = 900 #options['radiusgoal']
            y[1] = int(y[1]*scalefactor)
            y[0] = int(y[0]*scalefactor)
            cercle = tuple(y)
            #detransversaliumed, cercle0, options['ratio_fixe'], phi, borders = ellipse_to_circle(detransversaliumed, options, basefich)
            detransversaliumed = cv2.resize(detransversaliumed, new_frame_dimension, interpolation=cv2.INTER_CUBIC)
            #desired upper left x1,y1, lower right x2,y2
            roi = ( int(new_frame_width/2 - (2*900+600)/2), 0, #could use existing width field in UI???
                    int(new_frame_width/2 + (2*900+600)/2), new_frame_height)
            #print("DT shape ",detransversaliumed.shape)
            #print("ROI ",roi)
            detransversaliumed = imcrop(detransversaliumed,roi)
            if options['save_fit'] :
                DiskHDU = fits.PrimaryHDU(detransversaliumed, header=hdr)
                DiskHDU.writeto(basefich + '_dtresized.fits', overwrite='True')
            #print(basefich+'_dtresized.png')
            cv2.imwrite(basefich+'_dtresized.png',detransversaliumed) #want this for hugin stitching and exposure balancing
        #--- MattC end of resize code

    return image_process(detransversaliumed, cercle, options, hdr, basefich)
