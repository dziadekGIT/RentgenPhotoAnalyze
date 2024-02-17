"""
@TODO
"""
import os
import warnings
import json
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from scipy.optimize import curve_fit 
from scipy.signal import find_peaks
from scipy.optimize import OptimizeWarning
from multiprocessing import Process, Queue
from multiprocessing import Pool

def get_number_of_cores():
    try:
        #Linux/Unix
        num_cores = os.cpu_count()
        if num_cores is None:
            raise NotImplementedError
    except (AttributeError, NotImplementedError):
        #Windows
        try:
            num_cores = multiprocessing.cpu_count()
        except (AttributeError, NotImplementedError):
            num_cores = None

    return num_cores

def picture_size():
    size = Image.open('../choise_temp/choise.jpg').size
    return size
    
def read_images():
    """
    @TODO
    """
    path = '../datasets/'
    images = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(path, filename)
            image = Image.open(image_path, 'r')
            images.append(image)
    return images        


def background_analyse(x: int, y: int, images_data)->int:
    """
    @TODO
    """
    images = images_data
    background_data = []
    for image in images:
        pixel = image.getpixel((x, y))
        background_data.append(pixel)
    background_cutout = np.mean(background_data)
    
    return background_cutout


def analyse_image(images_data, clk_location, bg_cutout):
    """
    @TODO
    """
    clicked_location = clk_location
    background_cutout = bg_cutout
    images = images_data
    
    peak = 0
    pixel_data = []
    if clicked_location: 
        for image in images:
            pixel = image.getpixel((clicked_location[0][0], clicked_location[0][1]))
            pixel_data.append(pixel)
    
    average_contrasts = []
    for pixel_contrasts in pixel_data:
        average_contrasts.append(pixel_contrasts)
        # print(pixel_contrasts)
    
    

    #PLOT GAUSSA
    y_values = np.array(average_contrasts)
    x_values = np.arange(len(y_values))

    num_gaussians = 1
    def multiple_gauss(x, *params):
        """
        @TODO
        """
        result = 0
        for i in range(num_gaussians):
            A = params[i * 3]
            mu = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            result += A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + background_cutout
        return result
    
    # Początkowe wartości parametrów dla  dzwonów Gaussa
    initial_guess = [1.0, 110, 10, 1.0, 50, 10, 1.0, 60, 10, 1.0, 70, 10, 1.0, 80, 10, 1.0, 90, 10, 1.0, 100, 10, 1.0, 110, 10]
    
    fwhms = []
    parameters = None
    # Dopasowanie funkcji sumy  dzwonów Gaussa do danych
    try:
        parameters, covariance = curve_fit(multiple_gauss, x_values, y_values, p0=initial_guess, maxfev=1500000)
    except RuntimeError:
        print("Optimal parameters not found. Returning 0.")
        fwhms.append(0)
    if parameters is not None:
        # Wyciągnięcie dopasowanych parametrów
        fit_params = parameters
        x_fit = np.linspace(min(x_values), max(x_values), 1000)
        y_fit = multiple_gauss(x_fit, *fit_params)

        
        # Obliczenie FWHM dla dzwonu Gaussa z pominięciem tła
        try:
            peaks, _ = find_peaks(y_fit)
            peak = np.max(average_contrasts)
        except UnboundLocalError:  
            fwhms.append(0)  

    if peak <= 90:
        fwhms.append(0)
    else:
        for i in range(num_gaussians):
            A = fit_params[i * 3]
            mu = fit_params[i * 3 + 1]
            sigma = fit_params[i * 3 + 2]
            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            if fwhm < 1 :
                fwhms.append(0)
            elif fwhm > 50:    
                fwhms.append(0)
            else: fwhms.append(fwhm)    
            
    warnings.filterwarnings("ignore", category=OptimizeWarning)      
    return fwhms


def function_for_process(start_x, start_y, inner_area, images, background_cutout, queue):
    analyzed_area = []
    for x in range(start_x, start_x + inner_area[0]):
        for y in range(start_y, start_y + inner_area[1]): 
            location = [(x, y)]
            result = analyse_image(images_data=images, clk_location=location, bg_cutout=background_cutout)        
            analyzed_area.append(result)
    queue.put(analyzed_area)

def divide_regions_to_analyze(size, cores):
    """
    @TODO
    """
    width, height = size
    region_width = width // cores

    start_xs = []
    for i in range(cores):
        start_col = i * region_width
        region_2 = (start_col)
        start_xs.append(region_2)

    return start_xs

if __name__ == "__main__":
   
    if True:
        cores = get_number_of_cores()

    if True:
        images = read_images()

    if True:
        background_cutout = background_analyse(1,1,images_data=images)
        size = picture_size()

    if True:
        # start_xs = divide_regions_to_analyze(size=size, cores=cores) 
        start_xs = [i for i in range(32)]
        print(start_xs)


    if True:
        inner_area = (start_xs[1],size[1])
        print(inner_area)
        start_y = 0
        print(len(start_xs))
           
    if True:
        data_queue = Queue()
        processes = []

        with Pool(processes=cores) as pool:
            for i in range(cores):
                start_x = start_xs[i]
                end_x = start_xs[i + 1] if i < cores - 1 else size[0]
                region_width = end_x - start_x
                inner_area = (region_width, size[1])

                process = Process(
                    target=function_for_process,
                    args=(start_x, start_y, inner_area, images, background_cutout, data_queue)
                )
                processes.append(process)

            for process in processes:
                process.start()

            for process in processes:
                process.join()
   
        results = [data_queue.get() for _ in processes]





    if True:
        manager = multiprocessing.Manager()
        analyzed_data_shared = manager.list()

        for i, result in enumerate(results):
            for j, row in enumerate(result):
                x_position = start_xs[i] + j  # Pozycja x na obrazie
                y_position = start_y  # Pozycja y na obrazie
                analyzed_data_shared.append({"x": x_position, "y": y_position, "fwhms": row})

        with open('output_mt_ordered.json', 'w', encoding="utf-8") as file_2:
            json.dump(list(analyzed_data_shared), file_2)
        # active_processes = multiprocessing.active_children()
        # num_active_processes = len(active_processes)
        # print(num_active_processes) 
        