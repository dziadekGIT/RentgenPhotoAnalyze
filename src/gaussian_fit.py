"""
@TODO
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit 
from scipy.signal import find_peaks
from scipy.optimize import OptimizeWarning
import pandas as pd
import warnings
import json
from multiprocessing import pool


def clicked_location():
    """
    @TODO
    """
    
    clicked_location = []
    def click(event):
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            clicked_location.append((x,y)) 
            plt.close()
    choise = Image.open('../choise_temp/choise.jpg')   
    plt.imshow(choise)
    
    plt.title("Click on the image to get x and y coordinates")
    plt.connect('button_press_event', click)
    plt.show()
    print(clicked_location)

    return clicked_location

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


# def timing(f):
#     def measure(*args, **kwargs):
#         start = time.time()
#         result = f(*args, **kwargs)
#         end = time.time()
#         print(f"{f.__name__} executed in {end - start} seconds")
#     return measure


# @timing
def analyse_image(images_data, clk_location, bg_cutout):
    """
    @TODO
    """
    clicked_location = clk_location
    background_cutout = bg_cutout
    images = images_data
  
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

    # Dopasowanie funkcji sumy  dzwonów Gaussa do danych
    parameters, covariance = curve_fit(multiple_gauss, x_values, y_values, p0=initial_guess, maxfev=1500000)
    
    # Wyciągnięcie dopasowanych parametrów
    fit_params = parameters
    x_fit = np.linspace(min(x_values), max(x_values), 1000)
    y_fit = multiple_gauss(x_fit, *fit_params)

    # Obliczenie FWHM dla dzwonu Gaussa z pominięciem tła
    peaks, _ = find_peaks(y_fit)
    peak = np.max(average_contrasts)

    # peak_value = np.max(y_fit[peaks])
    # print(f'Peak : {peak_value}')

    fwhms = []
    if peak > 90:
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
            
    else:  
        fwhms.append(0)  
    
    
    
    # Rysowanie wykresu
    # fc = range(0, 138)
    # fig, ax = plt.subplots(1, 2, figsize=(20, 5))   
    # ax[0].plot(fc,average_contrasts)
    # ax[0].set_ylabel("Contrast")
    # ax[0].set_title("Photo count")
  
   
    # ax[1].scatter(x_values, y_values, s=5,c='red', marker='o', label='Data')
    # ax[1].plot(x_fit, y_fit, '-', label=f'Gaussian fit\nFWHMS: {fwhms}\nPeak: {peak}')
    # ax[1].legend()
    # plt.title(f'Gaussian fit (sum of {num_gaussians})')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')

    # plt.show()
    # plt.close()
    warnings.filterwarnings("ignore", category=OptimizeWarning)
        
    return fwhms


if __name__ == "__main__":
    
    if True:
        images = read_images()
    
    if False:
        location = clicked_location()
    
    if True:
        background_cutout = background_analyse(1,1,images_data=images)
        size = picture_size()
        # print(size)    

    if False:   
        location =[(1098, 1208)]
        wynik = analyse_image(images_data=images, clk_location=location, bg_cutout=background_cutout)
        print(wynik)

    if True:
        # size = (2896, 2416)
        
        inner_area = (1, 1)
        start_x = (size[0]-inner_area[0])//2
        start_y = (size[1]-inner_area[1])//2

        # df = pd.DataFrame(columns = [f'Col_{i}' for i in range(5)])

        analyzed_area = []


        for x in range(start_x, start_x + inner_area[0]):
            for y in range(start_y, start_y + inner_area[1]): 
                location = [(x,y)]
                result = analyse_image(images_data=images, clk_location=location, bg_cutout=background_cutout)
                
                analyzed_area.append(result)
                print(analyzed_area)
        # print(df)

    if False:
        with open('output.json', 'w', encoding="utf-8") as file:
            json.dump(analyzed_area, file)
   