# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit 
# from scipy.signal import find_peaks

# num_gaussians = 1

# def background_analyse(x: int, y: int, images):
#     background_data = []
#     for image in images:
#         pixel = image.getpixel((x, y))
#         background_data.append(pixel)
#     background_cutout = np.mean(background_data)
#     return background_cutout

# def multiple_gauss(x, *params, images):
#     result = 0
#     for i in range(num_gaussians):
#         A = params[i * 3]
#         mu = params[i * 3 + 1]
#         sigma = params[i * 3 + 2]
#         result += A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + background_analyse(1, 1, images)
#     return result

# def fit_function(x, *params):
#     return multiple_gauss(x, *params, images=images)

# def analyse_image():
#     path = '../datasets/'
#     images = []
#     for filename in os.listdir(path):
#         if filename.endswith('.jpg'):
#             image_path = os.path.join(path, filename)
#             image = Image.open(image_path, 'r')
#             images.append(image)

#     width, height = images[0].width, images[0].height

#     average_contrasts = []
#     fwhms_list = []

#     for x in range(width):
#         for y in range(height):
#             pixel_data = []
#             for image in images:
#                 pixel = image.getpixel((x, y))
#                 pixel_data.append(pixel)

#             average_contrasts.append(np.mean(pixel_data))

#     y_values = np.array(average_contrasts)
#     x_values = np.arange(len(y_values))

#     initial_guess = [1.0, 110, 10, 1.0, 50, 10, 1.0, 60, 10, 1.0, 70, 10, 1.0, 80, 10, 1.0, 90, 10, 1.0, 100, 10, 1.0, 110, 10]

#     fwhms_array = np.zeros((height, width))

#     for x in range(width):
#         for y in range(height):
#             pixel_data = []
#             for image in images:
#                 pixel = image.getpixel((x, y))
#                 pixel_data.append(pixel)

#             y_values = np.array(pixel_data)

#             try:
#                 parameters, covariance = curve_fit(fit_function, x_values, y_values, p0=initial_guess, maxfev=300000)
#                 fit_params = parameters

#                 peaks, _ = find_peaks(multiple_gauss(x_values, *fit_params, images=images))
#                 peak_value = np.max(multiple_gauss(x_values, *fit_params, images=images)[peaks])
#                 fwhm = 2 * np.sqrt(2 * np.log(2)) * fit_params[2]

#                 if peak_value > 90:
#                     fwhms_array[y, x] = fwhm
#                 else:
#                     fwhms_array[y, x] = 0
#             except RuntimeError:
#                 # W przypadku, gdy dopasowanie nie powiedzie się, ustaw FWHM na 0
#                 fwhms_array[y, x] = 0

#     fwhm_image = Image.fromarray((fwhms_array * 255).astype(np.uint8))
#     fwhm_image.save('fwhm_image.jpg')

#     plt.imshow(fwhm_image, cmap='hot')
#     plt.title("FWHM Image")
#     plt.show()

# if __name__ == "__main__":
#     analyse_image()


#-------------------
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from scipy.optimize import curve_fit 
from scipy.signal import find_peaks

selected_regions = []

def analyse_image():
    # D:\_NEWHOPE\Photo_analyze\datasets
    path = '../datasets/'
    images = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(path, filename)
            image = Image.open(image_path, 'r')
            images.append(image)

    choise = Image.open('../choise_temp/choise.jpg')        
    plt.imshow(choise)
    plt.title("Select regions for analysis")

    # Wybieranie obszaru
    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_regions.append(((x1, y1), (x2, y2)))
    
    # Inicjalizacja wybierania obszaru
    rs = RectangleSelector(plt.gca(), onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    plt.show()

    # Przetwarzanie wybranych obszarów
    for region in selected_regions:
        x1, y1 = region[0]
        x2, y2 = region[1]

        average_contrasts = []
        for y in range(int(y1), int(y2)):
            for x in range(int(x1), int(x2)):
                pixel_data = []
                for image in images:
                    pixel = image.getpixel((x, y))
                    pixel_data.append(pixel)
                average_contrasts.append(np.mean(pixel_data))

        fc = range(0, 138)
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))   
        ax[0].plot(fc, np.mean(np.array(average_contrasts).reshape(-1, len(fc)), axis=0))
        ax[0].set_ylabel("Contrast")
        ax[0].set_title("Photo count")

        # PLOT GAUSSIAN
        y_values = np.array(average_contrasts)
        x_values = np.arange(len(y_values))

        num_gaussians = 1
        def multiple_gauss(x, *params):
            result = 0
            for i in range(num_gaussians):
                A = params[i * 3]
                mu = params[i * 3 + 1]
                sigma = params[i * 3 + 2]
                result += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
            return result
        
        # Initial guess for Gaussian bells
        initial_guess = [1.0, 110, 10, 1.0, 50, 10, 1.0, 60, 10, 1.0, 70, 10, 1.0, 80, 10, 1.0, 90, 10, 1.0, 100, 10, 1.0, 110, 10]

        # Fitting Gaussian sum function to data
        parameters, covariance = curve_fit(multiple_gauss, x_values, y_values, p0=initial_guess, maxfev=300000)
        
        # Extracting fitted parameters
        fit_params = parameters
        x_fit = np.linspace(min(x_values), max(x_values), 1000)
        y_fit = multiple_gauss(x_fit, *fit_params)

        # Calculating FWHM for Gaussian bell excluding background
        peaks, _ = find_peaks(y_fit)
        peak = np.max(average_contrasts)
        peak_value = np.max(y_fit[peaks])

        print(f'Peak : {peak_value}')
        fwhms = []
        if peak > 90:
            for i in range(num_gaussians):
                A = fit_params[i * 3]
                mu = fit_params[i * 3 + 1]
                sigma = fit_params[i * 3 + 2]
                fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
                fwhms.append(fwhm)
                print(f'FWHM for Gaussian {i+1}: {fwhm}')
        else: 
            print('FWHM for Gaussian : 0')  

        # Saving FWHM values to a file
        np.savetxt(f'fwhm_values_{x1}_{y1}_{x2}_{y2}.txt', fwhms)

        # Drawing the plot
        ax[1].scatter(x_values, y_values, s=5,c='red', marker='o', label='Data')
        ax[1].plot(x_fit, y_fit, '-', label=f'Gaussian fit\nFWHMS: {fwhms}\nPeak: {peak}')
        ax[1].legend()
        plt.title(f'Gaussian fit (sum of {num_gaussians})')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        plt.show()
        plt.close()

if __name__ == "__main__":
    analyse_image()
